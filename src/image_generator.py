from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)

try:
    from diffusers import FluxPipeline
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False

from .configuration import ExperimentConfig
from .prompts import PromptSet
from .utils import ensure_dir, set_seed, timestamp_now

SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpm": DPMSolverMultistepScheduler,
    "unipc": UniPCMultistepScheduler,
}


class ImageGenerator:
    def __init__(
        self,
        model_path: str,
        config: ExperimentConfig,
        model_kind: str,
        lora_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.lora_path = lora_path
        self.config = config
        self.model_kind = model_kind
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        # Determine model type based on model path
        model_path_lower = self.model_path.lower()
        is_flux = "flux" in model_path_lower
        is_sdxl = "xl" in model_path_lower and not is_flux

        # Check environment variables for memory optimization settings early
        enable_cpu_offload = os.getenv("ENABLE_MODEL_CPU_OFFLOAD", "false").lower() == "true"

        # Select appropriate dtype
        if self.config.precision == "bf16" and self.device == "cuda":
            torch_dtype = torch.bfloat16
        elif self.config.precision == "fp16" and self.device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Select pipeline class
        if is_flux:
            if not FLUX_AVAILABLE:
                raise ImportError("FluxPipeline not available. Please upgrade diffusers.")
            pipeline_cls = FluxPipeline
        elif is_sdxl:
            pipeline_cls = StableDiffusionXLPipeline
        else:
            pipeline_cls = StableDiffusionPipeline

        model_path = Path(self.model_path)
        if model_path.is_file() and model_path.suffix.lower() in {".safetensors", ".ckpt"}:
            pipeline = pipeline_cls.from_single_file(
                str(model_path),
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
        else:
            pipeline = pipeline_cls.from_pretrained(
                str(model_path),
                torch_dtype=torch_dtype,
                safety_checker=None,
            )

        # Only set scheduler for non-FLUX models (FLUX uses FlowMatchEulerDiscreteScheduler by default)
        if not is_flux:
            scheduler_name = (self.config.scheduler or "euler_a").lower()
            scheduler_cls = SCHEDULERS.get(scheduler_name)
            if scheduler_cls is not None:
                pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)

        # Load LoRA weights if provided
        if self.lora_path:
            lora_path = Path(self.lora_path)
            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA weights not found at {lora_path}")
            pipeline.load_lora_weights(str(lora_path))
            lora_scale = float(os.getenv("LORA_SCALE", "1.0"))
            if hasattr(pipeline, "fuse_lora"):
                pipeline.fuse_lora(lora_scale=lora_scale)
            elif hasattr(pipeline, "set_adapters"):
                pipeline.set_adapters(["default"], weights=[lora_scale])
            else:
                # Fall back to scaling via internal attribute if available
                if hasattr(pipeline, "lora_scale"):
                    pipeline.lora_scale = lora_scale

        # Enable memory optimizations based on model type and environment variables
        if is_flux:
            # Retrieve environment variables for VAE optimizations
            enable_vae_tiling = os.getenv("ENABLE_VAE_TILING", "false").lower() == "true"
            enable_vae_slicing = os.getenv("ENABLE_VAE_SLICING", "false").lower() == "true"

            if enable_cpu_offload:
                # Enable sequential CPU offloading for FLUX models
                # This moves model components to GPU one at a time during inference
                pipeline.enable_sequential_cpu_offload()
            else:
                pipeline = pipeline.to(self.device)

            if enable_vae_tiling:
                # Enable VAE tiling to process images in tiles (reduces VRAM usage)
                pipeline.vae.enable_tiling()

            if enable_vae_slicing:
                # Enable VAE slicing for even lower memory usage
                pipeline.vae.enable_slicing()
        else:
            pipeline = pipeline.to(self.device)
            # Non-FLUX models support attention slicing
            pipeline.enable_attention_slicing()

        return pipeline

    def run(self, prompt_set: PromptSet, output_dir: Path) -> Path:
        images_dir = ensure_dir(output_dir / "images")
        set_seed(self.config.seed)

        metadata = {
            "experiment_name": self.config.experiment_name,
            "model_path": self.model_path,
            "lora_path": self.lora_path,
            "model_kind": self.model_kind,
            "generated_at": timestamp_now(),
            "config": self.config.to_dict(),
            "prompt_set": prompt_set.as_serializable,
            "images": [],
        }

        # Check if this is a FLUX pipeline
        is_flux = FLUX_AVAILABLE and isinstance(self.pipeline, FluxPipeline)

        for prompt_entry in prompt_set.prompts:
            for image_idx in range(self.config.num_images_per_prompt):
                seed = self.config.seed + prompt_entry.index * 1000 + image_idx
                generator = torch.Generator(device=self.device).manual_seed(seed)

                # FLUX doesn't support negative_prompt parameter
                if is_flux:
                    result = self.pipeline(
                        prompt=prompt_entry.formatted,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        height=self.config.height,
                        width=self.config.width,
                        generator=generator,
                    )
                else:
                    result = self.pipeline(
                        prompt=prompt_entry.formatted,
                        negative_prompt=self.config.negative_prompt,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        height=self.config.height,
                        width=self.config.width,
                        generator=generator,
                    )
                image = result.images[0]
                file_name = f"{prompt_entry.index:02d}-{prompt_entry.slug}-i{image_idx:02d}.png"
                relative_path = Path("images") / file_name
                image.save(images_dir / file_name)
                metadata["images"].append(
                    {
                        "prompt_index": prompt_entry.index,
                        "prompt_slug": prompt_entry.slug,
                        "prompt": prompt_entry.formatted,
                        "image_index": image_idx,
                        "seed": seed,
                        "path": str(relative_path),
                    }
                )

        if self.config.save_metadata:
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
        return output_dir
