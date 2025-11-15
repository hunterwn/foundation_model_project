from __future__ import annotations

import json
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
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.config = config
        self.model_kind = model_kind
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> StableDiffusionPipeline | StableDiffusionXLPipeline:
        # Determine if we're using SDXL based on model path
        is_sdxl = "xl" in self.model_path.lower()

        # Select appropriate dtype
        if self.config.precision == "bf16" and self.device == "cuda":
            torch_dtype = torch.bfloat16
        elif self.config.precision == "fp16" and self.device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Select pipeline class
        pipeline_cls = StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline

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

        scheduler_name = (self.config.scheduler or "euler_a").lower()
        scheduler_cls = SCHEDULERS.get(scheduler_name)
        if scheduler_cls is not None:
            pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(self.device)
        pipeline.enable_attention_slicing()
        return pipeline

    def run(self, prompt_set: PromptSet, output_dir: Path) -> Path:
        images_dir = ensure_dir(output_dir / "images")
        set_seed(self.config.seed)

        metadata = {
            "experiment_name": self.config.experiment_name,
            "model_path": self.model_path,
            "model_kind": self.model_kind,
            "generated_at": timestamp_now(),
            "config": self.config.to_dict(),
            "prompt_set": prompt_set.as_serializable,
            "images": [],
        }

        for prompt_entry in prompt_set.prompts:
            for image_idx in range(self.config.num_images_per_prompt):
                seed = self.config.seed + prompt_entry.index * 1000 + image_idx
                generator = torch.Generator(device=self.device).manual_seed(seed)
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
