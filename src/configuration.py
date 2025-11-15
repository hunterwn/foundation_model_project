from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ExperimentConfig:
    experiment_name: str
    base_model: str
    fine_tuned_model: str
    prompt_file: str
    negative_prompt: str
    num_images_per_prompt: int
    guidance_scale: float
    num_inference_steps: int
    seed: int
    output_root: str
    height: int
    width: int
    scheduler: str = "euler_a"
    precision: str = "fp16"
    save_metadata: bool = True

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "base_model": self.base_model,
            "fine_tuned_model": self.fine_tuned_model,
            "prompt_file": self.prompt_file,
            "negative_prompt": self.negative_prompt,
            "num_images_per_prompt": self.num_images_per_prompt,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "output_root": self.output_root,
            "height": self.height,
            "width": self.width,
            "scheduler": self.scheduler,
            "precision": self.precision,
            "save_metadata": self.save_metadata,
        }


def resolve_prompts_path(config: ExperimentConfig, override: str | None = None) -> Path:
    candidate = Path(override or config.prompt_file)
    if not candidate.exists():
        raise FileNotFoundError(f"Prompt file not found: {candidate}")
    return candidate
