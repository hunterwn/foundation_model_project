from __future__ import annotations

import argparse
from pathlib import Path

from .configuration import ExperimentConfig, resolve_prompts_path
from .image_generator import ImageGenerator
from .prompts import load_prompt_set
from .utils import ensure_dir, timestamp_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images for baseline or fine-tuned models.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to the experiment config YAML.")
    parser.add_argument(
        "--model-kind",
        choices=["baseline", "finetuned", "custom"],
        default="baseline",
        help="Which model checkpoint to use.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path/identifier for the model when using --model-kind custom.",
    )
    parser.add_argument("--prompts", default=None, help="Optional override for the prompts YAML file.")
    parser.add_argument("--tag", default=None, help="Optional suffix for the output directory.")
    parser.add_argument("--device", default=None, help="Force a device (cuda, cpu). Defaults to auto detection.")
    return parser.parse_args()


def determine_model_path(config: ExperimentConfig, args: argparse.Namespace) -> str:
    if args.model_kind == "baseline":
        return config.base_model
    if args.model_kind == "finetuned":
        return config.fine_tuned_model
    if args.model_path:
        return args.model_path
    raise ValueError("--model-path must be provided when --model-kind custom is used")


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.load(args.config)
    prompts_path = resolve_prompts_path(config, args.prompts)
    prompt_set = load_prompt_set(prompts_path)
    model_path = determine_model_path(config, args)

    tag = args.tag or args.model_kind
    run_dir_name = f"{config.experiment_name}-{tag}-{timestamp_now()}"
    run_dir = ensure_dir(Path(config.output_root) / run_dir_name)

    generator = ImageGenerator(model_path=model_path, config=config, model_kind=args.model_kind, device=args.device)
    generator.run(prompt_set, run_dir)
    print(f"[+] Images saved to {run_dir}")


if __name__ == "__main__":
    main()
