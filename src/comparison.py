from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

from .utils import ensure_dir, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs. fine-tuned generations.")
    parser.add_argument("--baseline", required=True, help="Path to the baseline metadata.json file.")
    parser.add_argument("--finetuned", required=True, help="Path to the fine-tuned metadata.json file.")
    parser.add_argument("--output", default="artifacts/comparisons", help="Directory to store the comparison panels.")
    parser.add_argument("--columns", type=int, default=2, help="Number of columns per grid.")
    return parser.parse_args()


def load_metadata(path: str | Path) -> Dict:
    metadata_path = Path(path)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    base_dir = metadata_path.parent
    for image in data.get("images", []):
        image["absolute_path"] = str((base_dir / image["path"]).resolve())
    return data


def group_by_prompt(metadata: Dict) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for entry in metadata.get("images", []):
        grouped.setdefault(entry["prompt_slug"], []).append(entry)
    return grouped


def make_grid(image_paths: List[Path], columns: int) -> Image.Image:
    if not image_paths:
        raise ValueError("No images supplied to build the grid")

    opened = [Image.open(path).convert("RGB") for path in image_paths]
    width, height = opened[0].size

    columns = max(1, columns)
    rows = (len(opened) + columns - 1) // columns

    grid = Image.new("RGB", (columns * width, rows * height), color=(0, 0, 0))
    for idx, img in enumerate(opened):
        row = idx // columns
        col = idx % columns
        grid.paste(img, (col * width, row * height))
    for img in opened:
        img.close()
    return grid


def draw_panel(prompt_text: str, baseline_grid: Image.Image, finetuned_grid: Image.Image) -> Image.Image:
    title_height = 80
    separation = 20
    width = baseline_grid.width + finetuned_grid.width + separation
    height = max(baseline_grid.height, finetuned_grid.height) + title_height
    panel = Image.new("RGB", (width, height), color=(15, 15, 15))

    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=20)
    draw.text((10, 10), prompt_text[:180], fill=(255, 255, 255), font=font)
    draw.text((10, 40), "Baseline", fill=(200, 200, 200), font=font)
    draw.text((baseline_grid.width + separation + 10, 40), "Fine-tuned", fill=(200, 200, 200), font=font)

    panel.paste(baseline_grid, (0, title_height))
    panel.paste(finetuned_grid, (baseline_grid.width + separation, title_height))
    return panel


def build_panels(baseline_meta: Dict, finetuned_meta: Dict, output_dir: Path, columns: int) -> List[Dict]:
    baseline_grouped = group_by_prompt(baseline_meta)
    finetuned_grouped = group_by_prompt(finetuned_meta)

    manifest = []

    for prompt in baseline_meta.get("prompt_set", {}).get("prompts", []):
        slug = prompt["slug"]
        prompt_text = prompt["formatted"]
        base_entries = baseline_grouped.get(slug)
        fine_entries = finetuned_grouped.get(slug)
        if not base_entries or not fine_entries:
            continue

        base_paths = [Path(entry["absolute_path"]) for entry in sorted(base_entries, key=lambda x: x["image_index"])]
        fine_paths = [Path(entry["absolute_path"]) for entry in sorted(fine_entries, key=lambda x: x["image_index"])]

        baseline_grid = make_grid(base_paths, columns=columns)
        finetuned_grid = make_grid(fine_paths, columns=columns)
        panel = draw_panel(prompt_text, baseline_grid, finetuned_grid)
        output_path = output_dir / f"{prompt['index']:02d}-{slugify(prompt_text)}.png"
        panel.save(output_path)

        manifest.append(
            {
                "prompt": prompt_text,
                "slug": slug,
                "panel": str(output_path),
            }
        )

    return manifest


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)
    baseline_meta = load_metadata(args.baseline)
    finetuned_meta = load_metadata(args.finetuned)
    manifest = build_panels(baseline_meta, finetuned_meta, output_dir, args.columns)
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[+] Saved {len(manifest)} comparison panels to {output_dir}")


if __name__ == "__main__":
    main()
