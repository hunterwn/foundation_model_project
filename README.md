# Stable Diffusion Subject Fine-Tuning Project

This repository contains the scaffolding for comparing a pre-trained Stable Diffusion checkpoint against a custom fine-tuned version for a specific subject. It keeps original glue code separate from third-party training code via folder structure and documentation.

## Repository Layout

```
.
├── configs/                # YAML configs for experiments
├── prompts/                # Prompt templates with your subject token
├── scripts/                # Helper shell scripts (env setup, training wrapper)
├── src/                    # Original Python modules for this project (generation, comparison)
├── external/               # Third-party repos (e.g., kohya-ss/sd-scripts with their own .venv)
├── artifacts/              # Auto-created outputs (datasets, generations, comparisons)
├── models/                 # Downloaded base checkpoints (sd-v1-5-pruned, etc.)
├── .venv/                  # Project-level virtual environment (for src/ scripts)
├── requirements.txt        # Dependencies for project venv (SD 1.5, SDXL, FLUX)
└── README.md
```

## 1. Environment Setup

This project uses **two separate virtual environments**:

1. **Project-level venv** (`.venv` at repo root): For running Python scripts from `src/` (image generation, comparison, etc.). Uses `requirements.txt`.
2. **sd-scripts venv** (in `external/*/sd-scripts/.venv`): For training with kohya-ss/sd-scripts. Uses sd-scripts' own `requirements.txt`.

This separation ensures that training dependencies (which may be older) don't conflict with generation dependencies (which need newer versions for FLUX support).

### Python Version

Set the Python version in `.python-version` at the repository root. The project venv will use this version:

```bash
# Most use cases (supports newer dependencies)
echo "3.11.11" > .python-version

# If you need to match a specific sd-scripts requirement
echo "3.10.9" > .python-version
```

If using pyenv, install the required version first:

```bash
pyenv install 3.11.11  # or whatever version you need
```

### Virtual Environment Setup

```bash
# Creates both .venv (project) and sd-scripts/.venv (training)
./scripts/setup_venv.sh
```

The helper script:
1. **Project venv**: Reads Python version from `.python-version` (uses pyenv if available), creates `.venv` at repo root, and installs `requirements.txt`
2. **sd-scripts venv**: Creates venv at `${SD_SCRIPTS_DIR}/.venv` using:
   - Python 3.10.9 if "flux" is detected in the path (for kohya_flux)
   - Python version from `.python-version` otherwise (for standard sd-scripts)
   - Installs packages from the sd-scripts' own `requirements.txt`

To set up multiple sd-scripts installations, run the script with different `SD_SCRIPTS_DIR` values:

```bash
# Set up standard sd-scripts
SD_SCRIPTS_DIR=external/sd-scripts ./scripts/setup_venv.sh

# Set up kohya_flux (automatically uses Python 3.10.9)
SD_SCRIPTS_DIR=external/kohya_flux/sd-scripts ./scripts/setup_venv.sh
```

### Activating the Project Virtual Environment

Before running Python scripts from `src/`, activate the project virtual environment:

```bash
source scripts/activate_venv.sh
```

If you prefer manual activation:

```bash
source .venv/bin/activate
```

### Training Environment

The training scripts (`scripts/run_training.sh`) automatically activate the venv from the sd-scripts directory specified by `SD_SCRIPTS_DIR`. This venv is created automatically by `setup_venv.sh` as described above.

### Environment Variables

All shell and Python scripts automatically load environment variables from `.env` at the repository root (using `python-dotenv`). Add entries like `WANDB_API_KEY=...` and `SD_SCRIPTS_DIR=...` there instead of exporting manually. Copy one of the templates (`example.env` for SD 1.5, `example.xl.env` for SDXL, `example.flux.env` for Flux/SD3) to `.env` and adjust paths or hyperparameters as needed.

## 2. Configure Experiments & Prompts

1. Edit the configuration file referenced by the `EXPERIMENT_CONFIG` environment variable (defaults to `configs/experiment.yaml`, or `configs/experiment.sdxl.yaml` when running the SDXL env). Update:
   - `experiment_name`
   - `base_model` (Hugging Face repo ID or a local checkpoint path)
   - `fine_tuned_model` (where kohya-ss will save your tuned weights, e.g., `artifacts/finetune/knightro/last`)
   - `prompt_file`, `output_root`, sampler settings, and seed.
2. Create a subject-specific prompt file (e.g., `prompts/knightro_prompts.yaml`). Set `subject_identifier` to the rare token used during training (e.g., `<knightro>`) and author several positive prompts in the `positive_prompts` list. A style prefix helps keep lighting/composition consistent.

Example prompt file:

```yaml
subject_identifier: "<knightro>"
style_prefix: "RAW action photo, ultra-detailed, stadium lighting"
positive_prompts:
  - "{subject_identifier} leading the crowd with raised sword, fireworks in background"
  - "portrait of {subject_identifier} posing with cheer team, golden hour sunlight"
```

Keep this file synchronized with the prompts you want to compare before and after fine-tuning.

Example experiment config for Knightro:

```yaml
experiment_name: "knightro"
base_model: "runwayml/stable-diffusion-v1-5"
fine_tuned_model: "artifacts/finetune/knightro/last"
prompt_file: "prompts/knightro_prompts.yaml"
negative_prompt: "blurry, lowres, text, watermark, extra limbs"
num_images_per_prompt: 4
guidance_scale: 7.5
num_inference_steps: 30
seed: 12345
output_root: "artifacts/outputs"
height: 512
width: 512
scheduler: "euler_a"
precision: "fp16"
save_metadata: true
```

## 3. Baseline Image Generation (pre fine-tune)

If you use a different experiment YAML, set `EXPERIMENT_CONFIG=/path/to/other.yaml` in `.env` (copying from the example files as needed) or pass `--config` explicitly. Example:

```bash
python -m src.generate \
  --config configs/experiment.yaml \
  --model-kind baseline \
  --tag baseline
```

Outputs land in `artifacts/outputs/<experiment>-baseline-<timestamp>/` with PNGs under `images/` and metadata in `metadata.json`.

## 4. Dataset Preparation

Place your subject images inside the `dataset/` directory (e.g., `dataset/raw_knightro/`). Then run the builder to copy the images into a kohya-friendly folder and auto-generate caption `.txt` files:

```bash
python -m src.dataset.build_training_set \
  --source dataset/raw_knightro \
  --output artifacts/datasets/knightro \
  --subject-token "<knightro>" \
  --caption-template "RAW action shot of {subject_token}, UCF stadium, high energy"
```

From here you can edit the caption .txt files if you would like.

Tweak `--subject-token` and `--caption-template` to fit your subject. Use the resulting directory (e.g., `artifacts/datasets/knightro`) as `--train_data_dir` when launching kohya-ss.

## 5. Fine-Tuning with `kohya-ss/sd-scripts`

1. Clone the upstream project outside of `src/`:
   ```bash
   git clone https://github.com/kohya-ss/sd-scripts external/sd-scripts
   ```
2. Follow their README to install the additional dependencies (virtual environment can be shared or separate).
3. Example command (LoRA fine-tune) run from `external/sd-scripts`:
   ```bash
   accelerate launch train_network.py \
     --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
     --train_data_dir="../artifacts/datasets/my_subject" \
     --resolution=512,512 \
     --output_dir="../artifacts/finetune/my_subject" \
     --logging_dir="../artifacts/finetune/my_subject/logs" \
     --network_module=networks.lora \
     --train_batch_size=1 \
     --max_train_steps=2000 \
     --learning_rate=1e-4 \
     --lr_scheduler=cosine \
     --mixed_precision=fp16 \
     --save_every_n_steps=200 \
     --caption_extension=.txt
   ```
   Adjust parameters (VRAM, steps, LoRA vs. full fine-tune) to suit your hardware.
4. When training completes, set `fine_tuned_model` in `configs/experiment.yaml` to the folder containing the checkpoint you want to evaluate (e.g., `artifacts/finetune/my_subject/last`).

Flux / SD3 note: bash-j's fork of `kohya_ss` contains the Flux patches required for Flux training. The Flux training instructions are based on the [PixelWave model guide on Civitai](https://civitai.com/models/141592/pixelwave).

Clone the repository with submodules into `external/kohya_flux` and point `SD_SCRIPTS_DIR` to the forked `sd-scripts` folder:

```bash
git clone --recurse-submodules https://github.com/bash-j/kohya_ss.git external/kohya_flux
```

Then copy `example.flux.env` to `.env` to get the recommended Flux defaults (Flux base model ID, Adafactor optimizer args, Flux-specific `EXTRA_TRAIN_ARGS`, etc.). The helper script uses those env vars to target the correct training/merging entrypoints.

The helper script `./scripts/run_training.sh` wraps these steps. It creates `models/` if needed, downloads `sd-v1-5-pruned.safetensors` into that folder when the file is missing, launches training (unless `SKIP_TRAINING=1`), and then merges the resulting LoRA into `artifacts/finetune/<subject>/merged.safetensors`. Override paths via env vars such as `DATASET_DIR`, `OUTPUT_DIR`, or `MERGE_SD_MODEL`. The script auto-detects whether you're targeting SD 1.5 or SDXL and chooses the correct kohya entrypoint (`train_network.py`, `sdxl_train_network.py`, `train_db.py`, or `sdxl_train.py`) plus the right merge script/base checkpoint; set `MODEL_VARIANT=sd15|sdxl` or `TRAIN_ENTRYPOINT=/path/to/custom.py` only if you need to override the detection. Need to pass extra kohya flags (e.g., `--gradient_checkpointing`, `--cache_latents`, `--xformers`)? Set `EXTRA_TRAIN_ARGS="--flag-a --flag-b"` in `.env` and they will be appended to the Accelerate command. Set `TRAIN_METHOD=full` when you want to run kohya-ss' `train_db.py`/`sdxl_train.py` for a full DreamBooth-style fine-tune—the wrapper will automatically skip the LoRA merge step because the checkpoint saved to `OUTPUT_DIR` already includes the base weights.

## 6. Post Fine-Tune Generation

```bash
python -m src.generate \
  --config configs/experiment.yaml \
  --model-kind finetuned \
  --tag finetuned
```

This reuses the exact prompts and parameters (seeded) so comparisons remain apples-to-apples.

## 7. Visual Comparison Panels

Provide the metadata files from the baseline and fine-tuned runs:

```bash
python -m src.comparison \
  --baseline artifacts/outputs/my_subject-baseline-*/metadata.json \
  --finetuned artifacts/outputs/my_subject-finetuned-*/metadata.json \
  --output artifacts/comparisons
```

Each prompt gets a PNG showing baseline vs. fine-tuned grids, plus `manifest.json` referencing all panels.

## 8. Code Origins

- All original logic written for this project lives under `src/`, `configs/`, `prompts/`, `scripts/`, and `README.md`.
- Third-party repositories (e.g., `sd-scripts`) stay in `external/`.

## Troubleshooting

- `ModuleNotFoundError: No module named 'yaml'` (or similar): activate your virtual environment and run `pip install -r requirements.txt` to ensure dependencies are installed.
- `Prompt file not found`: make sure the file referenced by `prompt_file` exists (e.g., `prompts/knightro_prompts.yaml`) or pass `--prompts` when calling `src.generate`.
- Dataset builder finds zero images: unzip archives before running the script and verify the `--source` directory contains supported formats (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`).

## Next Steps

- Fill `artifacts/datasets/` with your curated dataset.
- Experiment with different prompt templates and checkpoint variants.
- Consider adding automated metrics (CLIP similarity, face recognition) if qualitative comparison is insufficient.
