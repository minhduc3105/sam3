# SAM3 zero-shot on R2S100K

This pipeline uses the frozen Meta SAM3 image model with exactly 22 text prompts copied from `CAT-Seg/r2s100k/taxonomy.py` (SHA-256 `87d92fdd9d7652a256abeccd57235fc602de00167c5fb3b775edc034eb6bcd30`). The first 14 map to R2S100K IDs 1–14; eight concrete background prompts map to ID 0. Score threshold is 0.2. Highest score wins at each pixel; earlier prompt wins ties; uncovered pixels are background.

## Install

Use Python 3.10–3.12 and a compatible PyTorch/CUDA installation. From this directory:

```bash
bash scripts/setup_env.sh
```

The script installs the upstream SAM3 revision `f1647e8d0c1d280991f4d1b7d974c2704537fe66` into a separate venv. For a server with a specific CUDA driver, install the appropriate PyTorch build in that venv before running the script, or reinstall it afterward. Keep the `sam3.pt` checkpoint outside Git.

## One command

```bash
.venv-r2s-sam3/bin/python -m r2s_sam3 run \
  --dataset-root /data/r2s100k --split test \
  --checkpoint /models/sam3.pt --output runs/sam3_test
```

Dataset layout: `test/`, `test_labels/`; similarly `val/`, `val_labels/` and `train/`, `train_labels/Train-Labels/`. GT filename is `<image filename>___fuse.png`. `run` validates every GT before model loading. Predictions are saved both as class-ID PNGs in `predictions/` for evaluation and RGB palette masks in `colored_masks/` for viewing. Use `--max-images N` for a subset. `infer` may also accept `--images-dir` and needs no labels. For separate evaluation:

```bash
.venv-r2s-sam3/bin/python -m r2s_sam3 evaluate \
  --dataset-root /data/r2s100k --split test --output runs/sam3_test
```

A local one-image real-model smoke run:

```bash
.venv-r2s-sam3/bin/python -m r2s_sam3 smoke \
  --dataset-root /run/media/minhduc/D/Study/r2s100k \
  --checkpoint ../baseline/checkpoints/sam3/sam3.pt \
  --output runs/sam3_smoke
```

CUDA is the default; use `--device cpu` if GPU memory is insufficient. The model stays at 1008 pixels and all 22 prompts run in either case. `--resume` reuses completed class-ID PNGs only when the configuration and input fingerprint match. To evaluate saved predictions again, use the `evaluate` command. Metrics appear in `OUTPUT/evaluation/metrics.json`, per-class CSV and confusion matrices alongside up to 12 qualitative examples. A zero-union class is excluded from the mean and represented by `null`.

When evaluating a subset separately, pass the same `--max-images N` used for inference. The pinned upstream revision has CUDA positional caches and a fused BF16 ViT layer; the CPU path adapts those internals at runtime without changing upstream source or the checkpoint.

The Kaggle notebook in `notebooks/` installs this repository and runs the same CLI with editable dataset/checkpoint paths. On Kaggle, turn Internet on for the initial upstream install or supply a prebuilt wheel. Dataset and checkpoint can be Kaggle input mounts.
