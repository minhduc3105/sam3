"""CLI for zero-shot SAM3 on R2S100K."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from .core import atomic_json, atomic_png, class_map, colorize, evaluate, images_in, merge_masks, progress, sha256, validate_labels
from .taxonomy import model_to_dataset_ids, model_vocabulary

SAM3_REVISION = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"
TAXONOMY_SOURCE = "CAT-Seg/r2s100k/taxonomy.py"
THRESHOLD = 0.2
RESOLUTION = 1008


def log(stage, message):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] [{stage}] {message}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("run", "infer", "evaluate", "smoke"):
        q = sub.add_parser(command)
        q.add_argument("--dataset-root", type=Path)
        q.add_argument("--split", choices=("train", "val", "test"), default="val" if command == "smoke" else "test")
        q.add_argument("--images-dir", type=Path, help="Only for infer; dataset-root is optional")
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--max-images", type=int, default=1 if command == "smoke" else 0)
        if command != "evaluate":
            q.add_argument("--checkpoint", type=Path, required=True)
            q.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
            q.add_argument("--resume", action="store_true")
        else:
            q.add_argument("--predictions", type=Path, help="Defaults to OUTPUT/predictions")
    return p


def selected_images(args):
    if args.images_dir:
        if args.command != "infer":
            raise ValueError("--images-dir is available only for infer")
        directory = args.images_dir
    else:
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required")
        directory = args.dataset_root / args.split
    paths = images_in(directory)
    if args.max_images < 0:
        raise ValueError("--max-images must be nonnegative")
    return paths[:args.max_images] if args.max_images else paths


def infer(args, images):
    log("setup", "Importing PyTorch and SAM3")
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")
    if args.device == "cpu":
        # This pinned SAM3 revision hardcodes BF16 in its fused ViT MLP.
        # On CPU that leaves its next FP32 Linear with mismatched dtypes.
        import torch.nn.functional as F
        import sam3.model.vitdet as vitdet

        def cpu_addmm_act(activation, linear, inputs):
            return activation()(F.linear(inputs, linear.weight, linear.bias))

        vitdet.addmm_act = cpu_addmm_act
    precision = "bf16" if args.device == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    log("setup", f"Device={args.device}, precision={precision}, images={len(images)}")
    taxonomy_path = Path(__file__).with_name("taxonomy.py")
    log("setup", f"Hashing checkpoint: {args.checkpoint}")
    manifest = [{"image_id": p.name, "image_path": str(p.resolve()), "size": p.stat().st_size,
                 "mtime_ns": p.stat().st_mtime_ns} for p in images]
    config = {"sam3_revision": SAM3_REVISION, "taxonomy_source": TAXONOMY_SOURCE,
              "taxonomy_sha256": sha256(taxonomy_path), "vocabulary": model_vocabulary(),
              "model_to_dataset_ids": model_to_dataset_ids(), "threshold": THRESHOLD,
              "resolution": RESOLUTION, "device": args.device, "precision": precision,
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
              "images": manifest, "subset": args.max_images > 0}
    log("setup", "Checkpoint hash and run configuration ready")
    output = args.output
    pred_dir = output / "predictions"
    color_dir = output / "colored_masks"
    metadata_path = output / "run_metadata.json"
    old = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise FileExistsError(f"Output exists; choose new --output or use --resume: {output}")
        if old is None or old.get("config") != config:
            raise ValueError("Resume config/input fingerprint mismatch")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(metadata_path, {"config": config, "status": "in_progress"})
    pred_dir.mkdir(exist_ok=True)
    color_dir.mkdir(exist_ok=True)
    pending = []
    for p in images:
        dest = pred_dir / (p.name + ".png")
        colored = color_dir / (p.name + ".png")
        if args.resume and dest.is_file() and colored.is_file():
            with Image.open(dest) as im, Image.open(p) as source:
                if im.mode == "L" and im.size == source.size and np.asarray(im).max() < 15:
                    continue
        pending.append(p)
    log("setup", f"Output={output}; pending={len(pending)}, reusable={len(images) - len(pending)}")
    if pending:
        log("model", "Loading checkpoint and building SAM3 model")
        model = build_sam3_image_model(checkpoint_path=str(args.checkpoint), load_from_HF=False, device=args.device)
        model.eval()
        if args.device == "cpu":
            # SAM3 precomputes some positional caches on CUDA at construction.
            for module in model.modules():
                if isinstance(getattr(module, "cache", None), dict):
                    module.cache = {key: value.cpu() if torch.is_tensor(value) else value
                                    for key, value in module.cache.items()}
                coords = getattr(module, "compilable_cord_cache", None)
                if isinstance(coords, tuple):
                    module.compilable_cord_cache = tuple(x.cpu() for x in coords)
        processor = Sam3Processor(model, resolution=RESOLUTION, device=args.device, confidence_threshold=THRESHOLD)
        log("model", "Model ready; starting inference")
        start = time.monotonic()
        for index, image_path in enumerate(pending, 1):
            image_start = time.monotonic()
            log("infer", f"Image {index}/{len(pending)}: {image_path.name}; loading and encoding")
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            height, width = image.height, image.width
            best_score = np.full((height, width), -np.inf, dtype=np.float32)
            best_prompt = np.full((height, width), -1, dtype=np.int16)
            context = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
            with torch.inference_mode(), context:
                state = processor.set_image(image)
                log("infer", f"{image_path.name}: image encoded; evaluating 22 prompts")
                for prompt_idx, prompt in enumerate(model_vocabulary()):
                    processor.reset_all_prompts(state)
                    result = processor.set_text_prompt(prompt=prompt, state=state)
                    masks = result["masks"].squeeze(1).cpu().numpy()
                    # NumPy has no bfloat16 dtype; CUDA autocast may return BF16 scores.
                    scores = result["scores"].float().cpu().numpy()
                    merge_masks(best_score, best_prompt, prompt_idx, masks, scores, THRESHOLD)
                    if (prompt_idx + 1) % 4 == 0 or prompt_idx == 21:
                        log("infer", f"{image_path.name}: prompt {prompt_idx + 1}/22")
            ids = class_map(best_prompt)
            atomic_png(pred_dir / (image_path.name + ".png"), ids)
            atomic_png(color_dir / (image_path.name + ".png"), colorize(ids))
            log("infer", f"{image_path.name}: masks saved in {time.monotonic() - image_start:.1f}s")
            progress("infer", index, len(pending), start)
    atomic_json(metadata_path, {"config": config, "status": "complete", "predictions": len(images)})
    log("infer", f"Completed {len(images)} predictions")
    return pred_dir


def main():
    args = parser().parse_args()
    log("scan", f"Scanning images: split={args.split}, dataset={args.dataset_root}, output={args.output}")
    images = selected_images(args)
    log("scan", f"Found {len(images)} images")
    labels = None
    if args.command in ("run", "evaluate", "smoke"):
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for evaluation")
        log("validate", f"Checking image sizes and GT colors for {len(images)} images")
        validation_start = time.monotonic()
        labels = validate_labels(images, args.dataset_root, args.split,
                                 progress_callback=lambda done, total: progress("validate", done, total, validation_start))
        log("validate", "All labels passed")
    if args.command == "evaluate":
        predictions = args.predictions or args.output / "predictions"
    else:
        predictions = infer(args, images)
    if labels is not None:
        log("evaluate", f"Scoring {len(images)} saved predictions")
        report = evaluate(images, labels, predictions, args.output / "evaluation")
        print(json.dumps(report, indent=2))
        print(f"Metrics: {args.output / 'evaluation' / 'metrics.json'}")
    else:
        print(f"Predictions: {predictions}")


if __name__ == "__main__":
    main()
