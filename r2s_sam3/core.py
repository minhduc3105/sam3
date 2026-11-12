"""Dataset I/O, mask fusion, and semantic metrics; no SAM3 dependency."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .taxonomy import CLASSES, IMAGE_SUFFIXES, NUM_CLASSES, model_to_dataset_ids, model_vocabulary

PALETTE = np.array([CLASSES[i][2] for i in range(NUM_CLASSES)], dtype=np.uint8)
RGB_KEYS = {(int(c[0]) << 16) | (int(c[1]) << 8) | int(c[2]): i for i, c in enumerate(PALETTE)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_png(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(arr).save(tmp, format="PNG")
    os.replace(tmp, path)


def images_in(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory missing: {directory}")
    paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No images found in {directory}")
    names = [p.name for p in paths]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image names")
    return paths


def label_for(root, split, image):
    base = Path(root) / f"{split}_labels"
    name = image.name + "___fuse.png"
    matches = [p for p in (base / name, base / "Train-Labels" / name) if p.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one GT for {image}: {matches}; searched {base}")
    return matches[0]


def decode_gt(rgb):
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("GT must be RGB")
    packed = rgb[..., 0].astype(np.uint32) << 16 | rgb[..., 1].astype(np.uint32) << 8 | rgb[..., 2].astype(np.uint32)
    out = np.full(packed.shape, 255, np.uint8)
    for key, value in RGB_KEYS.items():
        out[packed == key] = value
    if (out == 255).any():
        bad = np.unique(packed[out == 255])[:8]
        colors = [(int(x >> 16), int((x >> 8) & 255), int(x & 255)) for x in bad]
        raise ValueError(f"Unknown GT colors: {colors}")
    return out


def validate_labels(images, root, split, progress_callback=None):
    labels = {}
    for index, image in enumerate(images, 1):
        gt_path = label_for(root, split, image)
        with Image.open(image) as im, Image.open(gt_path) as gt_im:
            if im.size != gt_im.size:
                raise ValueError(f"GT size mismatch: {image} vs {gt_path}")
            decode_gt(np.asarray(gt_im.convert("RGB")))
        labels[image.name] = gt_path
        if progress_callback is not None and (index == 1 or index % 100 == 0 or index == len(images)):
            progress_callback(index, len(images))
    return labels


def merge_masks(best_score, best_prompt, prompt_idx, masks, scores, threshold=0.2):
    """Update winning prompt per pixel; earlier prompt wins exact ties."""
    for mask, score in zip(masks, scores):
        score = float(score)
        if score <= threshold:
            continue
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != best_score.shape:
            raise ValueError(f"Mask shape {mask.shape} differs from image {best_score.shape}")
        update = mask & (score > best_score)
        best_score[update] = score
        best_prompt[update] = prompt_idx


def class_map(best_prompt):
    mapping = np.array(model_to_dataset_ids(), dtype=np.uint8)
    out = np.zeros(best_prompt.shape, dtype=np.uint8)
    valid = best_prompt >= 0
    out[valid] = mapping[best_prompt[valid]]
    return out


def colorize(class_ids):
    """Map R2S100K class IDs to the taxonomy's RGB palette."""
    class_ids = np.asarray(class_ids)
    if class_ids.size and (class_ids.min() < 0 or class_ids.max() >= NUM_CLASSES):
        raise ValueError("Class map contains an ID outside 0..14")
    return PALETTE[class_ids]


def confusion_for(gt, pred):
    if gt.shape != pred.shape:
        raise ValueError("GT/prediction size mismatch")
    if np.any(pred >= NUM_CLASSES):
        raise ValueError("Prediction contains class outside 0..14")
    encoded = NUM_CLASSES * gt.astype(np.int64).ravel() + pred.astype(np.int64).ravel()
    return np.bincount(encoded, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm)
    gt_count, pred_count = cm.sum(1), cm.sum(0)
    union = gt_count + pred_count - tp
    def ratio(n, d):
        return np.divide(n, d, out=np.full(NUM_CLASSES, np.nan), where=d > 0)
    iou = ratio(tp, union)
    dice = ratio(2 * tp, gt_count + pred_count)
    precision = ratio(tp, pred_count)
    recall = ratio(tp, gt_count)
    def mean(a):
        x = a[np.isfinite(a)]
        return float(x.mean()) if x.size else None
    fg_tp = int(cm[1:, 1:].sum())
    fg_gt = int(cm[1:, :].sum())
    fg_pred = int(cm[:, 1:].sum())
    rows = {}
    for i in range(NUM_CLASSES):
        rows[str(i)] = {"name": CLASSES[i][0], "iou": None if np.isnan(iou[i]) else float(iou[i]),
                        "dice": None if np.isnan(dice[i]) else float(dice[i]),
                        "precision": None if np.isnan(precision[i]) else float(precision[i]),
                        "recall": None if np.isnan(recall[i]) else float(recall[i]),
                        "gt_pixels": int(gt_count[i]), "pred_pixels": int(pred_count[i])}
    return {"miou_15": mean(iou), "miou_foreground14": mean(iou[1:]),
            "classes_in_miou_15": int(np.isfinite(iou).sum()),
            "classes_in_miou_foreground14": int(np.isfinite(iou[1:]).sum()),
            "foreground_binary_iou": fg_tp / (fg_gt + fg_pred - fg_tp) if fg_gt + fg_pred - fg_tp else None,
            "foreground_binary_dice": 2 * fg_tp / (fg_gt + fg_pred) if fg_gt + fg_pred else None,
            "class_metrics": rows}


def progress(label, done, total, start):
    elapsed = max(time.monotonic() - start, 1e-6)
    eta = elapsed * (total - done) / done if done else 0
    print(f"[{label}] {done}/{total} ETA {eta:.0f}s", flush=True)


def evaluate(images, labels, predictions, output):
    output = Path(output)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    examples = []
    start = time.monotonic()
    for index, image in enumerate(images, 1):
        pred_path = Path(predictions) / (image.name + ".png")
        if not pred_path.is_file():
            raise FileNotFoundError(f"Prediction missing: {pred_path}")
        gt_path = labels[image.name]
        pred_im = Image.open(pred_path)
        if pred_im.mode != "L":
            raise ValueError(f"Prediction must be class-ID L PNG: {pred_path}")
        pred = np.asarray(pred_im)
        with Image.open(gt_path) as gt_im:
            gt = decode_gt(np.asarray(gt_im.convert("RGB")))
        cm += confusion_for(gt, pred)
        if len(examples) < 12 and np.any(gt != pred):
            examples.append((image, gt, pred))
        progress("evaluate", index, len(images), start)
    report = {"evaluated_images": len(images), **metrics_from_confusion(cm)}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "metrics.json", report)
    with open(output / "class_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "iou", "dice", "precision", "recall", "gt_pixels", "pred_pixels"])
        w.writeheader()
        for i in range(NUM_CLASSES):
            w.writerow({"id": i, **report["class_metrics"][str(i)]})
    np.savetxt(output / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    norm = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(1, keepdims=True)>0)
    np.savetxt(output / "confusion_matrix_gt_normalized.csv", norm, fmt="%.6f", delimiter=",")
    qdir = output / "qualitative"
    qdir.mkdir(exist_ok=True)
    for image, gt, pred in examples:
        rgb = np.asarray(Image.open(image).convert("RGB"))
        err = np.zeros_like(rgb)
        err[gt == pred] = [30, 150, 50]
        err[(gt > 0) & (pred == 0)] = [240, 50, 50]
        err[(gt == 0) & (pred > 0)] = [50, 100, 250]
        err[(gt > 0) & (pred > 0) & (gt != pred)] = [255, 190, 0]
        atomic_png(qdir / (image.name + ".original.png"), rgb)
        atomic_png(qdir / (image.name + ".gt.png"), PALETTE[gt])
        atomic_png(qdir / (image.name + ".pred.png"), PALETTE[pred])
        atomic_png(qdir / (image.name + ".errors.png"), err)
    return report
