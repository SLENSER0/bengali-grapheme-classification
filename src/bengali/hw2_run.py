"""Evaluate a fixed 2×2 head ablation on the archived B2 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .hw2_controls import control_logits
from .hw2_heads import fit_heads, fuse_heads
from .models import FinetuneNet, count_params
from .runs import load_fold_data, prediction_frame, run_dir, save_run
from .utils import COMPONENTS, Timer, get_device, load_config, resolve, seed_everything

CELLS = ("b2", "p2p", "gda", "both")
CONTROLS = ("freq", "final_gda")
PARTS = ("val_id", "val_ood")


def selected_runs(cfg, folds=None, seeds=None):
    pairs = [(int(r["fold"]), int(r["seed"])) for r in cfg["runs"]]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate fold/seed in configured runs")
    selected = [
        (f, s) for f, s in pairs
        if (folds is None or f in folds) and (seeds is None or s in seeds)
    ]
    if not selected:
        raise ValueError("No configured run matches the fold/seed filters")
    return selected


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.inference_mode()
def collect(model, images, rows, stages, device, batch_size):
    """Capture GAP activations during the exact FinetuneNet forward used by B2."""
    model.eval()
    features, logits = {}, {}
    pooled = {}

    def capture(name):
        def hook(module, inputs, output):
            pooled[name] = output.mean(dim=(-2, -1))
        return hook

    with ExitStack() as hooks:
        for name in stages:
            layer = model.backbone.get_submodule(name)
            hooks.callback(layer.register_forward_hook(capture(name)).remove)
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            x = torch.from_numpy(images[batch]).to(device)
            out = model(x)
            for source, destination in ((pooled, features), (out, logits)):
                for name, value in source.items():
                    array = value.float().cpu().numpy()
                    if name not in destination:
                        destination[name] = np.empty((len(rows), array.shape[1]), dtype=np.float32)
                    destination[name][start : start + len(batch)] = array
    return features, logits


def archive_parity(frame, archive_path):
    """Require identical IDs, slice membership, and every predicted class."""
    key = ["image_id", "slice"]
    actual = frame.set_index(key).sort_index()
    archived = pd.read_csv(archive_path).set_index(key).sort_index()
    if not actual.index.is_unique or not archived.index.is_unique:
        raise ValueError("B2 archive parity requires unique image/slice keys")
    if not actual.index.equals(archived.index):
        raise ValueError(f"B2 image/slice keys disagree with {archive_path}")
    rounded_confidence_diff = {}
    for component in COMPONENTS:
        col = f"pred_{component}"
        disagreement = actual[col].ne(archived[col])
        if disagreement.any():
            raise ValueError(f"B2 archive mismatch for {component}: {disagreement.sum()} rows; "
                             f"first keys {disagreement[disagreement].index.tolist()[:3]}")
        col = f"conf_{component}"
        diff = np.abs(actual[col].to_numpy(dtype=float) - archived[col].to_numpy(dtype=float))
        if not np.isfinite(diff).all():
            raise ValueError(f"Nonfinite B2 confidence for {component}")
        rounded_confidence_diff[component] = float(diff.max())
    return {"rows": len(actual), "all_classes_match": True,
            "max_abs_rounded_confidence_difference": rounded_confidence_diff}


def classify(scores, device):
    preds, confs = {}, {}
    for component in COMPONENTS:
        probabilities = torch.as_tensor(scores[component], dtype=torch.float32, device=device).softmax(1)
        confidence, predicted = probabilities.max(1)
        preds[component] = predicted.cpu().numpy()
        confs[component] = confidence.cpu().numpy()
    return preds, confs


def save_cells(cfg, fold, seed, frames, extra):
    """Finish writing all runs before replacing any published result file."""
    root = resolve(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".writing-", dir=root) as temp:
        outputs = []
        for cell, frame in frames.items():
            cell_path = Path("controls") / cell if cell in CONTROLS else Path(cell)
            final_cfg = {**cfg, "experiment": f"{cfg['experiment']}_{cell}",
                         "output_dir": str(Path(cfg["output_dir"]) / cell_path)}
            stage_cfg = {**final_cfg, "output_dir": str(Path(temp) / cell)}
            metrics = save_run(stage_cfg, fold, seed, frame, extra)
            staged = run_dir(stage_cfg, fold, seed)
            with (staged / "config.yaml").open("w") as stream:
                yaml.safe_dump({**final_cfg, "run": {"fold": fold, "seed": seed}}, stream,
                               allow_unicode=True, sort_keys=False)
            outputs.append((staged, run_dir(final_cfg, fold, seed)))
            print(f"{cell}: main={metrics['main_score']:.6f} id={metrics['id_score']:.6f} "
                  f"ood={metrics['ood_score']:.6f}", flush=True)
        for staged, dest in outputs:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.replace(staged, dest)
                continue
            # metrics.json is the completion marker for an individual published run.
            (dest / "metrics.json").unlink(missing_ok=True)
            for name in ("preds.csv.gz", "config.yaml", "metrics.json"):
                os.replace(staged / name, dest / name)


def run(cfg, fold, seed):
    clock = Timer()
    seed_everything(seed)
    settings = cfg["hw2"]
    device = get_device(cfg.get("device", "auto"))
    batch_size = cfg["eval"]["batch_size"]
    stages = settings["stages"]
    if cfg["data"].get("subset") is not None or batch_size < 1:
        raise ValueError("Full folds and a positive eval.batch_size are required")
    if list(stages) != ["layer3", "layer4"]:
        raise ValueError("The registered HW2 stages are layer3 and layer4")
    run_name = f"fold{fold}_seed{seed}"
    checkpoint = resolve(settings["checkpoint_dir"]) / f"{run_name}.pt"
    archive = resolve(settings["archive_dir"]) / run_name
    archive_cfg = load_config(archive / "config.yaml")
    with (archive / "metrics.json").open() as stream:
        archived_metrics = json.load(stream)
    expected = {"fold": fold, "seed": seed}
    if (any(archived_metrics.get(key) != value for key, value in expected.items())
            or archive_cfg.get("run", expected) != expected):
        raise ValueError(f"Wrong fold or seed in {archive}")
    if archive_cfg["data"]["img_size"] != cfg["data"]["img_size"]:
        raise ValueError("B2 and HW2 image sizes differ")
    print(f"[{run_name}] loading B2 on {device}; resolution={cfg['data']['img_size']} "
          f"batch_size={batch_size}", flush=True)
    model = FinetuneNet(archive_cfg["model"]["backbone"], pretrained=False,
                        dropout=archive_cfg["model"].get("dropout", 0.0))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    images, folds, labels, ix = load_fold_data(cfg, fold, in_memory=False)
    train = ix["train"]
    counts = {name: len(rows) for name, rows in ix.items()}
    print(f"[{run_name}] extracting train features: {counts}", flush=True)
    timer = Timer()
    train_features, train_logits = collect(model, images, train, stages, device, batch_size)
    train_forward_sec = timer()
    print(f"[{run_name}] fitting heads after {train_forward_sec:.1f}s train extraction", flush=True)
    timer = Timer()
    train_labels = labels[train]
    log_priors, discriminants = fit_heads(train_features, train_logits, train_labels, settings["ridge"])
    fit_sec = timer()
    del train_features, train_logits
    cells = CELLS + CONTROLS
    preds = {cell: {} for cell in cells}
    confs = {cell: {} for cell in cells}
    timer = Timer()
    for part in PARTS:
        print(f"[{run_name}] scoring {part} ({len(ix[part])} images)", flush=True)
        features, raw = collect(model, images, ix[part], stages, device, batch_size)
        fused = fuse_heads(features, raw, log_priors, discriminants,
                           settings["alpha"], settings["prior_strength"])
        fused.update(control_logits(features, raw, log_priors, discriminants, train_labels, settings))
        for cell in cells:
            preds[cell][part], confs[cell][part] = classify(fused[cell], device)
        del features, raw, fused
    validation_sec = timer()
    frames = {cell: prediction_frame(folds, ix, preds[cell], confs[cell]) for cell in cells}
    archive_file = archive / "preds.csv.gz"
    parity = archive_parity(frames["b2"], archive_file)
    print(f"[{run_name}] archive parity: {parity}", flush=True)
    extra = {
        "checkpoint_sha256": file_hash(checkpoint), "folds_sha256": file_hash(resolve(cfg["data"]["folds_file"])),
        "archive_predictions_sha256": file_hash(archive_file), "archive_parity": parity,
        "device": str(device), "platform": platform.platform(), "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "timm", "numpy", "scipy", "pandas")},
        "n_train": len(train), "row_counts": counts, "backbone": archive_cfg["model"]["backbone"],
        "n_params": count_params(model), "n_params_trainable_backbone": 0,
        "shared_timings_sec": {"train_feature_extraction": train_forward_sec,
                               "fit_all_heads": fit_sec, "validation_all_cells": validation_sec,
                               "elapsed_before_writing": clock()},
    }
    save_cells(cfg, fold, seed, frames, extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, nargs="+")
    parser.add_argument("--seed", type=int, nargs="+")
    parser.add_argument("--set", nargs="*", default=[], help="Nested config overrides as key=value")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    for fold, seed in selected_runs(cfg, args.fold, args.seed):
        run(cfg, fold, seed)


if __name__ == "__main__":
    main()
