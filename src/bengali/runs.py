"""Common run plumbing: loading data for a fold, saving predictions and metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import load_processed
from .metric import slice_scores
from .splits import fold_indices, load_folds
from .utils import COMPONENTS, resolve, save_json


def load_fold_data(cfg: dict, fold: int, in_memory: bool = True):
    """Return images, the folds frame aligned with images, and index dict for the fold."""
    dcfg = cfg["data"]
    imgs, index = load_processed(dcfg["processed_dir"], dcfg["img_size"])
    folds = load_folds(dcfg["folds_file"], dcfg.get("train_csv", "data/raw/train.csv"))
    if not np.array_equal(index["image_id"].to_numpy(), folds["image_id"].to_numpy()):
        folds = folds.set_index("image_id").loc[index["image_id"]].reset_index()
    ix = fold_indices(folds, fold)
    sub = dcfg.get("subset")  # smoke runs: random subset of each part
    if sub:
        rng = np.random.default_rng(0)
        ix = {k: np.sort(rng.choice(v, size=min(len(v), int(sub * (1 if k == "train" else 0.25))), replace=False)) for k, v in ix.items()}
    if in_memory:
        imgs = np.array(imgs)  # real copy into RAM (~3.3 GB); otherwise the array stays memory-mapped
    labels = folds[COMPONENTS].to_numpy().astype(np.int64)
    return imgs, folds, labels, ix


def prediction_frame(folds: pd.DataFrame, ix: dict, preds: dict, confs: dict) -> pd.DataFrame:
    """preds/confs: {'val_id': {component: array}, 'val_ood': {...}}."""
    parts = []
    for part, slice_name in (("val_id", "id"), ("val_ood", "ood")):
        if len(ix[part]) == 0:
            continue
        f = folds.iloc[ix[part]][["image_id", "grapheme", *COMPONENTS]].copy()
        f.insert(2, "slice", slice_name)
        for c in COMPONENTS:
            f[f"pred_{c}"] = preds[part][c]
            f[f"conf_{c}"] = np.round(confs[part][c], 4)
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def run_dir(cfg: dict, fold: int, seed: int) -> Path:
    return resolve(cfg["output_dir"]) / f"fold{fold}_seed{seed}"


def save_run(cfg: dict, fold: int, seed: int, pred_df: pd.DataFrame, extra: dict) -> dict:
    d = run_dir(cfg, fold, seed)
    d.mkdir(parents=True, exist_ok=True)
    # labels are not stored (competition data must not be redistributed); evaluate.load_oof joins them back
    pred_df.drop(columns=["grapheme", *COMPONENTS]).to_csv(d / "preds.csv.gz", index=False)
    metrics = {"experiment": cfg["experiment"], "fold": fold, "seed": seed, **slice_scores(pred_df), **extra}
    save_json(metrics, d / "metrics.json")
    with open(d / "config.yaml", "w") as f:
        yaml.safe_dump({**cfg, "run": {"fold": fold, "seed": seed}}, f, allow_unicode=True, sort_keys=False)
    return metrics
