"""Raw parquet -> preprocessed uint8 arrays.

Preprocessing (per image, deterministic, nothing is fitted on data => no leakage):
1. invert so that ink is bright (background ~0) and stretch to [0, 255];
2. find the bounding box of ink pixels (> ``ink_thr``), ignoring a thin border
   where scanning artefacts live;
3. crop with a small margin, zero-out faint noise (< ``noise_thr``);
4. pad to a square (keeps aspect ratio) and resize to ``size`` x ``size``.

The crop-and-resize idea follows the public Kaggle kernel by Iafoss,
"Image preprocessing (128x128)"; this is an independent re-implementation
(see README, "Источники и заимствования").
"""

from __future__ import annotations

import argparse
import hashlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from .utils import RAW_H, RAW_W, resolve

BORDER = 5


def read_parquet_images(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (image_ids, images[N, 137, 236] uint8)."""
    table = pq.read_table(path)
    ids = table.column("image_id").to_numpy(zero_copy_only=False).astype(str)
    # pixel columns are named "0".."32331"; the files also carry a pandas "__index_level_0__" column
    cols = sorted((c for c in table.column_names if c.isdigit()), key=int)
    assert len(cols) == RAW_H * RAW_W, len(cols)
    imgs = np.empty((table.num_rows, RAW_H * RAW_W), dtype=np.uint8)
    for j, c in enumerate(cols):
        imgs[:, j] = table.column(c).to_numpy()
    return ids, imgs.reshape(-1, RAW_H, RAW_W)


def crop_resize(img: np.ndarray, size: int = 128, pad: int = 16, ink_thr: int = 80, noise_thr: int = 28) -> tuple[np.ndarray, dict]:
    """Crop the grapheme by its ink bounding box and resize to a square.

    Returns the processed image and a few raw statistics used in EDA.
    """
    inv = 255 - img.astype(np.int16)
    mx = int(inv.max())
    inv = (inv * (255.0 / max(mx, 1))).clip(0, 255).astype(np.uint8)
    inner = inv[BORDER:-BORDER, BORDER:-BORDER] > ink_thr
    rows = np.flatnonzero(inner.any(axis=1))
    cols = np.flatnonzero(inner.any(axis=0))
    if rows.size == 0 or cols.size == 0:  # empty image: keep everything
        y0, y1, x0, x1 = 0, inv.shape[0], 0, inv.shape[1]
    else:
        y0, y1 = rows[0] + BORDER, rows[-1] + BORDER + 1
        x0, x1 = cols[0] + BORDER, cols[-1] + BORDER + 1
        # small safety margin around the box
        y0, x0 = max(y0 - 8, 0), max(x0 - 10, 0)
        y1, x1 = min(y1 + 8, inv.shape[0]), min(x1 + 10, inv.shape[1])
    crop = inv[y0:y1, x0:x1].copy()
    crop[crop < noise_thr] = 0
    h, w = crop.shape
    side = max(h, w) + pad
    canvas = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy : oy + h, ox : ox + w] = crop
    out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    stats = {
        "bbox_h": int(y1 - y0),
        "bbox_w": int(x1 - x0),
        "ink_frac": float((inv > ink_thr).mean()),
        "raw_min": int(img.min()),
        "raw_max": int(img.max()),
    }
    return out, stats


def _work(args):
    img, size = args
    out, st = crop_resize(img, size=size)
    st["md5"] = hashlib.md5(img.tobytes()).hexdigest()
    return out, st


def preprocess_split(raw_dir: Path, out_dir: Path, split: str, size: int, workers: int) -> None:
    files = sorted(raw_dir.glob(f"{split}_image_data_*.parquet"))
    assert files, f"no {split} parquet files in {raw_dir}"
    all_ids, all_imgs, all_stats = [], [], []
    samples_raw = None
    for f in files:
        ids, imgs = read_parquet_images(f)
        if samples_raw is None:
            samples_raw = imgs[:64].copy()
        with Pool(workers) as pool:
            res = list(
                tqdm(pool.imap(_work, ((im, size) for im in imgs), chunksize=256), total=len(imgs), desc=f.name)
            )
        all_ids.append(ids)
        all_imgs.append(np.stack([r[0] for r in res]))
        all_stats.extend(r[1] for r in res)
    ids = np.concatenate(all_ids)
    imgs = np.concatenate(all_imgs)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split}_{size}.npy", imgs)
    stats = pd.DataFrame(all_stats)
    stats.insert(0, "image_id", ids)
    stats.to_csv(out_dir / f"{split}_{size}_index.csv", index=False)
    np.save(out_dir / f"{split}_raw_samples.npy", samples_raw)
    print(f"{split}: {imgs.shape} -> {out_dir}")


def load_processed(processed_dir: str | Path, size: int, split: str = "train") -> tuple[np.ndarray, pd.DataFrame]:
    d = resolve(processed_dir)
    imgs = np.load(d / f"{split}_{size}.npy", mmap_mode="r")
    index = pd.read_csv(d / f"{split}_{size}_index.csv")
    return imgs, index


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess raw parquet images.")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", nargs="+", default=["train"])
    a = ap.parse_args()
    for s in a.splits:
        preprocess_split(resolve(a.raw_dir), resolve(a.out_dir), s, a.size, a.workers)


if __name__ == "__main__":
    main()
