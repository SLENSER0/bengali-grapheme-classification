"""Exploratory data analysis: numbers for the README + figures in reports/figures.

Usage: python -m bengali.eda
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import COMPONENTS, N_CLASSES, SHORT, resolve, save_json

VOWEL_ROOTS = set("অআইঈউঊঋঌএঐওঔ")
VIRAMA = "্"


def root_type(glyph: str) -> str:
    if glyph in VOWEL_ROOTS:
        return "vowel"
    if VIRAMA in glyph:
        return "conjunct"
    return "consonant"


def load_class_map(raw_dir) -> pd.DataFrame:
    cm = pd.read_csv(raw_dir / "class_map.csv")
    return cm


def root_types(raw_dir) -> pd.Series:
    cm = load_class_map(raw_dir)
    roots = cm[cm.component_type == "grapheme_root"].set_index("label")["component"]
    return roots.map(root_type)


def near_duplicates(imgs: np.ndarray, graphemes: np.ndarray, thr: float = 0.95, chunk: int = 8192) -> pd.DataFrame:
    """Pairs of near-identical images of the same grapheme (same handwritten sample stored twice, different bytes).

    md5 only finds byte-identical files. Here every image is reduced to 32x32 (block mean of the 128x128 crop),
    and within each grapheme the Pearson correlation of all pairs is computed. The distribution of the best
    within-grapheme correlation is bimodal (ordinary neighbours mostly ~0.4-0.75, median ~0.55; copies >0.95;
    almost nothing in between), so the threshold is not sensitive.
    """
    n, s = imgs.shape[0], imgs.shape[1]
    small = np.empty((n, 32 * 32), dtype=np.float32)
    for a in range(0, n, chunk):
        b = np.asarray(imgs[a : a + chunk], dtype=np.float32)
        small[a : a + len(b)] = b.reshape(len(b), 32, s // 32, 32, s // 32).mean((2, 4)).reshape(len(b), -1)
    small -= small.mean(1, keepdims=True)
    small /= np.linalg.norm(small, axis=1, keepdims=True) + 1e-6
    rows = []
    for g, idx in pd.Series(np.arange(n)).groupby(graphemes).groups.items():
        idx = np.asarray(idx)
        c = small[idx] @ small[idx].T
        i, j = np.nonzero(np.triu(c, 1) > thr)
        rows += [(idx[a], idx[b], g, float(c[a, b])) for a, b in zip(i, j)]
    return pd.DataFrame(rows, columns=["row_a", "row_b", "grapheme", "corr"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    raw, proc, out = resolve(a.raw_dir), resolve(a.processed_dir), resolve(a.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(raw / "train.csv")
    cm = load_class_map(raw)
    rt = root_types(raw)
    index = pd.read_csv(proc / f"train_{a.size}_index.csv")
    assert (index.image_id.values == train.image_id.values).all()
    summary: dict = {}

    # ---------------------------------------------------------------- sizes, missing, dtypes
    summary["n_images"] = len(train)
    summary["n_graphemes"] = int(train.grapheme.nunique())
    summary["missing_values"] = int(train.isna().sum().sum())
    summary["columns"] = {c: str(t) for c, t in train.dtypes.items()}
    summary["n_classes_present"] = {SHORT[c]: int(train[c].nunique()) for c in COMPONENTS}
    summary["class_map_sizes"] = cm.groupby("component_type").size().to_dict()
    summary["root_types"] = rt.value_counts().to_dict()

    # ---------------------------------------------------------------- imbalance
    imb = {}
    for c in COMPONENTS:
        vc = train[c].value_counts()
        imb[SHORT[c]] = {
            "max": int(vc.max()),
            "min": int(vc.min()),
            "ratio_max_min": float(vc.max() / vc.min()),
            "median": float(vc.median()),
            "top_class": int(vc.idxmax()),
            "top_share": float(vc.max() / len(train)),
        }
    summary["imbalance"] = imb
    per_g = train.grapheme.value_counts()
    summary["images_per_grapheme"] = {
        "min": int(per_g.min()),
        "median": float(per_g.median()),
        "max": int(per_g.max()),
        "p05": float(per_g.quantile(0.05)),
    }
    g = train.drop_duplicates("grapheme")
    gpr = g.grapheme_root.value_counts()
    summary["graphemes_per_root"] = {
        "min": int(gpr.min()),
        "median": float(gpr.median()),
        "max": int(gpr.max()),
        "roots_in_single_grapheme": int((gpr == 1).sum()),
    }
    summary["possible_combinations"] = 168 * 11 * 7
    summary["share_of_combinations_seen"] = summary["n_graphemes"] / summary["possible_combinations"]

    # ---------------------------------------------------------------- label-noise / leakage hints
    multi = raw / "train_multi_diacritics.csv"
    if multi.exists():
        md = pd.read_csv(multi)
        summary["multi_diacritic_rows"] = int(len(md))
    dup = index.md5.duplicated(keep=False)
    summary["exact_duplicate_images"] = int(dup.sum())
    if dup.any():
        dd = index[dup].merge(train, on="image_id")
        conflicting = dd.groupby("md5").grapheme.nunique()
        summary["duplicate_groups_with_conflicting_labels"] = int((conflicting > 1).sum())
    imgs_mm = np.load(proc / f"train_{a.size}.npy", mmap_mode="r")
    nd = near_duplicates(imgs_mm, train.grapheme.to_numpy())
    nd.insert(0, "image_id_b", train.image_id.to_numpy()[nd.row_b])
    nd.insert(0, "image_id_a", train.image_id.to_numpy()[nd.row_a])
    nd.drop(columns=["row_a", "row_b", "grapheme"]).to_csv(out / "near_duplicates.csv", index=False)  # ids only, no labels
    summary["near_duplicate_pairs"] = int(len(nd))
    summary["images_with_near_duplicate"] = int(pd.unique(np.r_[nd.row_a, nd.row_b]).size)
    summary["bbox"] = {k: {"median": float(index[k].median()), "p01": float(index[k].quantile(0.01)), "p99": float(index[k].quantile(0.99))} for k in ("bbox_h", "bbox_w", "ink_frac")}
    summary["near_empty_images"] = int((index.ink_frac < 0.005).sum())
    # does the 'grapheme' string agree with components? (consistency check of labels)
    comp = train.groupby("grapheme")[COMPONENTS].nunique()
    summary["graphemes_with_inconsistent_components"] = int((comp > 1).any(axis=1).sum())
    save_json(summary, out / "eda_summary.json")

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, c in zip(axes, COMPONENTS):
        vc = train[c].value_counts().reindex(range(N_CLASSES[c]), fill_value=0).sort_values(ascending=False)
        if c == "grapheme_root":
            colors = [{"vowel": "#d62728", "consonant": "#1f77b4", "conjunct": "#2ca02c"}[rt[i]] for i in vc.index]
        else:
            colors = "#1f77b4"
        ax.bar(range(len(vc)), vc.values, color=colors, width=0.9)
        ax.set_yscale("log")
        ax.set_title(f"{c}: {N_CLASSES[c]} classes (sorted)")
        ax.set_xlabel("class rank")
        ax.set_ylabel("images (log)")
    from matplotlib.patches import Patch

    axes[0].legend(handles=[Patch(color=v, label=k) for k, v in {"vowel": "#d62728", "consonant": "#1f77b4", "conjunct": "#2ca02c"}.items()], fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_class_distribution.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    axes[0].hist(per_g.values, bins=60, color="#555")
    axes[0].set_title(f"images per grapheme ({summary['n_graphemes']} graphemes)")
    axes[0].set_xlabel("images")
    axes[1].hist(gpr.values, bins=40, color="#555")
    axes[1].set_title("distinct graphemes per root")
    axes[1].set_xlabel("graphemes")
    co = g.groupby(["vowel_diacritic", "consonant_diacritic"]).size().unstack(fill_value=0)
    im = axes[2].imshow(co.values, cmap="viridis", aspect="auto")
    axes[2].set_title("graphemes per (vowel, consonant) diacritic pair")
    axes[2].set_xlabel("consonant_diacritic")
    axes[2].set_ylabel("vowel_diacritic")
    axes[2].set_xticks(range(co.shape[1]))
    axes[2].set_yticks(range(co.shape[0]))
    for (i, j), v in np.ndenumerate(co.values):
        axes[2].text(j, i, v, ha="center", va="center", color="w" if v < co.values.max() / 2 else "k", fontsize=7)
    fig.colorbar(im, ax=axes[2])
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_graphemes.png", dpi=130)
    plt.close(fig)

    raw_s = np.load(proc / "train_raw_samples.npy")
    proc_imgs = np.load(proc / f"train_{a.size}.npy", mmap_mode="r")
    n = 8
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(raw_s[i], cmap="gray")
        axes[0, i].set_title(f"{train.grapheme_root[i]}/{train.vowel_diacritic[i]}/{train.consonant_diacritic[i]}", fontsize=8)
        axes[1, i].imshow(proc_imgs[i], cmap="gray")
        for r in range(2):
            axes[r, i].axis("off")
    axes[0, 0].text(-30, 68, "raw\n137x236", fontsize=8, ha="right")
    axes[1, 0].text(-10, 64, f"crop\n{a.size}x{a.size}", fontsize=8, ha="right")
    fig.suptitle("raw images vs preprocessed crops (title: root/vowel/consonant)")
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_examples.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    for ax, k in zip(axes, ("bbox_h", "bbox_w", "ink_frac")):
        ax.hist(index[k], bins=80, color="#555")
        ax.set_title(k)
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_image_stats.png", dpi=130)
    plt.close(fig)

    # markdown tables for README
    lines = ["| компонент | классов | max | min | max/min | доля топ-класса |", "|---|---:|---:|---:|---:|---:|"]
    for c in COMPONENTS:
        s = imb[SHORT[c]]
        lines.append(f"| {c} | {N_CLASSES[c]} | {s['max']} | {s['min']} | {s['ratio_max_min']:.1f} | {s['top_share']:.1%} |")
    (out / "eda_tables.md").write_text("\n".join(lines) + "\n")
    print(pd.Series({k: v for k, v in summary.items() if not isinstance(v, dict)}).to_string())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
