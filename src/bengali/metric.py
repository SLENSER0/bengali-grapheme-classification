"""Competition metric: hierarchical macro-averaged recall, plus bootstrap CIs.

Kaggle definition (see README, section "Метрика"):
    R = (2 * R_root + R_vowel + R_consonant) / 4,
where each R_c is sklearn's ``recall_score(average="macro")`` for component c.

Subtlety: sklearn macro-averages over ``union(y_true, y_pred)``; a class that is
predicted but absent from ``y_true`` contributes recall 0. On the full test set
all classes are present, so this does not matter. On *slices* (e.g. the OOD
slice that contains only ~1/10 of graphemes) it would artificially punish the
model, so by default we average over classes present in ``y_true``
(``labels="true"``). ``labels="union"`` reproduces sklearn/Kaggle exactly;
both coincide whenever every predicted class is present in ``y_true``
(checked in tests/test_metric.py).
"""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd

from .utils import COMPONENTS, N_CLASSES, SHORT, WEIGHTS

Labels = Literal["true", "union"]


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    if n_classes is not None:
        n = max(n, n_classes)
    support = np.bincount(y_true, minlength=n)
    tp = np.bincount(y_true[y_true == y_pred], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support > 0, tp / np.maximum(support, 1), np.nan)
    return pd.DataFrame({"cls": np.arange(n), "support": support, "tp": tp, "recall": recall})


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray, labels: Labels = "true") -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return float("nan")
    n = int(max(y_true.max(), y_pred.max()) + 1)
    support = np.bincount(y_true, minlength=n)
    tp = np.bincount(y_true[y_true == y_pred], minlength=n)
    if labels == "true":
        cls = np.flatnonzero(support)
    elif labels == "union":
        cls = np.union1d(np.unique(y_true), np.unique(y_pred))
    else:
        raise ValueError(labels)
    rec = np.where(support[cls] > 0, tp[cls] / np.maximum(support[cls], 1), 0.0)
    return float(rec.mean())


def hierarchical_recall(
    y_true: Mapping[str, np.ndarray] | pd.DataFrame,
    y_pred: Mapping[str, np.ndarray] | pd.DataFrame,
    labels: Labels = "true",
) -> dict[str, float]:
    """Return per-component macro recall and the weighted competition score."""
    out: dict[str, float] = {}
    total, wsum = 0.0, 0.0
    for c in COMPONENTS:
        r = macro_recall(np.asarray(y_true[c]), np.asarray(y_pred[c]), labels=labels)
        out[SHORT[c]] = r
        total += WEIGHTS[c] * r
        wsum += WEIGHTS[c]
    out["score"] = total / wsum
    return out


def score_frame(df: pd.DataFrame, labels: Labels = "true") -> dict[str, float]:
    """Score a prediction frame with columns `<component>` (truth) and `pred_<component>`."""
    truth = {c: df[c].to_numpy() for c in COMPONENTS}
    pred = {c: df[f"pred_{c}"].to_numpy() for c in COMPONENTS}
    return hierarchical_recall(truth, pred, labels=labels)


def slice_scores(df: pd.DataFrame) -> dict[str, float]:
    """Scores used throughout the project.

    * ``id``   – images of graphemes seen in training (in-dictionary slice);
    * ``ood``  – images of graphemes fully held out from training;
    * ``main`` – (id + ood) / 2, the primary model-selection score: it weighs the
      seen and the unseen-grapheme regimes equally and does not depend on the
      (arbitrary) relative size of the two slices.
    """
    res: dict[str, float] = {}
    for name, part in (("id", df[df["slice"] == "id"]), ("ood", df[df["slice"] == "ood"])):
        if len(part) == 0:
            continue
        s = score_frame(part)
        for k, v in s.items():
            res[f"{name}_{k}"] = v
        res[f"{name}_n"] = int(len(part))
    if "id_score" in res and "ood_score" in res:
        res["main_score"] = (res["id_score"] + res["ood_score"]) / 2
    # Kaggle-exact score on the whole validation set (both slices together).
    res["all_score_kaggle"] = score_frame(df, labels="union")["score"]
    return res


# ---------------------------------------------------------------------------------------------- bootstrap
#
# What is bootstrapped. The reported number is the *mean over folds* of the per-fold slice scores
# (id, ood, main = (id + ood) / 2), so every bootstrap replicate recomputes exactly that: it resamples
# inside every fold and slice, scores each fold, then averages over folds.
#
# Resampling scheme. Within a fold and slice, images are drawn with replacement *inside each grapheme*
# (stratified by grapheme). The set of graphemes, and therefore the set of classes the macro average runs
# over, is identical to the point estimate. This captures the noise from the finite sample of handwritten
# instances (≈ writers) given the fold's graphemes and model. Variability from *which* graphemes are held out
# is captured by the spread across folds (each fold holds out a different 10% of graphemes, reported as std).
#
# Why not a cluster bootstrap over OOD graphemes (the first version of this code). More than half of the roots in
# a fold's OOD slice (48-68%) occur in a single OOD grapheme; a cluster resample drops each such grapheme with p≈0.37, the
# macro average then silently loses these mostly hard, rare roots and the bootstrap distribution shifts upward
# (≈ +1 pp OOD for B2 on fold 0), so the percentile CI no longer brackets the estimate. See README §5.
#
# Pairing. All models are resampled with the *same* index draws (rows are aligned by image_id), so differences
# between models are paired and their CI is much narrower than the two marginal CIs suggest.


def _slice_layout(graphemes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order rows by grapheme; for each (sorted) row return the start and size of its grapheme block."""
    order = np.argsort(graphemes, kind="stable")
    g = graphemes[order]
    new = np.r_[True, g[1:] != g[:-1]] if len(g) else np.zeros(0, bool)
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    sizes = np.diff(np.r_[starts, len(g)])
    return order, starts[gid], sizes[gid]


def _boot_slice(cls: list[np.ndarray], correct: list[np.ndarray], row_start, row_size, rng, n_boot, chunk=50) -> np.ndarray:
    """Replicates of the hierarchical score for one fold/slice; returns [n_models, n_boot]."""
    n = len(row_start)
    k = [N_CLASSES[c] for c in COMPONENTS]
    den = [np.bincount(cls[j], minlength=k[j]).astype(float) for j in range(3)]
    present = [den[j] > 0 for j in range(3)]
    w = np.array([WEIGHTS[c] for c in COMPONENTS], dtype=float)
    out = np.empty((len(correct), n_boot))
    for b0 in range(0, n_boot, chunk):
        m = min(chunk, n_boot - b0)
        pick = row_start + (rng.random((m, n)) * row_size).astype(np.int64)  # same grapheme, random member
        offs = np.arange(m)[:, None]
        for mi, corr in enumerate(correct):
            score = np.zeros(m)
            for j in range(3):
                idx = (cls[j][None, :] + k[j] * offs).ravel()
                num = np.bincount(idx, weights=corr[pick, j].ravel(), minlength=m * k[j]).reshape(m, k[j])
                score += w[j] * (num[:, present[j]] / den[j][present[j]]).mean(1)
            out[mi, b0 : b0 + m] = score / w.sum()
    return out


def _fold_bootstrap(frames: list[pd.DataFrame], n_boot: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Point estimates and replicates of fold-averaged id/ood/main for several *aligned* prediction frames."""
    rng = np.random.default_rng(seed)
    base = frames[0]
    folds = sorted(base["fold"].unique()) if "fold" in base else [None]
    per_fold_pts = [{"id": [], "ood": []} for _ in frames]
    per_fold_reps = [{"id": [], "ood": []} for _ in frames]
    for f in folds:
        for sl in ("id", "ood"):
            mask = (base["slice"] == sl).to_numpy()
            if f is not None:
                mask = mask & (base["fold"] == f).to_numpy()
            if not mask.any():
                continue
            order, rs, rz = _slice_layout(base.loc[mask, "grapheme"].to_numpy())
            cls = [base.loc[mask, c].to_numpy().astype(np.int64)[order] for c in COMPONENTS]
            correct = []
            for fr in frames:
                part = fr.loc[mask]
                correct.append(np.stack([(part[c].to_numpy() == part[f"pred_{c}"].to_numpy())[order] for c in COMPONENTS], 1).astype(float))
            reps = _boot_slice(cls, correct, rs, rz, rng, n_boot)
            for mi, fr in enumerate(frames):
                per_fold_pts[mi][sl].append(score_frame(fr.loc[mask])["score"])
                per_fold_reps[mi][sl].append(reps[mi])
    points, reps_out = [], []
    for mi in range(len(frames)):
        pt = {sl: float(np.mean(v)) for sl, v in per_fold_pts[mi].items() if v}
        rp = {sl: np.mean(v, axis=0) for sl, v in per_fold_reps[mi].items() if v}
        if "id" in pt and "ood" in pt:
            pt["main"] = (pt["id"] + pt["ood"]) / 2
            rp["main"] = (rp["id"] + rp["ood"]) / 2
        points.append(pt)
        reps_out.append(rp)
    return points, reps_out


def _summarise(point: float, reps: np.ndarray, alpha: float) -> dict[str, float]:
    return {
        "point": float(point),
        "mean": float(reps.mean()),
        "std": float(reps.std(ddof=1)),
        "lo": float(np.quantile(reps, alpha / 2)),
        "hi": float(np.quantile(reps, 1 - alpha / 2)),
    }


def _align(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    # an image can occur twice in out-of-fold predictions: in val_ID of its id_fold and in val_OOD of its ood_group
    key = [k for k in ("image_id", "fold", "slice") if k in ref]
    assert len(df) == len(ref), "prediction frames cover different rows"
    out = df.set_index(key).loc[pd.MultiIndex.from_frame(ref[key])].reset_index()
    assert (out["grapheme"].to_numpy() == ref["grapheme"].to_numpy()).all(), "prediction frames come from different splits"
    return out


def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CI of the fold-averaged id/ood/main scores (see the comment block above).

    ``df`` holds out-of-fold predictions with columns fold (optional), slice, grapheme, the three components
    and ``pred_<component>``. Returns {"id"|"ood"|"main": {point, mean, std, lo, hi}}; ``point`` equals the
    mean over folds of :func:`slice_scores`.
    """
    (pt,), (rp,) = _fold_bootstrap([df.reset_index(drop=True)], n_boot, seed)
    return {k: _summarise(pt[k], rp[k], alpha) for k in pt}


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, dict[str, float]]:
    """CI of the paired difference a − b of fold-averaged id/ood/main (same resampled images for both)."""
    a = df_a.reset_index(drop=True)
    b = _align(df_b, a)
    (pa, pb), (ra, rb) = _fold_bootstrap([a, b], n_boot, seed)
    return {k: _summarise(pa[k] - pb[k], ra[k] - rb[k], alpha) for k in pa}
