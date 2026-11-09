"""Validation protocol: 5 folds, each with an in-dictionary (ID) and an
out-of-dictionary (OOD, unseen graphemes) validation slice.

Why: the hidden Kaggle test set (a) contains ~900 graphemes absent from train
(88% of them in the private part) and (b) has no writer overlap with train.
A plain random/stratified split measures only the "seen grapheme" regime, which
is what made public LB (0.995) so much more optimistic than private (0.976).

Construction (deterministic given the seed):
1. ``id_fold``: StratifiedKFold over all images, stratified by grapheme, so each
   seen grapheme is represented in every ID validation fold.
2. ``ood_group``: a disjoint subset of graphemes is assigned to each fold
   (~``ood_frac`` of all graphemes per fold). A grapheme may be held out only
   if, after removing *its whole group*, every class of every component still
   occurs in training (otherwise we would test on a component class the model
   has never seen, which the competition explicitly rules out: "the test set
   ... has no new grapheme components").
3. Fold k:  train   = id_fold != k  and ood_group != k
            val_id  = id_fold == k  and ood_group != k
            val_ood = ood_group == k
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .utils import COMPONENTS, N_CLASSES, resolve


def assign_ood_groups(
    graphemes: pd.DataFrame,
    n_folds: int,
    ood_frac: float,
    seed: int,
) -> dict[str, int]:
    """Greedy constrained assignment of graphemes to OOD groups.

    ``graphemes``: one row per grapheme with the three component columns.
    Returns {grapheme: group_id}; graphemes not held out are absent.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(graphemes))
    target = int(round(ood_frac * len(graphemes)))
    # How many graphemes contain each (component, class); per group we track how
    # many of them would still remain in training.
    total = {c: Counter(graphemes[c].tolist()) for c in COMPONENTS}
    removed = [{c: Counter() for c in COMPONENTS} for _ in range(n_folds)]
    sizes = [0] * n_folds
    out: dict[str, int] = {}
    for i in order:
        row = graphemes.iloc[i]
        # put into the currently smallest group that accepts it
        for g in np.argsort(sizes, kind="stable"):
            if sizes[g] >= target:
                continue
            ok = all(total[c][row[c]] - removed[g][c][row[c]] > 1 for c in COMPONENTS)
            if ok:
                for c in COMPONENTS:
                    removed[g][c][row[c]] += 1
                sizes[g] += 1
                out[row["grapheme"]] = int(g)
                break
        if all(s >= target for s in sizes):
            break
    return out


def make_folds(train: pd.DataFrame, n_folds: int = 5, ood_frac: float = 0.1, seed: int = 42) -> pd.DataFrame:
    df = train[["image_id", "grapheme", *COMPONENTS]].copy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["id_fold"] = -1
    for k, (_, va) in enumerate(skf.split(df, df["grapheme"])):
        df.iloc[va, df.columns.get_loc("id_fold")] = k
    g = df.drop_duplicates("grapheme")[["grapheme", *COMPONENTS]].reset_index(drop=True)
    groups = assign_ood_groups(g, n_folds, ood_frac, seed)
    df["ood_group"] = df["grapheme"].map(groups).fillna(-1).astype(int)
    return df[["image_id", "grapheme", *COMPONENTS, "id_fold", "ood_group"]]


LABEL_COLS = ["grapheme", *COMPONENTS]
FOLD_COLS = ["image_id", "id_fold", "ood_group"]


def load_folds(folds_file: str = "splits/folds.csv", train_csv: str = "data/raw/train.csv") -> pd.DataFrame:
    """Fold assignment joined with labels.

    The committed fold file holds only ``image_id, id_fold, ood_group``: the competition rules forbid
    redistributing the data, labels included. Labels and the grapheme string are joined from the
    competition's own train.csv (row order of the fold file is kept).
    """
    folds = pd.read_csv(resolve(folds_file))
    if all(c in folds for c in LABEL_COLS):
        return folds
    train = pd.read_csv(resolve(train_csv), usecols=["image_id", *LABEL_COLS])
    out = folds[FOLD_COLS].merge(train, on="image_id", how="left", validate="one_to_one")
    assert out[COMPONENTS].notna().all().all(), "fold file and train.csv disagree"
    return out[["image_id", *LABEL_COLS, "id_fold", "ood_group"]]


def fold_indices(folds: pd.DataFrame, k: int) -> dict[str, np.ndarray]:
    """Positional indices (into ``folds``) of train / val_id / val_ood for fold k."""
    idf = folds["id_fold"].to_numpy()
    oog = folds["ood_group"].to_numpy()
    return {
        "train": np.flatnonzero((idf != k) & (oog != k)),
        "val_id": np.flatnonzero((idf == k) & (oog != k)),
        "val_ood": np.flatnonzero(oog == k),
    }


def check_folds(folds: pd.DataFrame, n_folds: int) -> list[dict]:
    """Sanity checks used by tests and printed by the CLI."""
    report = []
    for k in range(n_folds):
        ix = fold_indices(folds, k)
        tr, vi, vo = (folds.iloc[ix[s]] for s in ("train", "val_id", "val_ood"))
        assert not (set(ix["train"]) & set(ix["val_id"])), "train/val_id overlap"
        assert not (set(ix["train"]) & set(ix["val_ood"])), "train/val_ood overlap"
        assert not (set(tr["grapheme"]) & set(vo["grapheme"])), "OOD grapheme leaked into train"
        for c in COMPONENTS:
            missing = set(range(N_CLASSES[c])) - set(tr[c])
            present = set(folds[c])
            assert not (missing & present), f"fold {k}: classes {sorted(missing & present)} of {c} absent from train"
        report.append(
            {
                "fold": k,
                "n_train": len(tr),
                "n_val_id": len(vi),
                "n_val_ood": len(vo),
                "graphemes_train": tr["grapheme"].nunique(),
                "graphemes_val_ood": vo["grapheme"].nunique(),
                # label-space leak: OOD images whose (root, vowel, cons) triple exists in train under another spelling
                "ood_images_with_seen_triple": int(
                    pd.Series(list(map(tuple, vo[COMPONENTS].to_numpy()))).isin(set(map(tuple, tr[COMPONENTS].drop_duplicates().to_numpy()))).sum()
                ),
            }
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the deterministic fold file.")
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--out", default="splits/folds.csv")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--ood-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    train = pd.read_csv(resolve(a.train_csv))
    folds = make_folds(train, a.n_folds, a.ood_frac, a.seed)
    rep = pd.DataFrame(check_folds(folds, a.n_folds))
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    folds[FOLD_COLS].to_csv(out, index=False)  # no labels in the committed file (see load_folds)
    rep.to_csv(out.with_name("folds_summary.csv"), index=False)
    print(rep.to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
