import numpy as np
import pandas as pd

from bengali.splits import check_folds, fold_indices, make_folds
from bengali.utils import COMPONENTS, N_CLASSES


def synthetic_train(seed=0, n_graphemes=400, per_grapheme=(20, 60)):
    """Grapheme table that mimics the real structure: some roots exist in a single grapheme."""
    rng = np.random.default_rng(seed)
    combos = set()
    # 11 roots that never take diacritics (=> a single grapheme each, never holdable); in the real data these are
    # 6 vowel roots, ং, ঃ and 16 other roots
    for r in range(11):
        combos.add((r, 0, 0))
    # every other class appears at least a few times
    for r in range(11, 168):
        for _ in range(2):
            combos.add((r, int(rng.integers(0, 11)), int(rng.integers(0, 7))))
    while len(combos) < n_graphemes:
        combos.add((int(rng.integers(11, 168)), int(rng.integers(0, 11)), int(rng.integers(0, 7))))
    rows = []
    i = 0
    for r, v, c in sorted(combos):
        for _ in range(int(rng.integers(*per_grapheme))):
            rows.append({"image_id": f"Train_{i}", "grapheme_root": r, "vowel_diacritic": v, "consonant_diacritic": c, "grapheme": f"{r}_{v}_{c}"})
            i += 1
    return pd.DataFrame(rows)


def test_folds_have_no_leakage_and_cover_all_classes():
    train = synthetic_train()
    folds = make_folds(train, n_folds=5, ood_frac=0.1, seed=42)
    rep = check_folds(folds, 5)  # asserts disjointness, no OOD grapheme in train, all classes in train
    assert len(rep) == 5
    for r in rep:
        assert r["n_val_ood"] > 0 and r["n_val_id"] > 0


def test_ood_groups_are_disjoint_and_sized():
    train = synthetic_train()
    folds = make_folds(train, n_folds=5, ood_frac=0.1, seed=42)
    g = folds.drop_duplicates("grapheme")
    counts = g[g.ood_group >= 0].groupby("ood_group").size()
    target = round(0.1 * g.grapheme.nunique())
    assert (counts == target).all(), counts
    # each grapheme belongs to at most one group (by construction of the column); a root that exists in a single
    # grapheme (here roots 0..10) can never be held out, otherwise it would vanish from train
    assert (g[g.grapheme_root < 11].ood_group == -1).all()


def test_partition_of_images():
    train = synthetic_train()
    folds = make_folds(train, n_folds=5, ood_frac=0.1, seed=42)
    for k in range(5):
        ix = fold_indices(folds, k)
        allv = np.concatenate([ix["train"], ix["val_id"], ix["val_ood"]])
        assert len(np.unique(allv)) == len(allv)
        assert len(allv) == len(folds)  # train/val_id/val_ood partition all images


def test_deterministic():
    train = synthetic_train()
    a = make_folds(train, seed=7)
    b = make_folds(train, seed=7)
    pd.testing.assert_frame_equal(a, b)
    c = make_folds(train, seed=8)
    assert not a["ood_group"].equals(c["ood_group"])


def test_id_folds_are_stratified_by_grapheme():
    train = synthetic_train()
    folds = make_folds(train, seed=1)
    per = folds.groupby(["grapheme", "id_fold"]).size().unstack(fill_value=0)
    assert (per.max(axis=1) - per.min(axis=1)).max() <= 1
    assert set(N_CLASSES) == set(COMPONENTS)
