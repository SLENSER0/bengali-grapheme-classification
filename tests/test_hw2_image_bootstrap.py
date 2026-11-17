import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import recall_score

from bengali.hw2_image_bootstrap import image_bootstrap
from bengali.utils import COMPONENTS, WEIGHTS


def paired_predictions():
    """Images a/b occur under two checkpoints, and graphemes x/v share a root."""
    rows = [
        (0, "id", "a", "x", 0, 0, 1), (0, "id", "b", "x", 0, 0, 1),
        (0, "id", "c", "v", 0, 1, 1), (0, "id", "d", "w", 1, 1, 0),
        (0, "ood", "e", "q", 1, 1, 0), (0, "ood", "f", "q", 1, 1, 0),
        (1, "id", "g", "u", 1, 0, 0), (1, "id", "h", "u", 1, 0, 0),
        (1, "ood", "a", "x", 0, 0, 1), (1, "ood", "b", "x", 0, 0, 1),
        (1, "ood", "i", "t", 1, 1, 1),
    ]
    base = pd.DataFrame(rows, columns=["fold", "slice", "image_id", "grapheme", *COMPONENTS])
    alt = base.copy()
    for i, c in enumerate(COMPONENTS):
        base[f"pred_{c}"] = base[c] ^ ((np.arange(len(base)) % (i + 3)) == 0)
        alt[f"pred_{c}"] = alt[c] ^ ((np.arange(len(alt)) % (i + 4)) == 1)
    return base, alt


def direct_weighted_scores(base, alt, image_weights):
    rows = base.copy()
    w = rows.image_id.map(image_weights)
    keys = [rows.fold, rows.slice, rows.grapheme]
    rows["weight"] = w * w.groupby(keys).transform("size") / w.groupby(keys).transform("sum")
    endpoints = {}
    for part in ("id", "ood"):
        deltas = []
        for _, group in rows[rows.slice == part].groupby("fold"):
            delta = 0
            for c in COMPONENTS:
                def score(pred):
                    return recall_score(group[c], pred.loc[group.index, f"pred_{c}"],
                                        sample_weight=group.weight, average="macro",
                                        labels=sorted(group[c].unique()), zero_division=0)
                delta += WEIGHTS[c] * (score(alt) - score(base)) / sum(WEIGHTS.values())
            deltas.append(delta)
        endpoints[part] = np.mean(deltas)
    endpoints["main"] = (endpoints["id"] + endpoints["ood"]) / 2
    endpoints["ood_minus_id"] = endpoints["ood"] - endpoints["id"]
    return endpoints


def test_paired_image_intervals_match_direct_stratum_weighted_recall():
    base, alt = paired_predictions()
    image_ids = sorted(set(base.image_id))
    rng = np.random.default_rng(834)
    reference = [direct_weighted_scores(base, alt, dict(zip(image_ids, rng.exponential(size=len(image_ids)))))
                 for _ in range(16)]
    point = direct_weighted_scores(base, alt, dict.fromkeys(image_ids, 1.0))
    result = image_bootstrap(base, alt.sample(frac=1, random_state=19), n_boot=16, seed=834)
    assert (result["rows"], result["unique_images"], result["reused_images"]) == (11, 9, 2)
    for name, value in point.items():
        score = result["contrast"][name]
        assert score["point"] == pytest.approx(value)
        lo, hi = np.quantile([draw[name] for draw in reference], [0.025, 0.975])
        assert score["lo"] == pytest.approx(lo)
        assert score["hi"] == pytest.approx(hi)


def test_shared_multiplier_cancels_between_identical_repeated_slices():
    sample = paired_predictions()[0].iloc[:2]
    base = pd.concat([sample.assign(fold=fold, slice=part) for fold in [0, 1] for part in ["id", "ood"]],
                     ignore_index=True)
    alt = base.copy()
    for c in COMPONENTS:
        base[f"pred_{c}"] = base[c]
        alt[f"pred_{c}"] = alt[c] ^ alt.image_id.eq("a")
    result = image_bootstrap(base, alt, n_boot=20, seed=11)["contrast"]
    assert result["id"]["hi"] > result["id"]["lo"]
    for k in ("point", "lo", "hi"):
        assert result["ood_minus_id"][k] == pytest.approx(0)


def test_mixed_truth_strata_and_misaligned_cells_fail():
    base, alt = paired_predictions()
    with pytest.raises(ValueError):
        image_bootstrap(base, alt.iloc[1:], 5, 42)
    for frame in (base, alt):
        frame.loc[1, COMPONENTS[0]] = 1
    with pytest.raises(ValueError, match="grapheme"):
        image_bootstrap(base, alt, 5, 42)
