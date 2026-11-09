import numpy as np
import pandas as pd
import pytest
import sklearn.metrics

from bengali.metric import bootstrap_ci, hierarchical_recall, macro_recall, paired_bootstrap, slice_scores
from bengali.utils import COMPONENTS, N_CLASSES


def kaggle_reference(truth: dict, pred: dict) -> float:
    """The snippet from the competition Evaluation page (with its column typo fixed)."""
    scores = []
    for c in ["grapheme_root", "consonant_diacritic", "vowel_diacritic"]:
        scores.append(sklearn.metrics.recall_score(truth[c], pred[c], average="macro", zero_division=0))
    order = {"grapheme_root": 2, "consonant_diacritic": 1, "vowel_diacritic": 1}
    return float(np.average(scores, weights=list(order.values())))


def random_labels(rng, n, noise=0.3):
    truth = {c: rng.integers(0, N_CLASSES[c], n) for c in COMPONENTS}
    pred = {c: np.where(rng.random(n) < noise, rng.integers(0, N_CLASSES[c], n), truth[c]) for c in COMPONENTS}
    return truth, pred


@pytest.mark.parametrize("seed", range(5))
def test_matches_kaggle_reference_union(seed):
    rng = np.random.default_rng(seed)
    truth, pred = random_labels(rng, 3000)
    ours = hierarchical_recall(truth, pred, labels="union")["score"]
    assert ours == pytest.approx(kaggle_reference(truth, pred), abs=1e-12)


def test_true_and_union_coincide_when_all_classes_present():
    rng = np.random.default_rng(0)
    truth, pred = random_labels(rng, 20000)  # every class present in y_true
    a = hierarchical_recall(truth, pred, labels="true")["score"]
    b = hierarchical_recall(truth, pred, labels="union")["score"]
    assert a == pytest.approx(b, abs=1e-12)


def test_union_penalises_classes_absent_from_truth():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 2, 1, 1])  # class 2 predicted but absent in truth
    assert macro_recall(y_true, y_pred, "true") == pytest.approx((0.5 + 1.0) / 2)
    assert macro_recall(y_true, y_pred, "union") == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert macro_recall(y_true, y_pred, "union") == pytest.approx(
        sklearn.metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
    )


def test_constant_predictor_is_one_over_k():
    rng = np.random.default_rng(1)
    truth = {c: rng.integers(0, N_CLASSES[c], 50000) for c in COMPONENTS}
    pred = {c: np.zeros_like(truth[c]) for c in COMPONENTS}
    s = hierarchical_recall(truth, pred)
    assert s["root"] == pytest.approx(1 / 168)
    assert s["score"] == pytest.approx((2 / 168 + 1 / 11 + 1 / 7) / 4)


def test_perfect_prediction():
    rng = np.random.default_rng(2)
    truth, _ = random_labels(rng, 1000)
    assert hierarchical_recall(truth, truth)["score"] == 1.0


def _pred_frame(rng, n_id=2000, n_ood=1000, n_folds=1, noise_id=0.2, noise_ood=0.5):
    """OOF-like frame: every grapheme is a fixed label triple, as in the real data."""
    rows, img = [], 0
    for fold in range(n_folds):
        for sl, n, noise in (("id", n_id, noise_id), ("ood", n_ood, noise_ood)):
            n_g = 50
            g_triples = {c: rng.integers(0, N_CLASSES[c], n_g) for c in COMPONENTS}
            g = np.arange(n) % n_g
            f = pd.DataFrame({c: g_triples[c][g] for c in COMPONENTS})
            for c in COMPONENTS:
                f[f"pred_{c}"] = np.where(rng.random(n) < noise, rng.integers(0, N_CLASSES[c], n), f[c])
            f["slice"] = sl
            f["fold"] = fold
            f["grapheme"] = [f"f{fold}{sl}{i}" for i in g]
            f["image_id"] = [f"Train_{img + i}" for i in range(n)]
            img += n
            rows.append(f)
    return pd.concat(rows, ignore_index=True)


def test_slice_scores():
    df = _pred_frame(np.random.default_rng(3))
    s = slice_scores(df)
    assert s["id_score"] > s["ood_score"]
    assert s["main_score"] == pytest.approx((s["id_score"] + s["ood_score"]) / 2)


def test_bootstrap_point_is_fold_mean_and_inside_ci():
    df = _pred_frame(np.random.default_rng(4), n_folds=3)
    per_fold = [slice_scores(d) for _, d in df.groupby("fold")]
    ci = bootstrap_ci(df, n_boot=300, seed=0)
    for k in ("id", "ood", "main"):
        assert ci[k]["point"] == pytest.approx(np.mean([s[f"{k}_score"] for s in per_fold]), abs=1e-12)
        assert ci[k]["lo"] < ci[k]["point"] < ci[k]["hi"]
        # stratified resampling keeps the class set fixed -> no systematic shift of the replicates
        assert abs(ci[k]["mean"] - ci[k]["point"]) < 0.25 * ci[k]["std"] + 1e-3


def test_bootstrap_keeps_rare_hard_class():
    """A class that lives in one grapheme and is always wrong must stay in every replicate (old cluster bug)."""
    df = _pred_frame(np.random.default_rng(5), n_id=10, n_ood=3000, noise_ood=0.1)
    ood = df.slice == "ood"
    rare = ood & (df.grapheme == df[ood].grapheme.iloc[0])
    df.loc[ood, "grapheme_root"] = 1
    df.loc[ood, "pred_grapheme_root"] = 1
    df.loc[rare, "grapheme_root"] = 0  # rare class with recall 0
    ci = bootstrap_ci(df, n_boot=200, seed=0)
    assert ci["ood"]["mean"] == pytest.approx(ci["ood"]["point"], abs=0.01)


def test_paired_bootstrap():
    rng = np.random.default_rng(6)
    a = _pred_frame(rng, n_folds=2)
    same = paired_bootstrap(a, a.sample(frac=1, random_state=0), n_boot=50)
    for k in ("id", "ood", "main"):
        assert same[k]["lo"] == same[k]["hi"] == same[k]["point"] == 0
    b = a.copy()
    flip = rng.random(len(b)) < 0.1
    b.loc[flip, "pred_grapheme_root"] = (b.loc[flip, "grapheme_root"] + 1) % 168  # b is strictly worse on root
    d = paired_bootstrap(a, b, n_boot=200)
    ca, cb = bootstrap_ci(a, n_boot=10), bootstrap_ci(b, n_boot=10)
    assert d["main"]["point"] == pytest.approx(ca["main"]["point"] - cb["main"]["point"], abs=1e-12)
    assert d["main"]["lo"] > 0
