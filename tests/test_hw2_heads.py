import numpy as np
import pytest

from bengali.hw2_controls import control_logits
from bengali.hw2_heads import fit_heads, fuse_heads
from bengali.utils import COMPONENTS


@pytest.fixture
def train():
    rng = np.random.default_rng(17)
    n = 31
    labels = np.stack([rng.permutation(n) % k for k in (4, 3, 2)], axis=1)
    stages = {s: rng.normal(size=(n, d)).astype("f") for s, d in (("layer3", 5), ("layer4", 7))}
    logits = {c: rng.normal(size=(n, k)).astype("f") for c, k in zip(COMPONENTS, (4, 3, 2))}
    return stages, logits, labels


def test_train_moments_match_explicit_within_class_residuals(train):
    stages, logits, labels = train
    priors, models = fit_heads(*train, ridge=0.17)
    for j, c in enumerate(COMPONENTS):
        probs = np.exp(logits[c].astype("d"))
        probs /= probs.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(priors[c], np.log(probs.mean(axis=0)), rtol=2e-6)
        for s, raw in stages.items():
            x = raw.astype("d") / np.linalg.norm(raw.astype("d"), axis=1, keepdims=True)
            means = np.stack([x[labels[:, j] == k].mean(axis=0) for k in range(probs.shape[1])])
            means /= np.linalg.norm(means, axis=1, keepdims=True)
            residuals = x - means[labels[:, j]]
            cov = residuals.T @ residuals / len(x)
            cov += np.eye(x.shape[1]) * (0.17 * np.trace(cov) / x.shape[1])
            w = np.linalg.solve(cov, means.T)
            b = -0.5 * np.sum(means.T * w, axis=0)
            weights, bias = models[s][c]
            assert weights.dtype == bias.dtype == np.float32
            np.testing.assert_allclose(weights, w, rtol=2e-6, atol=1e-6)
            np.testing.assert_allclose(bias, b, rtol=2e-6, atol=1e-6)


def test_fusion_matches_independent_additive_terms(train):
    stages, logits, _ = train
    priors, models = fit_heads(*train, ridge=1)
    got = fuse_heads(stages, logits, priors, models, alpha=1.4, prior_strength=0.6)
    for c in COMPONENTS:
        auxiliary = []
        for s, raw in stages.items():
            w, b = models[s][c]
            x = raw.astype("d") / np.linalg.norm(raw.astype("d"), axis=1, keepdims=True)
            g = x @ w + b
            auxiliary.append((g - g.mean(axis=1, keepdims=True)) / np.sqrt(g.var(axis=1, keepdims=True) + 1e-12))
        correction = 1.4 * np.mean(auxiliary, axis=0)
        np.testing.assert_array_equal(got["b2"][c], logits[c])
        np.testing.assert_allclose(got["p2p"][c], logits[c] - 0.6 * priors[c], atol=1e-6)
        np.testing.assert_allclose(got["gda"][c], logits[c] + correction, atol=2e-6)
        np.testing.assert_allclose(got["both"][c], logits[c] + correction - 0.6 * priors[c], atol=2e-6)
    identity = fuse_heads(stages, logits, priors, models, alpha=0, prior_strength=0)
    for variant in identity.values():
        for c in COMPONENTS:
            np.testing.assert_array_equal(variant[c], logits[c])


def test_fusion_uses_only_current_query_and_its_feature_direction(train):
    stages, logits, _ = train
    priors, models = fit_heads(*train, ridge=0.4)
    full = fuse_heads(stages, logits, priors, models, 1, 1)
    single = fuse_heads({s: x[:1] * 4 for s, x in stages.items()},
                        {c: x[:1] for c, x in logits.items()}, priors, models, 1, 1)
    for variant in full:
        for c in COMPONENTS:
            np.testing.assert_allclose(single[variant][c], full[variant][c][:1], atol=2e-6)


def test_attribution_controls_use_label_counts_and_only_final_stage(train):
    stages, raw, labels = train
    priors, models = fit_heads(*train, ridge=1)
    options = {"alpha": 0.75, "prior_strength": 0.5}
    got = control_logits(stages, raw, priors, models, labels, options)
    for j, c in enumerate(COMPONENTS):
        frequency = np.bincount(labels[:, j]) / len(labels)
        np.testing.assert_allclose(got["freq"][c], raw[c] - 0.5 * np.log(frequency), atol=2e-6)
        x = stages["layer4"].astype(np.float64)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        w, b = models["layer4"][c]
        scores = x @ w + b
        standardized = (scores - scores.mean(axis=1, keepdims=True)) / np.sqrt(scores.var(axis=1, keepdims=True) + 1e-12)
        np.testing.assert_allclose(got["final_gda"][c], raw[c] + 0.75 * standardized, atol=2e-6)


def test_zero_features_and_extreme_priors_stay_finite(train):
    stages, logits, labels = train
    stages = {s: np.zeros_like(x) for s, x in stages.items()}
    for z in logits.values():
        z[:, 0] = 1000
        z[:, 1:] = -1000
    priors, models = fit_heads(stages, logits, labels, 1)
    got = fuse_heads(stages, logits, priors, models, 1, 1)
    for c in COMPONENTS:
        assert priors[c][1] == pytest.approx(-2000)
        assert np.isfinite(got["both"][c]).all()
        np.testing.assert_array_equal(got["gda"][c], logits[c])


@pytest.mark.parametrize("bad", ["nan_features", "nan_logits", "misaligned", "missing_class", "bad_label", "bad_ridge"])
def test_fit_rejects_invalid_data(train, bad):
    stages, logits, labels = train
    ridge = 1
    if bad == "nan_features":
        stages["layer4"][2, 1] = np.nan
    elif bad == "nan_logits":
        logits[COMPONENTS[0]][0, 0] = np.inf
    elif bad == "misaligned":
        stages["layer3"] = stages["layer3"][:-1]
    elif bad == "missing_class":
        labels[:, 0] = 0
    elif bad == "bad_label":
        labels[0, 0] = -1
    elif bad == "bad_ridge":
        ridge = -0.1
    with pytest.raises(ValueError):
        fit_heads(stages, logits, labels, ridge)


def test_fuse_rejects_incompatible_features_and_nonfinite_settings(train):
    stages, logits, _ = train
    priors, models = fit_heads(*train, ridge=1)
    with pytest.raises(ValueError):
        fuse_heads(stages, logits, priors, models, np.nan, 1)
    stages["layer4"] = stages["layer4"][:, :-1]
    with pytest.raises(ValueError):
        fuse_heads(stages, logits, priors, models, 1, 1)
