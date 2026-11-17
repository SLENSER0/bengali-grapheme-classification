"""Post-hoc prior correction and layer-ensemble Gaussian discriminants for B2."""

from __future__ import annotations

import numpy as np

from .utils import COMPONENTS


HeadMap = dict[str, np.ndarray]
Discriminants = dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]
_BLOCK_SIZE = 8192


def _validate_inputs(stages: HeadMap, logits: HeadMap) -> int:
    if not stages or set(logits) != set(COMPONENTS):
        raise ValueError("Expected at least one feature stage and logits for all three components")
    n = len(logits[COMPONENTS[0]])
    for name, array in (*stages.items(), *logits.items()):
        if array.ndim != 2 or array.shape[0] != n or min(array.shape) == 0:
            raise ValueError(f"{name} must be a nonempty 2D array with {n} rows")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains nonfinite values")
    return n


def _unit_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.where(norms > 0, norms, 1)


def _log_mean_softmax(logits: np.ndarray) -> np.ndarray:
    """Average the posterior over train rows, including extremely unlikely classes."""
    log_total = np.full(logits.shape[1], -np.inf)
    for start in range(0, len(logits), _BLOCK_SIZE):
        z = logits[start : start + _BLOCK_SIZE].astype(np.float64)
        z -= z.max(axis=1, keepdims=True)
        z -= np.log(np.exp(z).sum(axis=1, keepdims=True))
        peak = z.max(axis=0)
        log_block_total = peak + np.log(np.exp(z - peak).sum(axis=0))
        log_total = np.logaddexp(log_total, log_block_total)
    return (log_total - np.log(len(logits))).astype(np.float32)


def fit_heads(stages: HeadMap, logits: HeadMap, train_labels: np.ndarray, ridge: float) -> tuple[HeadMap, Discriminants]:
    """Fit from the current fold's training rows. Label columns follow COMPONENTS."""
    n = _validate_inputs(stages, logits)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if train_labels.shape != (n, len(COMPONENTS)) or not np.issubdtype(train_labels.dtype, np.integer):
        raise ValueError("Expected one integer label per component and training row")

    counts = {}
    for j, c in enumerate(COMPONENTS):
        y = train_labels[:, j]
        classes = logits[c].shape[1]
        if y.min() < 0 or y.max() >= classes:
            raise ValueError(f"{c} train labels outside the logit class range")
        counts[c] = np.bincount(y, minlength=classes)
        if (counts[c] == 0).any():
            raise ValueError(f"{c} needs at least one training sample for each prototype")

    log_priors = {c: _log_mean_softmax(logits[c]) for c in COMPONENTS}
    discriminants: Discriminants = {}
    for stage, raw in stages.items():
        dim = raw.shape[1]
        gram = np.zeros((dim, dim))
        sums = {c: np.zeros((len(counts[c]), dim)) for c in COMPONENTS}
        for start in range(0, n, _BLOCK_SIZE):
            block = slice(start, start + _BLOCK_SIZE)
            x = _unit_rows(raw[block].astype(np.float64))
            gram += x.T @ x
            for j, c in enumerate(COMPONENTS):
                np.add.at(sums[c], train_labels[block, j], x)

        stage_heads = {}
        for c in COMPONENTS:
            means = _unit_rows(sums[c])
            # Expand sum_i (x_i - mu_y)(x_i - mu_y)^T about unit class means.
            cross = sums[c].T @ means
            cov = (gram - cross - cross.T + means.T @ (counts[c][:, None] * means)) / n
            scale = np.trace(cov) / dim
            if scale <= np.finfo(np.float64).eps:
                # Identical within-class observations supply no covariance information.
                weights = means.T
            else:
                cov = (cov + cov.T) * 0.5
                cov.flat[:: dim + 1] += ridge * scale
                if ridge == 0:
                    weights = np.linalg.pinv(cov, hermitian=True) @ means.T
                else:
                    weights = np.linalg.solve(cov, means.T)
            bias = -0.5 * np.sum(means.T * weights, axis=0)
            stage_heads[c] = (weights.astype(np.float32), bias.astype(np.float32))
        discriminants[stage] = stage_heads
    return log_priors, discriminants


def fuse_heads(
    stages: HeadMap,
    logits: HeadMap,
    log_priors: HeadMap,
    discriminants: Discriminants,
    alpha: float,
    prior_strength: float,
) -> dict[str, HeadMap]:
    """Apply fitted additive corrections independently to each query."""
    _validate_inputs(stages, logits)
    if not np.isfinite([alpha, prior_strength]).all():
        raise ValueError("Fusion coefficients must be finite")
    if set(stages) != set(discriminants) or set(log_priors) != set(COMPONENTS):
        raise ValueError("Fit and inference must use matching stages and components")

    evidence = {c: np.zeros_like(logits[c], dtype=np.float32) for c in COMPONENTS}
    for stage, raw in stages.items():
        if set(discriminants[stage]) != set(COMPONENTS):
            raise ValueError(f"Missing fitted heads for {stage}")
        features = _unit_rows(raw.astype(np.float64))
        for c in COMPONENTS:
            w, b = discriminants[stage][c]
            classes = logits[c].shape[1]
            if w.shape != (raw.shape[1], classes) or b.shape != (classes,):
                raise ValueError(f"Incompatible discriminant shape for {stage}/{c}")
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError(f"Nonfinite discriminant for {stage}/{c}")
            scores = features @ w + b
            scores -= scores.mean(axis=1, keepdims=True)
            sd = np.sqrt(np.mean(np.square(scores), axis=1, keepdims=True) + 1e-12)
            evidence[c] += (scores / sd).astype(np.float32)

    result = {name: {} for name in ("b2", "p2p", "gda", "both")}
    for c in COMPONENTS:
        prior = log_priors[c]
        if prior.shape != (logits[c].shape[1],) or not np.isfinite(prior).all():
            raise ValueError(f"Invalid log prior for {c}")
        base = np.asarray(logits[c], dtype=np.float32)
        correction = prior_strength * prior
        gda = base + (alpha / len(stages)) * evidence[c]
        result["b2"][c] = base
        result["gda"][c] = gda
        result["p2p"][c] = base - correction
        result["both"][c] = gda - correction
    return result
