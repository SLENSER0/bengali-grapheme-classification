"""Matched controls for attributing a gain to learned priors or intermediate features."""

import numpy as np

from .hw2_heads import fuse_heads
from .utils import COMPONENTS


def control_logits(features, raw, log_priors, discriminants, train_labels, settings):
    freq = {}
    for j, component in enumerate(COMPONENTS):
        counts = np.bincount(train_labels[:, j], minlength=raw[component].shape[1])
        if (counts == 0).any():
            raise ValueError(f"Every {component} class must be present in training")
        log_frequency = np.log(counts / len(train_labels)).astype(np.float32)
        freq[component] = raw[component] - settings["prior_strength"] * log_frequency
    stage = "layer4"
    final_only = fuse_heads(
        {stage: features[stage]}, raw, log_priors, {stage: discriminants[stage]},
        alpha=settings["alpha"], prior_strength=0,
    )["gda"]
    return {"freq": freq, "final_gda": final_only}
