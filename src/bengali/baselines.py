"""B0: constant (majority-class) predictor — the floor and a sanity check of the metric.

For a constant prediction the macro recall of a component with K classes is
exactly 1/K, so the expected score is (2/168 + 1/11 + 1/7) / 4 ≈ 0.0614.

Usage: python -m bengali.baselines --config configs/constant.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from .runs import load_fold_data, prediction_frame, save_run
from .utils import COMPONENTS, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    for fold in cfg.get("folds", [0]):
        _, folds, labels, ix = load_fold_data(cfg, fold, in_memory=False)
        majority = {c: int(np.bincount(labels[ix["train"], j]).argmax()) for j, c in enumerate(COMPONENTS)}
        preds = {p: {c: np.full(len(ix[p]), majority[c]) for c in COMPONENTS} for p in ("val_id", "val_ood")}
        confs = {p: {c: np.ones(len(ix[p])) for c in COMPONENTS} for p in ("val_id", "val_ood")}
        m = save_run(cfg, fold, 0, prediction_frame(folds, ix, preds, confs), {"majority": majority})
        print(f"[constant fold {fold}]", {k: round(v, 4) for k, v in m.items() if k.endswith("score")})


if __name__ == "__main__":
    main()
