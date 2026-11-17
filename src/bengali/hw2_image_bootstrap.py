"""Sensitivity of the seed-42 P2P + GDA contrast to shared physical images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .hw2_evaluate import FOLDS, PREDICTIONS, SHARED_METADATA, expected_rows, load_run
from .splits import load_folds
from .utils import COMPONENTS, WEIGHTS, resolve, save_json


def image_bootstrap(base: pd.DataFrame, both: pd.DataFrame, n_boot: int = 1000, seed: int = 2026) -> dict:
    if n_boot < 2:
        raise ValueError("At least two multiplier draws are required")
    key = ["image_id", "fold", "slice"]
    truth = ["grapheme", *COMPONENTS]
    if base.empty or len(base) != len(both) or any(frame.duplicated(key).any() for frame in (base, both)):
        raise ValueError("Expected one paired prediction per fold/slice/image observation")
    rows = base[key + truth + PREDICTIONS].merge(
        both[key + truth + PREDICTIONS], on=key + truth, suffixes=("_b2", "_both"), validate="one_to_one",
    )
    if len(rows) != len(base) or rows[key + truth].isna().any().any():
        raise ValueError("Predictions disagree on evaluation rows or true labels")
    strata = rows.groupby(["fold", "slice", "grapheme"], sort=False)
    if (strata[COMPONENTS].nunique() != 1).any().any():
        raise ValueError("Every fold/slice/grapheme stratum must have one true component tuple")
    stratum = strata.ngroup().to_numpy(dtype=np.int64)
    stratum_size = np.bincount(stratum)
    image, image_names = pd.factorize(rows.image_id, sort=True)
    part_id = rows["slice"].map({"id": 0, "ood": 1})
    if part_id.isna().any():
        raise ValueError("Unrecognized evaluation slice")
    part_id = part_id.to_numpy(dtype=np.int64)
    groups = rows.groupby(["fold", "slice"]).indices
    n_folds = rows.fold.nunique()
    if len(groups) != 2 * n_folds:
        raise ValueError("Both evaluation slices must be present in each fold")

    # After within-stratum normalization, every class keeps its original mass.
    # A corrected classification then contributes 1/(class support × present classes)
    # to that fold's component macro-recall, and a newly wrong one contributes its negative.
    delta = np.zeros(len(rows))
    for c in COMPONENTS:
        label = rows[c].to_numpy(dtype=np.int64)
        column = f"pred_{c}"
        fixed = (rows[column + "_both"].to_numpy() == label).astype(float)
        fixed -= rows[column + "_b2"].to_numpy() == label
        component_weight = WEIGHTS[c] / (sum(WEIGHTS.values()) * n_folds)
        for pos in groups.values():
            classes = label[pos]
            support = np.bincount(classes)
            delta[pos] += component_weight * fixed[pos] / (support[classes] * np.count_nonzero(support))

    def endpoints(row_weight):
        id_score, ood_score = np.bincount(part_id, weights=delta * row_weight, minlength=2)
        return id_score, ood_score, (id_score + ood_score) / 2, ood_score - id_score

    point = endpoints(1.0)
    rng = np.random.default_rng(seed)
    replicates = np.empty((n_boot, len(point)))
    for r in range(n_boot):
        raw = rng.exponential(size=len(image_names))[image]
        mass = np.bincount(stratum, weights=raw)
        row_weight = raw * (stratum_size / mass)[stratum]
        replicates[r] = endpoints(row_weight)
    lo, hi = np.quantile(replicates, [0.025, 0.975], axis=0)
    return {
        "comparison": "both-b2",
        "checkpoint_seed": 42,
        "folds": sorted(int(fold) for fold in rows.fold.unique()),
        "n_boot": n_boot, "rng_seed": seed,
        "multiplier_unit": "image_id",
        "strata": ["fold", "slice", "grapheme"],
        "rows": len(rows), "unique_images": len(image_names), "reused_images": int((np.bincount(image) > 1).sum()),
        "contrast": {name: {"point": float(point[i]), "lo": float(lo[i]), "hi": float(hi[i])}
                     for i, name in enumerate(("id", "ood", "main", "ood_minus_id"))},
    }


def evaluate(results: Path, folds: pd.DataFrame, n_boot: int, seed: int) -> dict:
    frames = {"b2": [], "both": []}
    shared_protocol = None
    for fold in FOLDS:
        expected = expected_rows(folds, fold)
        reference = None
        for cell in frames:
            path = results / cell / f"fold{fold}_seed42"
            frame, metrics, protocol = load_run(path, expected, fold, 42)
            if shared_protocol is None:
                shared_protocol = protocol
            elif shared_protocol != protocol:
                raise ValueError(f"{path}: configuration mismatch across runs")
            metadata = {k: metrics[k] for k in SHARED_METADATA}
            if reference is None:
                reference = metadata
            elif metadata != reference:
                raise ValueError(f"{path}: mismatch with B2 checkpoint or archive")
            frames[cell].append(frame)
    result = image_bootstrap(pd.concat(frames["b2"]), pd.concat(frames["both"]), n_boot, seed)
    save_json(result, results / "image_bootstrap.json")
    description = [
        "# Чувствительность интервалов к повторному использованию изображений", "",
        f"Для контраста P2P + GDA минус B2 использованы {result['rows']} наблюдений "
        f"и {result['unique_images']} уникальных изображений. В нескольких наблюдениях встречаются "
        f"{result['reused_images']} изображений.", "",
        f"В каждой из {n_boot} реплик байесовского бутстрапа одно случайное значение Exp(1) "
        "назначается каждому image_id и используется для всех его появлений. Веса нормируются "
        "до исходной численности отдельно внутри каждой графемы, фолда и среза. "
        "95%-е интервалы построены по 2.5-му и 97.5-му процентилям. Они условны на наблюдаемых "
        "графемах, разбиении и пяти чекпойнтах с seed 42. Зависимость между результатами "
        "разных фолдов из-за общих изображений учитывается через общий случайный вес. "
        f"Поправка на множественные сравнения не применялась. Seed генератора равен {seed}.", "",
        "| Метрика | Разность recall | Номинальный 95% ДИ |", "|---|---:|---:|",
    ]
    for name, label in [("id", "ΔID"), ("ood", "ΔOOD"), ("main", "Δmain"), ("ood_minus_id", "ΔOOD − ΔID")]:
        est = result["contrast"][name]
        description.append(f"| {label} | {est['point']:+.5f} | [{est['lo']:+.5f}, {est['hi']:+.5f}] |")
    (results / "image_bootstrap.md").write_text("\n".join(description) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/hw2")
    parser.add_argument("--folds-file", default="splits/folds.csv")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    evaluate(resolve(args.results), load_folds(args.folds_file, args.train_csv), args.n_boot, args.seed)


if __name__ == "__main__":
    main()
