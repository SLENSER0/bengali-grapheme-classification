"""Compare the prespecified HW2 controls and stratify OOD root recall by training context."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import fold_context
from .hw2_evaluate import BOOT_SEED, CELLS, FOLDS, GRID, SCORES, SEEDS, SHARED_METADATA, expected_rows, load_run
from .metric import paired_bootstrap
from .splits import load_folds
from .utils import resolve, save_json

CONTROL_PAIRS = (("p2p", "freq"), ("gda", "final_gda"))
CONTROL_HYPOTHESES = {
    "freq": "Достаточно ли наблюдаемых частот меток вместо среднего предсказанного prior?",
    "final_gda": "Достаточно ли признаков последнего слоя вместо двух слоёв?",
}
BINS = ("1–2", "3–5", "6–10", ">10")
NAMES = {**CELLS, "freq": "Частоты меток", "final_gda": "GDA по слою 4"}


def read_artifacts(results: Path, folds: pd.DataFrame):
    frames = {cell: [] for cell in NAMES}
    scores = {cell: [] for cell in NAMES}
    if not folds.image_id.is_unique:
        raise ValueError("Fold assignments must have unique image IDs")
    protocol = None
    folds_sha256 = None
    tracked = []
    for fold, seed in GRID:
        expected = expected_rows(folds, fold)
        if expected["slice"].nunique() != 2:
            raise ValueError(f"Fold {fold} must have both ID and OOD rows")
        cells = NAMES if seed == SEEDS[0] else ("b2", *CONTROL_HYPOTHESES)
        for cell in cells:
            parent = results if cell in CELLS else results / "controls"
            path = parent / cell / f"fold{fold}_seed{seed}"
            frame, metrics, run_protocol = load_run(path, expected, fold, seed)
            identity = {k: metrics.get(k) for k in SHARED_METADATA}
            if cell == "b2":
                reference = identity
            elif identity != reference:
                raise ValueError(f"{path}: metadata differ from B2")
            if protocol is None:
                protocol = run_protocol
                folds_sha256 = metrics["folds_sha256"]
            elif run_protocol != protocol:
                raise ValueError(f"{path}: protocol differs from B2")
            if metrics["folds_sha256"] != folds_sha256:
                raise ValueError(f"{path}: fold assignment checksum differs across runs")
            if cell in CONTROL_HYPOTHESES:
                tracked.append({"cell": cell, "method": NAMES[cell], "hypothesis": CONTROL_HYPOTHESES[cell],
                                "config_path": str((path / "config.yaml").relative_to(results / "controls")),
                                **metrics})
            if seed == SEEDS[0]:
                frames[cell].append(frame)
                scores[cell].append({k: float(metrics[f"{k}_score"]) for k in SCORES})
    return {cell: pd.concat(parts, ignore_index=True) for cell, parts in frames.items()}, scores, tracked


def root_contexts(frames: dict, folds: pd.DataFrame) -> dict:
    """Pool per-fold, per-root recalls; each observed (fold, root) receives equal weight."""
    contexts = fold_context(folds)
    pairs = []
    for fold, rows in frames["b2"].query("slice == 'ood'").groupby("fold", sort=True):
        roots = rows["grapheme_root"]
        summary = roots.value_counts(sort=False).rename("n_images").to_frame()
        summary["n_contexts"] = summary.index.map(contexts[fold]["graphemes_per_root"])
        if summary["n_contexts"].isna().any():
            raise ValueError(f"An OOD root has no training context in fold {fold}")
        for cell in CELLS:
            # load_run joins to the same expected row order for every cell and fold.
            correct = frames[cell].loc[rows.index, "pred_grapheme_root"].eq(roots)
            summary[cell] = correct.groupby(roots).mean()
        pairs.append(summary)
    data = pd.concat(pairs, ignore_index=True)
    grouped = pd.cut(data["n_contexts"], [0, 2, 5, 10, np.inf], labels=BINS)
    result = {}
    for label in BINS:
        group = data.loc[grouped == label]
        result[label] = {
            "fold_root_pairs": len(group), "ood_images": int(group.n_images.sum()),
            "recall": {cell: float(group[cell].mean()) if len(group) else None for cell in CELLS},
        }
    return result


def markdown(summary: dict) -> str:
    lines = [
        "# ДЗ2. Контрольные сравнения и контекст корня", "",
        "Дополнительные сравнения используют те же пять фолдов и чекпойнтов B2 при seed 42. "
        "Выбор модели опирается на основной эксперимент. Здесь P2P сравнивается с поправкой по частоте "
        "обучающих меток, а GDA по слоям 3 и 4 — с GDA только по слою 4. "
        "Коэффициенты поправок зафиксированы до измерения. Реестр охватывает также два дополнительных seed на фолде 0. "
        "[Все 14 контрольных запусков](controls/experiments.csv)", "",
        "| Метод | main | ID | OOD |", "|---|---:|---:|---:|",
    ]
    for cell, scores in summary["mean_scores"].items():
        lines.append("| " + " | ".join([NAMES[cell], *(f"{scores[k]:.4f}" for k in SCORES)]) + " |")
    lines += [
        "", "В следующей таблице положительная разность означает преимущество первого метода. "
        "Все разности и границы интервалов даны в процентных пунктах.", "",
        "| Сравнение | Метрика | Средняя Δ [95% ДИ] | Фолд 0 | Фолд 1 | Фолд 2 | Фолд 3 | Фолд 4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, control in CONTROL_PAIRS:
        effect = summary["comparisons"][f"{candidate}-{control}"]
        for k in SCORES:
            ci = effect["bootstrap"][k]
            interval = f"{100 * ci['point']:+.3f} [{100 * ci['lo']:+.3f}, {100 * ci['hi']:+.3f}]"
            deltas = [f"{100 * delta:+.3f}" for delta in effect["per_fold"][k]]
            lines.append("| " + " | ".join([f"{NAMES[candidate]} − {NAMES[control]}", k, interval, *deltas]) + " |")
    lines += [
        "", f"Парный бутстрап использует {summary['bootstrap']['n_resamples']} повторов. "
        "В каждом повторе для обоих методов выбираются одни и те же изображения с возвращением внутри "
        "каждой графемы, среза и фолда. 95% процентильный интервал оценивает разброс среднего по пяти "
        "фиксированным фолдам при фиксированных графемах и чекпойнтах.", "",
        "## OOD-recall корня по числу обучающих контекстов", "",
        "Контекст — отдельная графема обучающей части фолда с данным корнем. "
        "Сначала считаем recall каждого корня по его OOD-изображениям в одном фолде. Затем внутри диапазона "
        "усредняем эти значения с одинаковым весом для каждой пары «фолд, корень», как в ДЗ1. "
        "Число изображений показывает поддержку группы. В этих данных частота корня и число его контекстов связаны.", "",
        "| Графем в train | Пар «фолд, корень» | OOD-изображений | B2 | P2P | GDA | P2P + GDA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in summary["ood_root_by_context"].items():
        recalls = ["—" if row["recall"][c] is None else f"{row['recall'][c]:.4f}" for c in CELLS]
        lines.append("| " + " | ".join([label, str(row["fold_root_pairs"]), str(row["ood_images"]), *recalls]) + " |")
    return "\n".join(lines) + "\n"


def evaluate(results: Path, folds: pd.DataFrame, n_boot: int = 1000) -> dict:
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")
    frames, scores, tracked = read_artifacts(results, folds)
    means = {c: {k: float(np.mean([row[k] for row in scores[c]])) for k in SCORES} for c in NAMES}
    comparisons = {}
    for candidate, control in CONTROL_PAIRS:
        ci = paired_bootstrap(frames[candidate], frames[control], n_boot=n_boot, seed=BOOT_SEED)
        per_fold = {k: [a[k] - b[k] for a, b in zip(scores[candidate], scores[control], strict=True)] for k in SCORES}
        comparisons[f"{candidate}-{control}"] = {"bootstrap": ci, "per_fold": per_fold}
    summary = {"seed": SEEDS[0], "fold_order": list(FOLDS), "bootstrap": {"seed": BOOT_SEED, "n_resamples": n_boot},
               "mean_scores": means, "comparisons": comparisons, "ood_root_by_context": root_contexts(frames, folds)}
    pd.json_normalize(tracked).to_csv(results / "controls" / "experiments.csv", index=False)
    save_json(summary, results / "diagnostics.json")
    (results / "diagnostics.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/hw2")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--folds-file", default="splits/folds.csv")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    args = parser.parse_args()
    results = resolve(args.results)
    evaluate(results, load_folds(args.folds_file, args.train_csv), args.n_boot)
    print(f"Saved HW2 diagnostics in {results}")


if __name__ == "__main__":
    main()
