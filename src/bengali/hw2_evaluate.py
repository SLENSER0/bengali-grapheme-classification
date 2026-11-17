"""Validate and compare the four cells of the HW2 frozen-checkpoint experiment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

from .evaluate import fold_context, fold_mean_scores, leak_masks
from .metric import _fold_bootstrap, _summarise, slice_scores
from .splits import fold_indices, load_folds
from .utils import COMPONENTS, N_CLASSES, SHORT, resolve, save_json

matplotlib.use("Agg")
from matplotlib import pyplot as plt


CELLS = {"b2": "B2", "p2p": "P2P", "gda": "GDA", "both": "P2P + GDA"}
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (42, 43, 44)
GRID = [(fold, SEEDS[0]) for fold in FOLDS] + [(0, seed) for seed in SEEDS[1:]]
SCORES = ("main", "id", "ood")
METRICS = (*SCORES, "ood_minus_id")
KEY = ["image_id", "slice"]
PREDICTIONS = [f"pred_{c}" for c in COMPONENTS]
CHECKSUMS = ("checkpoint_sha256", "folds_sha256", "archive_predictions_sha256")
SHARED_METADATA = (*CHECKSUMS, "archive_parity", "device", "versions")
HW2_PARAMETERS = {"stages", "ridge", "alpha", "prior_strength"}
CONTRASTS = {
    "p2p-b2": {"p2p": 1, "b2": -1},
    "gda-b2": {"gda": 1, "b2": -1},
    "both-b2": {"both": 1, "b2": -1},
    "both-p2p": {"both": 1, "p2p": -1},
    "both-gda": {"both": 1, "gda": -1},
    "interaction": {"both": 1, "p2p": -1, "gda": -1, "b2": 1},
}
BOOT_SEED = 1


def expected_rows(folds: pd.DataFrame, fold: int) -> pd.DataFrame:
    indices = fold_indices(folds, fold)
    return pd.concat(
        [folds.iloc[indices[f"val_{part}"]][["image_id", "grapheme", *COMPONENTS]].assign(slice=part)
         for part in ("id", "ood")],
        ignore_index=True,
    )


def load_run(path: Path, expected: pd.DataFrame, fold: int, seed: int) -> tuple[pd.DataFrame, dict, dict]:
    """Check artifacts produced by save_run and restore official labels in fold order."""
    for name in ("preds.csv.gz", "metrics.json", "config.yaml"):
        if not (path / name).is_file():
            raise ValueError(f"{path}: missing {name}")
    metrics = json.loads((path / "metrics.json").read_text())
    config = yaml.safe_load((path / "config.yaml").read_text())
    if not isinstance(metrics, dict) or not isinstance(config, dict):
        raise ValueError(f"{path}: expected mappings in metrics.json and config.yaml")
    experiment = f"hw2_{path.parent.name}"
    if config.get("experiment") != experiment or metrics.get("experiment") != experiment:
        raise ValueError(f"{path}: expected experiment {experiment} in config and metrics")
    if (metrics.get("fold"), metrics.get("seed")) != (fold, seed) or config.get("run") != {"fold": fold, "seed": seed}:
        raise ValueError(f"{path}: fold/seed metadata disagree with the run directory")
    settings = config.get("hw2")
    if not isinstance(settings, dict) or not HW2_PARAMETERS.issubset(settings):
        raise ValueError(f"{path}: effective HW2 parameters are missing")
    for key in CHECKSUMS:
        checksum = metrics.get(key)
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"{path}: missing or invalid {key}")
    parity = metrics.get("archive_parity")
    if not isinstance(parity, dict) or parity.get("all_classes_match") is not True or parity.get("rows") != len(expected):
        raise ValueError(f"{path}: archive parity was not established for the whole validation set")

    predictions = pd.read_csv(path / "preds.csv.gz")
    missing = set(KEY + PREDICTIONS) - set(predictions.columns)
    if missing:
        raise ValueError(f"{path}: missing prediction columns {sorted(missing)}")
    if predictions.duplicated(KEY).any():
        raise ValueError(f"{path}: duplicate validation row")
    actual_keys = pd.MultiIndex.from_frame(predictions[KEY])
    expected_keys = pd.MultiIndex.from_frame(expected[KEY])
    if len(actual_keys) != len(expected_keys) or not actual_keys.isin(expected_keys).all():
        raise ValueError(f"{path}: prediction rows differ from the fold's expected ID/OOD rows")
    for component, column in zip(COMPONENTS, PREDICTIONS, strict=True):
        values = predictions[column]
        if not pd.api.types.is_integer_dtype(values) or not values.between(0, N_CLASSES[component] - 1).all():
            raise ValueError(f"{path}: invalid class index in {column}")

    frame = expected.merge(predictions[KEY + PREDICTIONS], on=KEY, validate="one_to_one", sort=False)
    scores = slice_scores(frame)
    for key, value in scores.items():
        recorded = metrics.get(key)
        if not isinstance(recorded, (int, float)) or not np.isclose(recorded, value, atol=1e-12, rtol=0):
            raise ValueError(f"{path}: recorded {key}={recorded!r} differs from recomputed {value}")
    frame["fold"] = fold
    protocol = {k: v for k, v in config.items() if k not in ("run", "experiment", "output_dir")}
    return frame, {**metrics, **scores}, protocol


def load_grid(results: Path, folds: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    if not folds["image_id"].is_unique:
        raise ValueError("Fold assignments must have unique image IDs")
    expected = {fold: expected_rows(folds, fold) for fold in FOLDS}
    if any(rows["slice"].nunique() != 2 for rows in expected.values()):
        raise ValueError("Every fold must contain both validation slices")
    runs, primary = [], {}
    shared = {}
    registered_protocol, folds_checksum = None, None
    for cell in CELLS:
        frames = []
        for fold, seed in GRID:
            path = results / cell / f"fold{fold}_seed{seed}"
            frame, metrics, protocol = load_run(path, expected[fold], fold, seed)
            if registered_protocol is None:
                registered_protocol = protocol
                folds_checksum = metrics["folds_sha256"]
            elif protocol != registered_protocol:
                raise ValueError(f"{path}: configuration differs from the other runs")
            if metrics["folds_sha256"] != folds_checksum:
                raise ValueError(f"{path}: fold assignments checksum differs from the other runs")
            metadata = {key: metrics[key] for key in SHARED_METADATA if key in metrics}
            if cell == "b2":
                shared[fold, seed] = metadata
            elif metadata != shared[fold, seed]:
                raise ValueError(f"{path}: checkpoint, archive, or runtime metadata differ from B2")
            runs.append({**metrics, "cell": cell})
            if seed == SEEDS[0]:
                frames.append(frame)
        primary[cell] = pd.concat(frames, ignore_index=True)
    return pd.DataFrame(runs), primary, registered_protocol


def paired_results(runs: pd.DataFrame, frames: dict[str, pd.DataFrame], n_boot: int) -> tuple[dict, dict]:
    points, reps = _fold_bootstrap([frames[cell] for cell in CELLS], n_boot, BOOT_SEED)
    cell_points = dict(zip(CELLS, points, strict=True))
    cell_reps = dict(zip(CELLS, reps, strict=True))
    primary = runs[runs["seed"] == SEEDS[0]].set_index(["cell", "fold"])
    fold_metrics = {cell: primary.loc[cell].sort_index() for cell in CELLS}
    table = {}
    for cell, frame in fold_metrics.items():
        scores = {
            column: {"mean": float(frame[column].mean()), "std": float(frame[column].std(ddof=1))}
            for column in slice_scores(frames[cell])
            if not column.endswith("_n")
        }
        ci = {k: _summarise(cell_points[cell][k], cell_reps[cell][k], 0.05) for k in SCORES}
        table[cell] = {"scores": scores, "ci": ci}

    for value in (*points, *reps):
        value["ood_minus_id"] = value["ood"] - value["id"]
    paired = {}
    for name, weights in CONTRASTS.items():
        ci = {
            k: _summarise(
                sum(weight * cell_points[cell][k] for cell, weight in weights.items()),
                sum(weight * cell_reps[cell][k] for cell, weight in weights.items()),
                0.05,
            )
            for k in METRICS
        }
        deltas = {
            k: sum(weight * fold_metrics[cell][f"{k}_score"].to_numpy() for cell, weight in weights.items())
            for k in SCORES
        }
        deltas["ood_minus_id"] = deltas["ood"] - deltas["id"]
        paired[name] = {
            "bootstrap": ci,
            "per_fold": {k: values.tolist() for k, values in deltas.items()},
            "positive_main_folds": int((deltas["main"] > 0).sum()),
            "std_main_folds": float(deltas["main"].std(ddof=1)),
        }
    return table, paired


def seed_results(runs: pd.DataFrame, paired: dict) -> tuple[dict, dict]:
    fold_zero = runs[runs["fold"] == 0].set_index(["cell", "seed"]).sort_index()
    columns = [f"{k}_score" for k in SCORES]
    baseline = fold_zero.loc["b2", columns]
    threshold = 2 * float(baseline["main_score"].std(ddof=1))
    seeds, decisions = {}, {}
    for cell in CELLS:
        scores = fold_zero.loc[cell, columns]
        deltas = scores - baseline
        seeds[cell] = {
            "std_main": float(scores["main_score"].std(ddof=1)),
            "scores": scores.reset_index().to_dict("records"),
            "deltas": deltas.reset_index().to_dict("records"),
        }
        if cell == "b2":
            continue
        result = paired[f"{cell}-b2"]
        ci = result["bootstrap"]
        checks = {
            "positive_at_least_four_folds": result["positive_main_folds"] >= 4,
            "main_ci_above_zero": ci["main"]["lo"] > 0,
            "main_above_seed_threshold": ci["main"]["point"] > threshold,
            "ood_noninferior": ci["ood"]["lo"] > -0.005,
        }
        decisions[cell] = {"checks": checks, "accepted": all(checks.values())}
    return seeds, {"main_seed_threshold": threshold, "ood_margin": -0.005, "candidates": decisions}


def diagnostics(frames: dict[str, pd.DataFrame], folds: pd.DataFrame, near_dup: pd.DataFrame) -> tuple[dict, dict]:
    context = fold_context(folds)
    base = frames["b2"]
    near_duplicate, seen_truth = leak_masks(base, context, near_dup)
    truth = base[COMPONENTS].to_numpy()
    base_wrong = truth != base[PREDICTIONS].to_numpy()
    id_rows = (base["slice"] == "id").to_numpy()
    ood_rows = ~id_rows
    slice_rows = {"id": id_rows, "ood": ood_rows, "ood_unseen_truth": ood_rows & ~seen_truth}
    all_stats = {}
    sensitivity = {
        "n_id_near_duplicate": int(near_duplicate.sum()),
        "n_ood_seen_truth": int(seen_truth.sum()),
        "scores": {},
    }
    for cell, frame in frames.items():
        wrong = truth != frame[PREDICTIONS].to_numpy()
        seen_prediction = np.zeros(len(frame), dtype=bool)
        for fold, part in frame.loc[ood_rows].groupby("fold"):
            seen = context[fold]["seen"]
            seen_prediction[part.index] = [tuple(row) in seen for row in part[PREDICTIONS].to_numpy()]
        stats = {}
        for name, mask in slice_rows.items():
            current = np.column_stack((wrong[mask], wrong[mask].any(axis=1)))
            before = np.column_stack((base_wrong[mask], base_wrong[mask].any(axis=1)))
            n = int(mask.sum())
            components = {}
            for i, label in enumerate([*(SHORT[c] for c in COMPONENTS), "triple"]):
                error_n = int(current[:, i].sum())
                components[label] = {
                    "errors": error_n,
                    "error_rate": error_n / n if n else None,
                    "repaired": int((before[:, i] & ~current[:, i]).sum()),
                    "broken": int((~before[:, i] & current[:, i]).sum()),
                }
            result = {"n": n, "errors": components}
            if name != "id":
                wrong_n = components["triple"]["errors"]
                wrong_seen_n = int((current[:, -1] & seen_prediction[mask]).sum())
                result.update(
                    seen_errors=wrong_seen_n,
                    conditional_seen_error_rate=wrong_seen_n / wrong_n if wrong_n else None,
                    unconditional_seen_error_rate=wrong_seen_n / n if n else None,
                )
            stats[name] = result
        all_stats[cell] = stats
        sensitivity["scores"][cell] = {
            "full": fold_mean_scores(frame),
            "no_near_duplicate": fold_mean_scores(frame, near_duplicate),
            "no_seen_truth": fold_mean_scores(frame, seen_truth),
            "no_both": fold_mean_scores(frame, near_duplicate | seen_truth),
        }
    return all_stats, sensitivity


def markdown(summary: dict) -> str:
    def interval(ci: dict) -> str:
        return f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}]"

    def percent(value: float | None) -> str:
        return "—" if value is None else f"{value:.2%}"

    lines = [
        "# ДЗ2. Результаты эксперимента",
        "",
        "Каждая ячейка проверена на пяти фолдах с seed 42 и на фолде 0 с seed 43 и 44. "
        "Во всех ячейках совпадают изображения и срезы каждого запуска. "
        "Во всех семи запусках предсказанные классы B2 полностью совпали с архивом ДЗ1. "
        "В таблице среднее ± выборочное стандартное отклонение по пяти фолдам с seed 42.",
        "",
        "| Ячейка | main | ID | OOD | ID root / vowel / consonant | OOD root / vowel / consonant |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell, name in CELLS.items():
        scores = summary["table"][cell]["scores"]
        weighted = [f"{scores[f'{k}_score']['mean']:.4f} ± {scores[f'{k}_score']['std']:.4f}" for k in SCORES]
        per_head = [" / ".join(f"{scores[f'{part}_{SHORT[c]}']['mean']:.4f}" for c in COMPONENTS) for part in ("id", "ood")]
        lines.append("| " + " | ".join([name, *weighted, *per_head]) + " |")
    lines += [
        "",
        "## Парные эффекты",
        "",
        "Эффекты измерены в единицах recall. Взаимодействие равно `(P2P + GDA) − P2P − GDA + B2`. "
        "Положительное `ΔOOD − ΔID` означает больший выигрыш на невиданных графемах.",
        "",
        "| Контраст | Δmain [95% ДИ] | ΔID [95% ДИ] | ΔOOD [95% ДИ] | ΔOOD − ΔID [95% ДИ] | Фолдов с Δmain > 0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in summary["paired"].items():
        confidence = result["bootstrap"]
        columns = [f"{confidence[k]['point']:+.4f} {interval(confidence[k])}" for k in METRICS]
        lines.append("| " + " | ".join([name, *columns, f"{result['positive_main_folds']}/{len(FOLDS)}"]) + " |")
    lines += [
        "",
        f"Для 95% ДИ использован парный процентильный бутстрап ({summary['bootstrap']['n_resamples']} реплик). "
        "Изображения пересэмплируются с возвращением внутри графемы, фолда и среза. "
        "Каждая реплика использует одни и те же строки для четырёх ячеек и заново вычисляет среднее по фолдам. "
        "Интервалы условны на данных графемах и чекпойнтах. ID авторов неизвестны. "
        "Одно изображение может участвовать в разных фолдах, а обучающие выборки пересекаются.",
        "",
        "![Парные эффекты, фолды и recall корня](comparison.png)",
        "",
        "## Критерий выбора из ДЗ1",
        "",
        f"Порог по обучающему seed равен {summary['selection']['main_seed_threshold']:.6f} "
        "(два стандартных отклонения B2 на фолде 0). Остальные условия требуют положительного Δmain на ≥4 фолдах, "
        "положительной нижней границы 95% ДИ Δmain и нижней границы 95% ДИ ΔOOD выше −0.005.",
        "",
        "| Ячейка | ≥4 фолда | ДИ Δmain > 0 | Δmain > порога seed | ДИ ΔOOD выше −0.005 | Проходит |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell, result in summary["selection"]["candidates"].items():
        checks = ["да" if passed else "нет" for passed in [*result["checks"].values(), result["accepted"]]]
        lines.append("| " + " | ".join([CELLS[cell], *checks]) + " |")
    lines += [
        "",
        "## Чувствительность к обучающему seed",
        "",
        "Сравнения с B2 на фолде 0 используют одинаковый чекпойнт в каждой паре.",
        "",
        "| Ячейка | Seed | main | ID | OOD | Δmain | ΔID | ΔOOD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell, result in summary["seeds"].items():
        for scores, deltas in zip(result["scores"], result["deltas"], strict=True):
            lines.append("| " + " | ".join([
                CELLS[cell], str(scores["seed"]), *[f"{scores[f'{k}_score']:.4f}" for k in SCORES],
                *[f"{deltas[f'{k}_score']:+.4f}" for k in SCORES],
            ]) + " |")
    lines += [
        "",
        "## Ошибки и знакомые тройки на OOD",
        "",
        "Ошибки ниже посчитаны по изображениям. В скобках число исправленных и новых ошибок относительно B2. "
        "Тройка считается знакомой, если она встречалась в train того же фолда.",
        "",
        "| Ячейка | Срез | N | Корень | Гласный | Согласный | Тройка |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell, parts in summary["diagnostics"].items():
        for part in ("id", "ood"):
            result = parts[part]
            counts = [f"{percent(v['error_rate'])} ({v['repaired']}/{v['broken']})" for v in result["errors"].values()]
            lines.append("| " + " | ".join([CELLS[cell], part.upper(), str(result["n"]), *counts]) + " |")
    lines += [
        "",
        "| Ячейка | OOD-срез | N | Ошибки тройки | Ошибки со знакомой тройкой | Доля среди ошибок | Доля среди всех |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell, parts in summary["diagnostics"].items():
        for part, label in [("ood", "все"), ("ood_unseen_truth", "новая истинная тройка")]:
            row = parts[part]
            lines.append(f"| {CELLS[cell]} | {label} | {row['n']} | {row['errors']['triple']['errors']} | {row['seen_errors']} | "
                         f"{percent(row['conditional_seen_error_rate'])} | {percent(row['unconditional_seen_error_rate'])} |")
    sensitivity = summary["leak_sensitivity"]
    lines += [
        "",
        "## Чувствительность к утечкам из ДЗ1",
        "",
        f"Повторная оценка исключает {sensitivity['n_id_near_duplicate']} ID-изображений с известной копией в train "
        f"и {sensitivity['n_ood_seen_truth']} OOD-изображений, чья истинная тройка встречалась в train. "
        "Исключения одинаковы для четырёх ячеек. Чекпойнты обучены на исходных фолдах.",
        "",
        "| Ячейка | ID без копий | OOD без знакомых истинных троек | main без обоих | Δmain к B2 без обоих |",
        "|---|---:|---:|---:|---:|",
    ]
    clean_base = sensitivity["scores"]["b2"]["no_both"]["main"]
    for cell, row in sensitivity["scores"].items():
        clean_main = row["no_both"]["main"]
        lines.append(f"| {CELLS[cell]} | {row['no_near_duplicate']['id']:.4f} | {row['no_seen_truth']['ood']:.4f} | "
                     f"{clean_main:.4f} | {clean_main - clean_base:+.4f} |")
    return "\n".join(lines) + "\n"


def plot_comparison(summary: dict, path: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2), layout="constrained")
    colors = {"p2p": "#1d6f98", "gda": "#a1531f", "both": "#59633b"}
    for index, (cell, color) in enumerate(colors.items()):
        paired = summary["paired"][f"{cell}-b2"]
        for part, marker, offset in [("id", "o", -0.12), ("ood", "D", 0.12)]:
            effect = paired["bootstrap"][part]
            ordinate = index + offset
            axs[0].hlines(ordinate, 100 * effect["lo"], 100 * effect["hi"], color=color)
            axs[0].scatter(100 * effect["point"], ordinate, color=color, marker=marker, label=part.upper() if index == 0 else None)
        axs[1].plot(FOLDS, 100 * np.asarray(paired["per_fold"]["main"]), "o-", color=color, label=CELLS[cell])

    for index, cell in enumerate(CELLS):
        scores = summary["table"][cell]["scores"]
        left, right = [100 * scores[f"{part}_root"]["mean"] for part in ("id", "ood")]
        axs[2].plot([left, right], [index, index], color="#a6a6a6", linewidth=1.2, zorder=1)
        axs[2].scatter(left, index, marker="o", color="#335a6c", zorder=2)
        axs[2].scatter(right, index, marker="D", color="#335a6c", zorder=2)

    axs[0].set(title="ID и OOD против B2", xlabel="Δ recall, п.п.")
    axs[0].set_yticks(range(len(colors)), [CELLS[cell] for cell in colors])
    axs[0].invert_yaxis()
    axs[0].axvline(0, color="#555555", linewidth=0.8)
    axs[0].legend(loc="best")
    axs[1].set(title="Выигрыш main по фолдам", xlabel="Фолд (seed 42)", ylabel="Δ main, п.п.", xticks=FOLDS)
    axs[1].axhline(0, color="#555555", linewidth=0.8)
    axs[1].legend(loc="best")
    axs[2].set(title="Macro-recall корня", xlabel="Среднее по фолдам, %")
    axs[2].set_yticks(range(len(CELLS)), list(CELLS.values()))
    axs[2].invert_yaxis()
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dedede", linewidth=0.6)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate(results: Path, folds: pd.DataFrame, near_dup: pd.DataFrame, n_boot: int) -> dict:
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")
    runs, frames, protocol = load_grid(results, folds)
    table, paired = paired_results(runs, frames, n_boot)
    seeds, selection = seed_results(runs, paired)
    errors, sensitivity = diagnostics(frames, folds, near_dup)
    summary = {
        "primary_seed": SEEDS[0],
        "n_runs": len(runs),
        "protocol": protocol,
        "bootstrap": {"n_resamples": n_boot, "seed": BOOT_SEED},
        "table": table,
        "paired": paired,
        "selection": selection,
        "seeds": seeds,
        "diagnostics": errors,
        "leak_sensitivity": sensitivity,
    }
    tracker = pd.json_normalize(runs.to_dict("records"))
    tracker.insert(1, "prior_strength", tracker.cell.isin(["p2p", "both"]).astype(int) * protocol["hw2"]["prior_strength"])
    tracker.insert(2, "gda_stages", np.where(tracker.cell.isin(["gda", "both"]), "+".join(protocol["hw2"]["stages"]), ""))
    tracker.insert(3, "config_file", [f"{r.cell}/fold{r.fold}_seed{r.seed}/config.yaml" for r in tracker.itertuples()])
    tracker.to_csv(results / "experiments.csv", index=False)
    save_json(summary, results / "summary.json")
    (results / "summary.md").write_text(markdown(summary), encoding="utf-8")
    plot_comparison(summary, results / "comparison.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/hw2")
    parser.add_argument("--folds-file", default="splits/folds.csv")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--near-duplicates", default="reports/near_duplicates.csv")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    results = resolve(args.results)
    folds = load_folds(args.folds_file, args.train_csv)
    near_dup = pd.read_csv(resolve(args.near_duplicates))
    evaluate(results, folds, near_dup, args.n_boot)
    print(f"Saved HW2 evaluation in {results}")


if __name__ == "__main__":
    main()
