"""Aggregate runs, compute CIs, compare models on identical splits, analyse errors.

Usage: python -m bengali.evaluate [--main b2_resnet18]
Writes results/runs.csv, results/summary.json, results/summary.md, reports/figures/res_*.png

Conventions
* The reported score of a model is the mean over the 5 folds of the per-fold slice scores (as in metrics.json);
  "± std" is the spread over folds. It includes the choice of held-out graphemes, which differs per fold.
* 95% CI: percentile bootstrap of that same fold mean (metric.bootstrap_ci). Images are resampled inside each
  grapheme, graphemes and trained models stay fixed, so the CI measures the noise of the finite image sample only.
* Paired comparisons use the same folds and the same bootstrap draws for both models (metric.paired_bootstrap).
* "Seen in train" always means seen in the *fold's* train (folds.csv), never in the whole train.csv.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metric import bootstrap_ci, paired_bootstrap, per_class_recall, slice_scores
from .splits import fold_indices, load_folds
from .utils import COMPONENTS, N_CLASSES, resolve, save_json

ORDER = ["b0_constant", "b1_pixels_linear", "b3_dinov3_vits16_probe", "b4_mobileclip2_s0_probe", "b5_dinov3_vits16_ft_last4", "b2_resnet18"]
LABEL = {
    "b0_constant": "B0 constant",
    "b1_pixels_linear": "B1 pixels32 + linear",
    "b3_dinov3_vits16_probe": "B3 DINOv3 ViT-S/16 (frozen) + linear",
    "b4_mobileclip2_s0_probe": "B4 MobileCLIP2-S0 (frozen) + linear",
    "b5_dinov3_vits16_ft_last4": "B5 DINOv3 ViT-S/16 (4 blocks tuned; fold 0)",
    "b2_resnet18": "B2 ResNet18 fine-tune",
}


def collect_runs(results_dir) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("*/fold*_seed*/metrics.json")):
        if p.parts[-3].startswith("_"):
            continue
        rows.append(json.loads(p.read_text()))
    return pd.DataFrame(rows)


def load_oof(results_dir, exp: str, seed: int, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Out-of-fold predictions of all folds; labels (image_id, grapheme, components) are joined from train.csv."""
    parts = []
    for p in sorted((results_dir / exp).glob(f"fold*_seed{seed}/preds.csv.gz")):
        f = pd.read_csv(p)
        f = f.drop(columns=[c for c in ("grapheme", *COMPONENTS) if c in f])  # older runs stored labels; always re-join
        f["fold"] = int(p.parent.name.split("_")[0][4:])
        parts.append(f)
    oof = pd.concat(parts, ignore_index=True)
    if labels is None:
        labels = pd.read_csv(resolve("data/raw/train.csv"))
    oof = oof.merge(labels[["image_id", "grapheme", *COMPONENTS]], on="image_id", how="left", validate="many_to_one")
    assert oof[COMPONENTS].notna().all().all(), "predictions for unknown image ids"
    return oof


def fmt(m, s=None, d=4):
    return f"{m:.{d}f}" if s is None or np.isnan(s) else f"{m:.{d}f} ± {s:.{d}f}"


def triples(df: pd.DataFrame) -> list[tuple]:
    return list(map(tuple, df[COMPONENTS].to_numpy()))


def fold_context(folds: pd.DataFrame) -> dict:
    """Per fold: label triples, graphemes per root and images per grapheme in the fold's *train* part."""
    ctx = {}
    for k in sorted(folds.id_fold.unique()):
        tr = folds.iloc[fold_indices(folds, k)["train"]]
        ctx[int(k)] = {
            "seen": set(triples(tr.drop_duplicates(COMPONENTS))),
            "graphemes_per_root": tr.drop_duplicates("grapheme").grapheme_root.value_counts(),
            "train_ids": set(tr.image_id),
        }
    return ctx


def leak_masks(oof: pd.DataFrame, ctx: dict, near_dup: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """(ID rows whose near-duplicate twin is in the fold's train, OOD rows whose label triple is in the fold's train)."""
    twin = {}
    if near_dup is not None:
        for a, b in zip(near_dup.image_id_a, near_dup.image_id_b):
            twin.setdefault(a, []).append(b)
            twin.setdefault(b, []).append(a)
    nd = np.array([s == "id" and any(t in ctx[f]["train_ids"] for t in twin.get(i, ())) for i, s, f in zip(oof.image_id, oof.slice, oof.fold)])
    tw = np.array([s == "ood" and t in ctx[f]["seen"] for t, s, f in zip(triples(oof), oof.slice, oof.fold)])
    return nd, tw


def fold_mean_scores(oof: pd.DataFrame, drop: np.ndarray | None = None) -> dict:
    d = oof if drop is None else oof[~drop]
    per = [slice_scores(g) for _, g in d.groupby("fold")]
    return {k: float(np.mean([p[f"{k}_score"] for p in per])) for k in ("id", "ood", "main")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--main", default="b2_resnet18")
    ap.add_argument("--seed", type=int, default=42, help="seed used for the 5-fold comparison")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--folds-file", default="splits/folds.csv")
    ap.add_argument("--figures", default="reports/figures")
    a = ap.parse_args()
    res = resolve(a.results)
    figs = resolve(a.figures)
    figs.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(resolve(a.raw_dir) / "train.csv")
    folds = load_folds(a.folds_file, resolve(a.raw_dir) / "train.csv")
    ctx = fold_context(folds)
    nd_path = resolve("reports/near_duplicates.csv")
    near_dup = pd.read_csv(nd_path) if nd_path.exists() else None
    runs = collect_runs(res)
    runs.to_csv(res / "runs.csv", index=False)
    md: list[str] = []
    out: dict = {}

    exps = [e for e in ORDER if e in set(runs.experiment)] + sorted(set(runs.experiment) - set(ORDER))
    seed_of = {e: (0 if e == "b0_constant" else a.seed) for e in exps}
    oofs = {e: load_oof(res, e, seed_of[e], train) for e in exps}

    # ------------------------------------------------------------ 1. per-fold mean ± std + bootstrap CI of that mean
    md += [
        "### Сводная таблица",
        "",
        "Все модели на одних и тех же 5 фолдах. Значение — среднее по фолдам ± std по фолдам. В std входит и то, какие графемы отложены в OOD (в каждом фолде свои 10%).",
        "95% CI — бутстрап того же среднего: изображения пересэмплируются внутри графем, графемы и обученные модели фиксированы. Поэтому CI отражает только шум конечной выборки изображений и намного уже std по фолдам.",
        "",
        "| модель | main = (ID+OOD)/2 | 95% CI main | ID | OOD | ID root / vowel / cons | OOD root / vowel / cons | обучение, с/фолд | признаки, с (разово) |",
        "|---|---|---|---|---|---|---|---:|---:|",
    ]
    table = {}
    for e in exps:
        r = runs[(runs.experiment == e) & (runs.seed == seed_of[e])]
        if r.empty:
            continue
        ci = bootstrap_ci(oofs[e], n_boot=a.n_boot, seed=0)
        assert abs(ci["main"]["point"] - r.main_score.mean()) < 1e-6, (e, ci["main"]["point"], r.main_score.mean())
        fit = next((float(r[c].mean()) for c in ("train_sec", "probe_sec") if c in r and r[c].notna().any()), 0.0)
        ext = float(r["extract_sec"].dropna().iloc[0]) if "extract_sec" in r and r["extract_sec"].notna().any() else None
        row = {k: (float(r[k].mean()), float(r[k].std(ddof=1)) if len(r) > 1 else float("nan")) for k in r.columns if k.endswith(("_score", "_root", "_vowel", "_consonant"))}
        table[e] = {"n_folds": int(len(r)), "metrics": row, "ci": ci, "fit_sec_per_fold": fit, "extract_sec": ext}
        md.append(
            f"| {LABEL.get(e, e)} | {fmt(*row['main_score'])} | [{ci['main']['lo']:.4f}, {ci['main']['hi']:.4f}] | {fmt(*row['id_score'])} | {fmt(*row['ood_score'])} | "
            f"{row['id_root'][0]:.3f} / {row['id_vowel'][0]:.3f} / {row['id_consonant'][0]:.3f} | "
            f"{row['ood_root'][0]:.3f} / {row['ood_vowel'][0]:.3f} / {row['ood_consonant'][0]:.3f} | {fit:.0f} | {'' if ext is None else f'{ext:.0f}'} |"
        )
    out["table"] = table
    md += [
        "",
        "_main_ — основная метрика выбора модели. ID — графемы, которые встречались в train фолда. OOD — 130 графем, целиком убранных из train фолда (прокси невиданных графем private-теста). Покомпонентные recall усреднены по фолдам.",
        "",
    ]

    # ------------------------------------------------------------ 2. paired comparison with the main baseline
    if a.main in table:
        md += [
            f"### Парное сравнение с {LABEL.get(a.main, a.main)} (одни и те же фолды и изображения)",
            "",
            "| модель | Δmain, mean ± std по фолдам | фолдов лучше | 95% CI Δmain (парный бутстрап) | ΔID | ΔOOD | 95% CI ΔOOD |",
            "|---|---|---:|---|---|---|---|",
        ]
        base = runs[(runs.experiment == a.main) & (runs.seed == a.seed)].set_index("fold")
        out["paired"] = {}
        for e in exps:
            if e == a.main or e not in table:
                continue
            r = runs[(runs.experiment == e) & (runs.seed == seed_of[e])].set_index("fold")
            common = base.index.intersection(r.index)
            if len(common) == 0:
                continue
            d = {k: (r.loc[common, f"{k}_score"] - base.loc[common, f"{k}_score"]) for k in ("main", "id", "ood")}
            pb = paired_bootstrap(oofs[e][oofs[e].fold.isin(common)], oofs[a.main][oofs[a.main].fold.isin(common)], n_boot=a.n_boot, seed=1)
            out["paired"][e] = {"per_fold": {k: v.round(6).tolist() for k, v in d.items()}, "bootstrap": pb}
            md.append(
                f"| {LABEL.get(e, e)} | {fmt(d['main'].mean(), d['main'].std(ddof=1))} | {int((d['main'] > 0).sum())}/{len(common)} | "
                f"[{pb['main']['lo']:+.4f}, {pb['main']['hi']:+.4f}] | {d['id'].mean():+.4f} | {d['ood'].mean():+.4f} | [{pb['ood']['lo']:+.4f}, {pb['ood']['hi']:+.4f}] |"
            )
        md.append("")

    # ------------------------------------------------------------ 3. seed vs fold variance (RQ2)
    mr = runs[runs.experiment == a.main]
    if mr.seed.nunique() > 1:
        f0 = mr[mr.fold == mr[mr.seed != a.seed].fold.min()]
        folds_ = mr[mr.seed == a.seed]
        seed_std = {k: float(f0[f"{k}_score"].std(ddof=1)) for k in ("main", "id", "ood")}
        fold_std = {k: float(folds_[f"{k}_score"].std(ddof=1)) for k in ("main", "id", "ood")}
        boot_std = {k: table[a.main]["ci"][k]["std"] * np.sqrt(len(folds_)) for k in ("main", "id", "ood")} if a.main in table else {}
        out["variance"] = {"seed_std_fold": seed_std, "fold_std_seed": fold_std, "n_seeds": int(f0.seed.nunique()), "fold": int(f0.fold.iloc[0]), "seed_scores": f0[["seed", "main_score", "id_score", "ood_score"]].to_dict("records")}
        md += [
            "### Разброс: seed vs фолд vs выборка изображений (RQ2)",
            "",
            f"| score | std по seed (фолд {int(f0.fold.iloc[0])}, n={f0.seed.nunique()}) | std по фолдам (seed {a.seed}, n={len(folds_)}) | бутстрап-std одного фолда (≈) |",
            "|---|---:|---:|---:|",
        ]
        for k in ("main", "id", "ood"):
            md.append(f"| {k} | {seed_std[k]:.4f} | {fold_std[k]:.4f} | {boot_std.get(k, float('nan')):.4f} |")
        md += ["", "Seed: " + ", ".join(f"{int(s.seed)} → main {s.main_score:.4f} (ID {s.id_score:.4f}, OOD {s.ood_score:.4f})" for s in f0.itertuples()), ""]

    # ------------------------------------------------------------ 4. leakage sensitivity
    md += [
        "### Чувствительность к найденным утечкам",
        "",
        "Пересчёт среднего по фолдам без затронутых изображений: (а) ID-изображения, у которых в train фолда есть почти точная копия (тот же рукописный образец, сохранённый дважды; `reports/near_duplicates.csv`); (б) OOD-изображения графем, чья тройка меток всё-таки есть в train фолда под другим написанием (3 пары графем-двойников, §3).",
        "",
        "| модель | ID | ID без копий | Δ | OOD | OOD без двойников | Δ | main без обоих |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    out["leak_sensitivity"] = {}
    for e in exps:
        if e not in table:
            continue
        nd, tw = leak_masks(oofs[e], ctx, near_dup)
        full, no_nd, no_tw, no_both = fold_mean_scores(oofs[e]), fold_mean_scores(oofs[e], nd), fold_mean_scores(oofs[e], tw), fold_mean_scores(oofs[e], nd | tw)
        out["leak_sensitivity"][e] = {"n_id_near_dup_per_fold": float(nd.sum() / oofs[e].fold.nunique()), "n_ood_twin_per_fold": float(tw.sum() / oofs[e].fold.nunique()), "full": full, "no_near_dup": no_nd, "no_twin": no_tw, "no_both": no_both}
        md.append(
            f"| {LABEL.get(e, e)} | {full['id']:.4f} | {no_nd['id']:.4f} | {no_nd['id'] - full['id']:+.4f} | {full['ood']:.4f} | {no_tw['ood']:.4f} | {no_tw['ood'] - full['ood']:+.4f} | {no_both['main']:.4f} |"
        )
    if out["leak_sensitivity"]:
        s = next(iter(out["leak_sensitivity"].values()))
        md += ["", f"Затронуто в среднем на фолд: {s['n_id_near_dup_per_fold']:.0f} ID-изображений с копией в train и {s['n_ood_twin_per_fold']:.0f} OOD-изображений графем-двойников.", ""]

    # ------------------------------------------------------------ 5. error analysis of the main baseline
    if a.main in table:
        md += error_analysis(oofs[a.main].copy(), ctx, out, figs, resolve(a.raw_dir), resolve(a.processed_dir), a.main)

    # ------------------------------------------------------------ figures
    plot_comparison(table, figs)
    plot_learning_curves(res, a.main, a.seed, figs)
    save_json(out, res / "summary.json")
    (res / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def snap_rates(err: pd.DataFrame, ctx: dict, conf: list[np.ndarray], rng, n_rep: int = 20) -> tuple[float, float, float]:
    """Share of OOD errors whose predicted triple is a triple of the fold's train, and two chance levels.

    Chance levels keep *which* components are wrong and redraw only their values:
    uniformly among the wrong classes, or from the model's own ID confusion distribution P(pred | true, pred != true).
    """
    if err.empty:
        return float("nan"), float("nan"), float("nan")
    obs = np.mean([t in ctx[f]["seen"] for t, f in zip(zip(*(err[f"pred_{c}"] for c in COMPONENTS)), err.fold)])
    t = err[COMPONENTS].to_numpy()
    p = err[[f"pred_{c}" for c in COMPONENTS]].to_numpy()
    wrong = t != p
    hits = {"uniform": [], "confusion": []}
    for _ in range(n_rep):
        for kind in hits:
            q = t.copy()
            for j, c in enumerate(COMPONENTS):
                rows = np.flatnonzero(wrong[:, j])
                q[rows, j] = (t[rows, j] + rng.integers(1, N_CLASSES[c], len(rows))) % N_CLASSES[c]  # uniform fallback
                if kind == "confusion":
                    for cls in np.unique(t[rows, j]):
                        sel = rows[t[rows, j] == cls]
                        w = conf[j][cls].astype(float).copy()
                        w[cls] = 0
                        if w.sum() > 0:
                            q[sel, j] = rng.choice(N_CLASSES[c], size=len(sel), p=w / w.sum())
            hits[kind].append(np.mean([tuple(x) in ctx[f]["seen"] for x, f in zip(q, err.fold)]))
    return float(obs), float(np.mean(hits["uniform"])), float(np.mean(hits["confusion"]))


def error_analysis(oof, ctx, out, figs, raw_dir, proc_dir, exp) -> list[str]:
    md = []
    rng = np.random.default_rng(0)
    train = pd.read_csv(raw_dir / "train.csv")
    cm = pd.read_csv(raw_dir / "class_map.csv")
    glyph = {(r.component_type, r.label): r.component for r in cm.itertuples()}
    from .eda import root_types

    rt = root_types(raw_dir)
    freq = train.grapheme_root.value_counts()

    # per-component error rate per slice
    md += [
        "### Анализ ошибок основного бейзлайна (OOF, 5 фолдов)",
        "",
        "| срез | n | ошибка root | ошибка vowel | ошибка cons | хотя бы одна ошибка |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    oof["any_err"] = np.any([oof[c] != oof[f"pred_{c}"] for c in COMPONENTS], axis=0)
    oof["root_type"] = oof.grapheme_root.map(rt)
    multi_ids = set(pd.read_csv(raw_dir / "train_multi_diacritics.csv").image_id) if (raw_dir / "train_multi_diacritics.csv").exists() else set()
    oof["multi_diacritic"] = oof.image_id.isin(multi_ids)
    oof["grapheme_count"] = oof.grapheme.map(train.grapheme.value_counts())

    def row(name, d):
        errs = [(d[c] != d[f"pred_{c}"]).mean() for c in COMPONENTS]
        return f"| {name} | {len(d)} | {errs[0]:.2%} | {errs[1]:.2%} | {errs[2]:.2%} | {d.any_err.mean():.2%} |"

    for sl in ("id", "ood"):
        md.append(row(f"slice={sl}", oof[oof.slice == sl]))
    for sl in ("id", "ood"):
        for t in ("vowel", "consonant", "conjunct"):
            d = oof[(oof.slice == sl) & (oof.root_type == t)]
            if len(d):
                md.append(row(f"{sl.upper()}, тип корня = {t}", d))
    q = oof[oof.slice == "id"].grapheme_count
    for b, d in oof[oof.slice == "id"].groupby(pd.qcut(q, 4, duplicates="drop"), observed=True):
        md.append(row(f"ID, картинок графемы в датасете {b}", d))
    md.append(row("ID, двойной согласный диакритик (шум разметки)", oof[(oof.slice == "id") & oof.multi_diacritic]))
    md.append("")

    # ---- "snap to a seen grapheme": OOD errors whose predicted triple is a triple of the fold's train
    pid = oof[oof.slice == "id"]
    conf = [np.bincount(pid[c] * N_CLASSES[c] + pid[f"pred_{c}"], minlength=N_CLASSES[c] ** 2).reshape(N_CLASSES[c], N_CLASSES[c]) for c in COMPONENTS]
    ood_err = oof[(oof.slice == "ood") & oof.any_err]
    snap, null_u, null_c = snap_rates(ood_err, ctx, conf, rng)
    out["ood_snap"] = {"observed": snap, "chance_uniform": null_u, "chance_id_confusion": null_c, "n_errors": int(len(ood_err))}
    verdict = "заметно выше" if snap > null_c + 0.05 else "не выше"
    md += [
        f"**Притяжение к знакомым графемам.** В {snap:.1%} ошибок на OOD-срезе предсказанная тройка (root, vowel, cons) — это графема из train того же фолда.",
        f"Уровень случайности при тех же ошибочных компонентах: {null_u:.1%}, если ошибочное значение равномерно случайно, и {null_c:.1%}, если оно берётся из матрицы путаниц модели на ID.",
        f"Наблюдаемая доля {verdict} второго уровня: модель {'действительно' if verdict == 'заметно выше' else 'не'} притягивает невиданную комбинацию к знакомой сверх того, что объясняется путаницей отдельных компонент.",
        "",
    ]

    # ---- worst roots and confusions (ID slice)
    pcr = per_class_recall(pid.grapheme_root.to_numpy(), pid.pred_grapheme_root.to_numpy(), 168)
    pcr["images"] = pcr.cls.map(freq)
    pcr["type"] = pcr.cls.map(rt)
    pcr["glyph"] = pcr.cls.map(lambda i: glyph.get(("grapheme_root", i), "?"))
    worst = pcr.dropna().sort_values("recall").head(10)
    md += ["#### 10 корней с худшим recall (ID)", "", "| root | глиф | тип | картинок в датасете | recall |", "|---:|:---:|---|---:|---:|"]
    md += [f"| {r.cls} | {r.glyph} | {r.type} | {r.images} | {r.recall:.3f} |" for r in worst.itertuples()]
    md.append("")
    err = pid[pid.grapheme_root != pid.pred_grapheme_root]
    confs = err.groupby(["grapheme_root", "pred_grapheme_root"]).size().sort_values(ascending=False).head(10)
    md += ["#### Самые частые путаницы корней (ID)", "", "| истинный | → предсказанный | раз |", "|---|---|---:|"]
    md += [f"| {t} {glyph.get(('grapheme_root', t), '')} | {p} {glyph.get(('grapheme_root', p), '')} | {n} |" for (t, p), n in confs.items()]
    md.append("")
    out["worst_roots"] = worst[["cls", "glyph", "type", "images", "recall"]].to_dict("records")
    out["top_root_confusions"] = [{"true": int(t), "pred": int(p), "n": int(n)} for (t, p), n in confs.items()]

    # ---- OOD: does root recall depend on how many graphemes with that root the fold's train contains?
    rows = []
    for f, d in oof[oof.slice == "ood"].groupby("fold"):
        r = per_class_recall(d.grapheme_root.to_numpy(), d.pred_grapheme_root.to_numpy(), 168).dropna()
        r["graphemes_in_train"] = r.cls.map(ctx[int(f)]["graphemes_per_root"]).fillna(0).astype(int)
        r["fold"] = f
        rows.append(r)
    ood_roots = pd.concat(rows, ignore_index=True)
    rho = float(ood_roots.recall.rank().corr(ood_roots.graphemes_in_train.rank()))
    bins = pd.cut(ood_roots.graphemes_in_train, [0, 2, 5, 10, 100], labels=["1–2", "3–5", "6–10", ">10"])
    by_bin = ood_roots.groupby(bins, observed=True).recall.agg(["count", "mean"])
    out["ood_root_recall_vs_context"] = {"spearman": rho, "by_bin": {str(k): {"n": int(v["count"]), "mean_recall": float(v["mean"])} for k, v in by_bin.iterrows()}}
    md += [
        "#### OOD: recall корня vs число графем с этим корнем в train фолда",
        "",
        "| графем с корнем в train | пар (корень, фолд) | средний recall корня на OOD |",
        "|---|---:|---:|",
    ]
    md += [f"| {k} | {int(v['count'])} | {v['mean']:.3f} |" for k, v in by_bin.iterrows()]
    md += ["", f"Ранговая корреляция Спирмена: ρ = {rho:.2f}.", ""]

    # ---- figures: recall vs frequency (ID) and vs context diversity (OOD)
    colors = {"vowel": "#d62728", "consonant": "#1f77b4", "conjunct": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for t, d in pcr.dropna().groupby("type"):
        axes[0].scatter(d.images, d.recall, s=12, label=t, color=colors[t])
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    axes[0].set_xscale("log")
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xticks([150, 300, 1000, 3000, 6000])
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_xlabel("images of the root in the dataset (log)")
    axes[0].set_ylabel("root recall")
    axes[0].set_title("ID slice: root recall vs frequency")
    axes[0].legend()
    jitter = np.exp(rng.uniform(-0.08, 0.08, len(ood_roots)))  # multiplicative jitter on a log axis
    axes[1].scatter(ood_roots.graphemes_in_train * jitter, ood_roots.recall, s=10, alpha=0.6, color="#555")
    axes[1].set_xscale("log")
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xticks([1, 2, 3, 5, 10, 20, 35])
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].set_xlabel("graphemes with this root in the fold's train")
    axes[1].set_ylabel("root recall")
    axes[1].set_title(f"OOD slice: root recall vs context diversity (Spearman {rho:.2f})")
    fig.tight_layout()
    fig.savefig(figs / "res_root_recall_vs_freq.png", dpi=130)
    plt.close(fig)

    # ---- confidence vs accuracy (root), both slices
    fig, ax = plt.subplots(figsize=(6, 4))
    for sl, d in oof.groupby("slice"):
        b = pd.cut(d.conf_grapheme_root, np.linspace(0, 1, 11))
        acc = (d.grapheme_root == d.pred_grapheme_root).groupby(b, observed=True).mean()
        ax.plot([i.mid for i in acc.index], acc.values, marker="o", label=sl)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("max softmax prob (root)")
    ax.set_ylabel("root accuracy")
    ax.set_title("Reliability: ID vs OOD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "res_reliability.png", dpi=130)
    plt.close(fig)

    # ---- gallery of typical errors: random examples (fixed seed), not the most confident ones
    imgs = np.load(proc_dir / "train_128.npy", mmap_mode="r")
    pos = pd.Series(np.arange(len(train)), index=train.image_id)
    snap_mask = np.array([t in ctx[f]["seen"] for t, f in zip(zip(*(ood_err[f"pred_{c}"] for c in COMPONENTS)), ood_err.fold)], dtype=bool)
    ok_rv = (pid.grapheme_root == pid.pred_grapheme_root) & (pid.vowel_diacritic == pid.pred_vowel_diacritic)
    cases = {
        "ID: путаница похожих корней": pid[(pid.grapheme_root == confs.index[0][0]) & (pid.pred_grapheme_root == confs.index[0][1])] if len(confs) else pid.head(0),
        "ID: ошибка только в согласном диакритике": pid[ok_rv & (pid.consonant_diacritic != pid.pred_consonant_diacritic)],
        "OOD: невиданная графема → графема из train": ood_err[snap_mask],
    }
    cases = {k: v.sample(min(4, len(v)), random_state=0) for k, v in cases.items() if len(v)}
    if cases:
        fig, axes = plt.subplots(len(cases), 4, figsize=(10, 2.8 * len(cases)), squeeze=False)
        for i, (title, d) in enumerate(cases.items()):
            for j in range(4):
                ax = axes[i, j]
                ax.axis("off")
                if j < len(d):
                    r = d.iloc[j]
                    ax.imshow(imgs[pos[r.image_id]], cmap="gray")
                    ax.set_title(
                        f"true {r.grapheme_root}/{r.vowel_diacritic}/{r.consonant_diacritic}\npred {r.pred_grapheme_root}/{r.pred_vowel_diacritic}/{r.pred_consonant_diacritic}",
                        fontsize=8,
                    )
            axes[i, 0].text(-8, 64, title, fontsize=9, ha="right", va="center", wrap=True)
        fig.suptitle("Typical errors, random examples (root/vowel/consonant)")
        fig.tight_layout()
        fig.savefig(figs / "res_error_gallery.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    out["gallery_cases"] = {k: v[["image_id", *(f"pred_{c}" for c in COMPONENTS)]].to_dict("records") for k, v in cases.items()}  # no labels
    return md


def plot_comparison(table: dict, figs) -> None:
    if not table:
        return
    names = list(table)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(names))
    for k, (key, color) in enumerate((("id", "#1f77b4"), ("ood", "#d62728"), ("main", "#333"))):
        means = [table[n]["metrics"][f"{key}_score"][0] for n in names]
        stds = [np.nan_to_num(table[n]["metrics"][f"{key}_score"][1]) for n in names]
        ax.bar(x + (k - 1) * 0.27, means, 0.27, yerr=stds, capsize=3, color=color, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(n, n).replace(" + ", "\n+ ").replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("hierarchical macro recall")
    ax.legend()
    ax.set_title("Same folds for every model (B5: fold 0 only): mean over folds, whiskers = ±1 std over folds", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "res_model_comparison.png", dpi=130)
    plt.close(fig)


def plot_learning_curves(res, exp, seed, figs) -> None:
    hs = sorted((res / exp).glob(f"fold*_seed{seed}/history.json"))
    if not hs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for p in hs:
        h = pd.DataFrame(json.loads(p.read_text()))
        f = p.parent.name.split("_")[0]
        axes[0].plot(h.epoch, h.train_loss, marker="o", label=f)
        if "id_score" in h:
            axes[1].plot(h.epoch, h.id_score, marker="o", color="#1f77b4", alpha=0.6)
            axes[1].plot(h.epoch, h.ood_score, marker="s", color="#d62728", alpha=0.6)
    axes[0].set_title("train loss (sum of 3 CE)")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=7)
    axes[1].set_title("val score per epoch (blue ID, red OOD), logged only", fontsize=10)
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(figs / "res_learning_curves.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
