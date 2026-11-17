import json

import pandas as pd
import pytest
import yaml

from bengali import hw2_diagnostics as diagnostics
from bengali.hw2_evaluate import CELLS, FOLDS, expected_rows
from bengali.metric import slice_scores
from bengali.utils import COMPONENTS


def test_root_context_bins_average_fold_root_units_not_images():
    records = []
    for root, n_contexts in enumerate([1, 2, 5, 10, 11]):
        for j in range(n_contexts):
            records.append((f"train-{root}-{j}", f"train-{root}-{j}", root, 1, -1))
        for j in range(3 if root == 1 else 1):
            records.append((f"held-{root}-{j}", f"held-{root}", root, 0, 0))
    folds = pd.DataFrame(records, columns=["image_id", "grapheme", "grapheme_root", "id_fold", "ood_group"])
    folds["consonant_diacritic"] = folds["vowel_diacritic"] = 0
    held = folds[folds.ood_group == 0].reset_index(drop=True).assign(fold=0, slice="ood")
    held["pred_grapheme_root"] = held["grapheme_root"].mask(held["grapheme_root"] == 1, 0)
    frames = {cell: held for cell in CELLS}
    bins = diagnostics.root_contexts(frames, folds)
    assert list(bins) == ["1–2", "3–5", "6–10", ">10"]
    assert bins["1–2"] == {"fold_root_pairs": 2, "ood_images": 4, "recall": dict.fromkeys(CELLS, 0.5)}
    for label in ("3–5", "6–10", ">10"):
        assert bins[label] == {"fold_root_pairs": 1, "ood_images": 1, "recall": dict.fromkeys(CELLS, 1.0)}


@pytest.fixture
def artifacts(tmp_path):
    rows = []
    for k in FOLDS:
        rows += [(f"id-{k}", "id-root", 0, 0, 0, k, -1), (f"held-{k}", f"held-root-{k}", 1, 1, 1, k, k)]
    folds = pd.DataFrame(rows, columns=["image_id", "grapheme", *COMPONENTS, "id_fold", "ood_group"])
    for k in FOLDS:
        for cell in diagnostics.NAMES:
            run = tmp_path / ("" if cell in CELLS else "controls") / cell / f"fold{k}_seed42"
            run.mkdir(parents=True)
            frame = expected_rows(folds, k)
            for c in COMPONENTS:
                frame[f"pred_{c}"] = frame[c]
            if cell == "freq":
                frame.loc[frame["slice"] == "ood", "pred_grapheme_root"] = 0
            frame[["image_id", "slice", *["pred_" + c for c in COMPONENTS]]].iloc[::-1].to_csv(run / "preds.csv.gz", index=False)
            name = f"hw2_{cell}"
            metrics = {
                "experiment": name, "fold": k, "seed": 42, **slice_scores(frame),
                "checkpoint_sha256": str(k) * 64, "folds_sha256": "b" * 64, "archive_predictions_sha256": "c" * 64,
                "archive_parity": {"rows": len(frame), "all_classes_match": True},
                "device": "cpu", "versions": {"torch": "fixture"},
            }
            (run / "metrics.json").write_text(json.dumps(metrics))
            config = {"experiment": name, "run": {"fold": k, "seed": 42},
                      "hw2": {"ridge": 1.0, "alpha": 1.0, "prior_strength": 1.0, "stages": ["layer3", "layer4"]}}
            (run / "config.yaml").write_text(yaml.safe_dump(config))
    return tmp_path, folds


def test_supplemental_report_uses_five_paired_folds_and_controls(artifacts):
    results, folds = artifacts
    report = diagnostics.evaluate(results, folds, n_boot=20)
    contrast = report["comparisons"]["p2p-freq"]
    assert contrast["per_fold"]["main"] == [0.25] * 5
    for stat in ("point", "lo", "hi"):
        assert contrast["bootstrap"]["main"][stat] == pytest.approx(0.25)
        assert contrast["bootstrap"]["ood"][stat] == pytest.approx(0.5)
    assert report["comparisons"]["gda-final_gda"]["bootstrap"]["main"]["point"] == 0.0
    assert json.loads((results / "diagnostics.json").read_text()) == report
    md = (results / "diagnostics.md").read_text()
    assert "Фолд 4" in md and "Графем в train" in md
    assert report["ood_root_by_context"]["3–5"]["fold_root_pairs"] == 5
    assert report["ood_root_by_context"][">10"]["recall"]["b2"] is None


@pytest.mark.parametrize("tamper", ["checkpoint_sha256", "main_score", "experiment"])
def test_corrupt_control_cannot_publish_report(artifacts, tamper):
    results, folds = artifacts
    path = results / "controls" / "freq" / "fold0_seed42" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics[tamper] = {"checkpoint_sha256": "e" * 64, "main_score": -1.0, "experiment": "hw2_b2"}[tamper]
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="fold0_seed42"):
        diagnostics.evaluate(results, folds, n_boot=2)
    assert not (results / "diagnostics.json").exists()
