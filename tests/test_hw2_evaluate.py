import hashlib
import json

import pandas as pd
import pytest
import yaml

from bengali import hw2_evaluate as hw2
from bengali.metric import slice_scores
from bengali.splits import fold_indices
from bengali.utils import COMPONENTS


def write_run(results, folds, cell, fold, seed):
    run = results / cell / f"fold{fold}_seed{seed}"
    run.mkdir(parents=True)
    indices = fold_indices(folds, fold)
    frame = pd.concat(
        [folds.iloc[indices[f"val_{part}"]].assign(slice=part) for part in ("id", "ood")],
        ignore_index=True,
    )
    for column in COMPONENTS:
        frame[f"pred_{column}"] = frame[column]
        frame[f"conf_{column}"] = 0.9
    if cell in ("b2", "p2p") and seed != 44:
        frame.loc[frame["slice"] == "ood", [f"pred_{column}" for column in COMPONENTS]] = 0
    if cell == "both":
        frame.loc[frame["image_id"] == f"id0-{fold}", "pred_grapheme_root"] = 1
    columns = ["image_id", "slice", *[f"pred_{c}" for c in COMPONENTS], *[f"conf_{c}" for c in COMPONENTS]]
    frame[columns].sample(frac=1, random_state=len(cell)).to_csv(run / "preds.csv.gz", index=False)
    experiment = f"hw2_{cell}"
    metrics = {
        "experiment": experiment,
        "fold": fold,
        "seed": seed,
        "checkpoint_sha256": hashlib.sha256(f"checkpoint-{fold}-{seed}".encode()).hexdigest(),
        "folds_sha256": hashlib.sha256(b"folds").hexdigest(),
        "archive_predictions_sha256": hashlib.sha256(f"archive-{fold}-{seed}".encode()).hexdigest(),
        "archive_parity": {"all_classes_match": True, "rows": len(frame)},
        **slice_scores(frame),
    }
    (run / "metrics.json").write_text(json.dumps(metrics))
    config = {
        "experiment": experiment,
        "hw2": {"stages": ["layer3", "layer4"], "ridge": 1.0, "alpha": 1.0, "prior_strength": 1.0},
        "run": {"fold": fold, "seed": seed},
    }
    (run / "config.yaml").write_text(yaml.safe_dump(config))


@pytest.fixture
def grid(tmp_path):
    rows = []
    for fold in range(5):
        rows.extend((f"id{copy}-{fold}", "seen", 0, 0, 0, fold, -1) for copy in range(2))
        rows.append((f"ood-{fold}", f"held-{fold}", 1, fold + 1, 0, fold, fold))
    folds = pd.DataFrame(rows, columns=["image_id", "grapheme", *COMPONENTS, "id_fold", "ood_group"])
    near_dup = pd.DataFrame({"image_id_a": ["id0-0"], "image_id_b": ["id0-1"]})
    for cell in ("b2", "p2p", "gda", "both"):
        for fold, seed in [(k, 42) for k in range(5)] + [(0, 43), (0, 44)]:
            write_run(tmp_path, folds, cell, fold, seed)
    return tmp_path, folds, near_dup


def test_complete_grid_generates_order_independent_paired_effects_and_report(grid):
    results, folds, near_dup = grid
    summary = hw2.evaluate(results, folds, near_dup, n_boot=20)
    assert len(pd.read_csv(results / "experiments.csv")) == 28
    assert json.loads((results / "summary.json").read_text()) == summary
    assert (results / "summary.md").read_text().startswith("# ДЗ2")
    assert (results / "comparison.png").read_bytes().startswith(b"\x89PNG")
    paired = summary["paired"]
    assert paired["gda-b2"]["bootstrap"]["main"]["point"] == pytest.approx(0.375)
    assert paired["both-gda"]["bootstrap"]["main"]["point"] == pytest.approx(-0.125)
    assert paired["interaction"]["bootstrap"]["ood"]["point"] == 0
    assert summary["diagnostics"]["b2"]["ood"]["conditional_seen_error_rate"] == 1
    assert summary["diagnostics"]["gda"]["ood"]["conditional_seen_error_rate"] is None
    assert summary["diagnostics"]["gda"]["ood"]["unconditional_seen_error_rate"] == 0
    assert summary["leak_sensitivity"]["n_id_near_duplicate"] == 2


@pytest.mark.parametrize("tamper", [
    "missing", "mislabeled", "duplicate", "wrong_metrics", "mixed_checkpoint", "wrong_variant",
    "missing_parity", "missing_checksum", "different_protocol",
])
def test_invalid_grid_does_not_write_summary(grid, tamper):
    results, folds, near_dup = grid
    run = results / "gda" / "fold0_seed44"
    path = run / "preds.csv.gz"
    if tamper == "missing":
        path.unlink()
    elif tamper == "different_protocol":
        config = yaml.safe_load((run / "config.yaml").read_text())
        config["hw2"]["alpha"] = 1.1
        (run / "config.yaml").write_text(yaml.safe_dump(config))
    elif tamper in ("wrong_metrics", "mixed_checkpoint", "wrong_variant", "missing_parity", "missing_checksum"):
        metrics = json.loads((run / "metrics.json").read_text())
        if tamper == "wrong_metrics":
            metrics["main_score"] = 0.0
        elif tamper == "mixed_checkpoint":
            metrics["checkpoint_sha256"] = "d" * 64
        elif tamper == "wrong_variant":
            metrics["experiment"] = "hw2_p2p"
        elif tamper == "missing_parity":
            del metrics["archive_parity"]
        elif tamper == "missing_checksum":
            del metrics["archive_predictions_sha256"]
        (run / "metrics.json").write_text(json.dumps(metrics))
    else:
        frame = pd.read_csv(path)
        if tamper == "mislabeled":
            frame.loc[0, "slice"] = "wrong"
        else:
            frame.iloc[0] = frame.iloc[1]
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="gda/fold0_seed44"):
        hw2.evaluate(results, folds, near_dup, n_boot=20)
    assert not (results / "summary.json").exists()
