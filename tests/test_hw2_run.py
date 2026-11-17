import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn

from bengali import hw2_run as runner
from bengali.train import predict
from bengali.utils import COMPONENTS


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential()
        self.backbone.add_module("layer3", nn.Conv2d(1, 3, 1))
        self.backbone.add_module("layer4", nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 2, 1)))
        self.heads = nn.ModuleDict({c: nn.Linear(2, 2) for c in COMPONENTS})

    def forward(self, images):
        features = self.backbone(images.float().unsqueeze(1) / 255).mean(dim=(2, 3))
        return {c: head(features) for c, head in self.heads.items()}


def test_run_filters_only_configured_pairs():
    cfg = {"runs": [{"fold": 0, "seed": 42}, {"fold": 1, "seed": 42}, {"fold": 0, "seed": 43}]}
    assert runner.selected_runs(cfg, [0, 1], [43]) == [(0, 43)]
    assert runner.selected_runs(cfg, [0]) == [(0, 42), (0, 43)]
    with pytest.raises(ValueError, match="No configured run"):
        runner.selected_runs(cfg, [1], [43])


def test_capture_matches_backbone_and_removes_hooks():
    model = TinyNet().eval()
    images = np.arange(80, dtype=np.uint8).reshape(5, 4, 4)
    idx = np.array([3, 0, 2])
    stages = ["layer3", "layer4"]
    feats, logits = runner.collect(model, images, idx, stages, "cpu", 2)
    with torch.inference_mode():
        x = torch.from_numpy(images[idx])
        expected_logits = model(x)
        x = x.float().unsqueeze(1) / 255
        for name in stages:
            module = model.backbone.get_submodule(name)
            x = module(x)
            np.testing.assert_allclose(feats[name], x.mean((2, 3)).numpy(), atol=1e-7)
            assert not module._forward_hooks
        for c in COMPONENTS:
            np.testing.assert_allclose(logits[c], expected_logits[c].numpy(), atol=1e-7)


def test_archive_parity_rejects_missing_rows_and_class_drift(tmp_path):
    archive = pd.DataFrame({"image_id": ["a", "b"], "slice": ["id", "ood"]})
    for c in COMPONENTS:
        archive[f"pred_{c}"] = [0, 1]
        archive[f"conf_{c}"] = [0.75, 0.8]
    path = tmp_path / "preds.csv.gz"
    archive.to_csv(path, index=False)
    current = archive.iloc[::-1].copy()
    current.loc[0, "conf_grapheme_root"] += 0.0001
    result = runner.archive_parity(current, path)
    assert result["max_abs_rounded_confidence_difference"]["grapheme_root"] == pytest.approx(0.0001)
    with pytest.raises(ValueError, match="keys disagree"):
        runner.archive_parity(current.iloc[:1], path)
    current.loc[0, "pred_grapheme_root"] = 1
    with pytest.raises(ValueError, match="archive mismatch"):
        runner.archive_parity(current, path)


def test_whole_run_fits_train_only_and_saves_primary_and_control_cells(tmp_path, monkeypatch):
    torch.manual_seed(42)
    images = np.arange(9 * 16, dtype=np.uint8).reshape(9, 4, 4)
    ids = [f"img{i}" for i in range(len(images))]
    folds = pd.DataFrame({"image_id": ids, "grapheme": ["a", "b"] * 4 + ["b"],
                          "id_fold": [1, 1, 1, 1, 1, 0, 0, 1, 1],
                          "ood_group": [-1] * 7 + [0, 0]})
    for i, c in enumerate(COMPONENTS):
        folds[c] = (np.arange(len(folds)) + i) % 2
    paths = {key: tmp_path / key for key in ("processed", "archive", "weights", "results")}
    paths["processed"].mkdir()
    idx = np.array([5, 7, 0, 1, 8, 4, 2, 6, 3])
    np.save(paths["processed"] / "train_4.npy", images[idx])
    pd.DataFrame({"image_id": np.array(ids)[idx]}).to_csv(paths["processed"] / "train_4_index.csv", index=False)
    folds_path = tmp_path / "folds.csv"
    folds.to_csv(folds_path, index=False)
    archive = paths["archive"] / "fold0_seed42"
    archive.mkdir(parents=True)
    checkpoint = paths["weights"] / "fold0_seed42.pt"
    checkpoint.parent.mkdir()
    model = TinyNet()
    torch.save(model.state_dict(), checkpoint)
    # Early B2 configs omit run metadata; metrics.json identifies the checkpoint.
    base = {"model": {"backbone": "tiny"}, "data": {"img_size": 4}}
    (archive / "metrics.json").write_text(json.dumps({"fold": 0, "seed": 42}))
    (archive / "config.yaml").write_text(yaml.safe_dump(base))
    predictions = []
    for part, rows in (("id", [5, 6]), ("ood", [7, 8])):
        pred, conf = predict(model, images, np.array(rows), torch.device("cpu"), batch_size=2, amp=None)
        frame = pd.DataFrame({"image_id": np.array(ids)[rows], "slice": part})
        for c in COMPONENTS:
            frame[f"pred_{c}"] = pred[c]
            frame[f"conf_{c}"] = np.round(conf[c], 4)
        predictions.append(frame)
    archived = pd.concat(predictions)
    archived.to_csv(archive / "preds.csv.gz", index=False)

    def tiny_factory(backbone, pretrained, dropout):
        assert (backbone, pretrained, dropout) == ("tiny", False, 0.0)
        return TinyNet()

    original_fit = runner.fit_heads

    def fit_checked(stages, logits, train_labels, ridge):
        expected_rows = idx[idx < 5]
        np.testing.assert_array_equal(train_labels, folds.iloc[expected_rows][COMPONENTS].to_numpy())
        assert all(len(v) == len(expected_rows) for v in [*stages.values(), *logits.values()])
        return original_fit(stages, logits, train_labels, ridge)

    monkeypatch.setattr(runner, "FinetuneNet", tiny_factory)
    monkeypatch.setattr(runner, "fit_heads", fit_checked)
    cfg = {"device": "cpu", "experiment": "hw2", "output_dir": str(paths["results"]),
           "data": {"processed_dir": str(paths["processed"]), "img_size": 4, "folds_file": str(folds_path)},
           "eval": {"batch_size": 2},
           "hw2": {"checkpoint_dir": str(paths["weights"]), "archive_dir": str(paths["archive"]),
                   "stages": ["layer3", "layer4"], "ridge": 1.0, "alpha": 1.0, "prior_strength": 1.0}}
    runner.run(cfg, 0, 42)
    for cell in runner.CELLS + runner.CONTROLS:
        cell_dir = paths["results"] / ("controls" if cell in runner.CONTROLS else "") / cell
        result_dir = cell_dir / "fold0_seed42"
        metrics = json.loads((result_dir / "metrics.json").read_text())
        assert metrics["checkpoint_sha256"] == runner.file_hash(checkpoint)
        assert metrics["row_counts"] == {"train": 5, "val_id": 2, "val_ood": 2}
        assert metrics["archive_parity"]["all_classes_match"]
        assert metrics["id_n"] == metrics["ood_n"] == 2
        saved = pd.read_csv(result_dir / "preds.csv.gz")
        assert set(saved.columns) == set(archived.columns)
        assert set(saved["image_id"]) == set(ids[5:])
        config = yaml.safe_load((result_dir / "config.yaml").read_text())
        assert config["output_dir"] == str(cell_dir)
