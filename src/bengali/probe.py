"""Linear probe on frozen features (raw pixels / DINOv3 / MobileCLIP2).

Step 1 (once per encoder): extract features for all images and cache them.
  The encoder is frozen and applied per image, nothing is fitted on the data,
  so extracting features for train and validation images together is leak-free.
Step 2 (per fold): standardise features with TRAIN statistics, train three
  linear heads (multinomial logistic regression) with AdamW for a fixed number
  of epochs, predict the two validation slices.

Usage:
    python -m bengali.probe --config configs/probe_dinov3.yaml
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from .models import MultiHead, build_encoder, count_params, standardize
from .runs import load_fold_data, prediction_frame, save_run
from .utils import COMPONENTS, Timer, get_device, load_config, resolve, save_json, seed_everything


def feature_path(cfg: dict):
    e = cfg["encoder"]
    tag = f"{e['encoder']}_{e.get('model_name', '')}_{e.get('input_size')}_inv{int(e.get('invert', False))}".replace("/", "-")
    sub = cfg["data"].get("subset")
    return resolve("data/features") / f"{tag}{'_sub' + str(sub) if sub else ''}.npy"


@torch.no_grad()
def extract(cfg: dict, imgs: np.ndarray, rows: np.ndarray, device) -> np.ndarray:
    enc = build_encoder(cfg["encoder"]).to(device).eval()
    bs = cfg["encoder"].get("batch_size", 256)
    out = np.zeros((len(imgs), enc.dim), dtype=np.float16)
    t = Timer()
    for s in range(0, len(rows), bs):
        b = rows[s : s + bs]
        x = torch.from_numpy(imgs[b]).to(device)
        out[b] = enc(x).float().cpu().numpy().astype(np.float16)
        if s // bs % 100 == 0:
            print(f"  features {s + len(b)}/{len(rows)}  {t():.0f}s", flush=True)
    return out, t(), getattr(enc, "n_params", None) or count_params(enc)


def get_features(cfg: dict, imgs: np.ndarray, ix: dict, device):
    path = feature_path(cfg)
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        from .utils import load_json

        feats, meta = np.load(path), load_json(meta_path)
        assert feats.shape[0] == len(imgs), f"stale feature cache {path}: {feats.shape[0]} rows vs {len(imgs)} images"
        return feats, meta
    if cfg["data"].get("subset"):
        # smoke runs: extract the union of the subset rows of *all* requested folds (cache is shared across folds)
        from .runs import load_fold_data

        rows = np.unique(np.concatenate([np.concatenate(list(load_fold_data(cfg, k, in_memory=False)[3].values())) for k in cfg["folds"]]))
    else:
        rows = np.arange(len(imgs))
    print(f"extracting {cfg['encoder']} features for {len(rows)} images -> {path}")
    feats, sec, n_params = extract(cfg, imgs, rows, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feats)
    meta = {"extract_sec": sec, "n_params_encoder": n_params, "dim": int(feats.shape[1]), "n_rows": int(len(rows))}
    save_json(meta, meta_path)
    return feats, meta


def train_probe(cfg: dict, feats: np.ndarray, labels: np.ndarray, ix: dict, seed: int, device):
    pcfg = cfg["probe"]
    seed_everything(seed)
    xtr, xvi, xvo = standardize(*(feats[ix[p]].astype(np.float32) for p in ("train", "val_id", "val_ood")))
    xtr_t = torch.from_numpy(xtr).to(device)
    ytr = torch.from_numpy(labels[ix["train"]]).to(device)
    head = MultiHead(xtr.shape[1], pcfg.get("dropout", 0.0)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=pcfg["lr"], weight_decay=pcfg["weight_decay"])
    bs = pcfg["batch_size"]
    steps = pcfg["epochs"] * math.ceil(len(xtr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=pcfg["lr"], total_steps=steps, pct_start=0.1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    hist = []
    for ep in range(pcfg["epochs"]):
        perm = torch.randperm(len(xtr), generator=g).to(device)
        tot = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s : s + bs]
            out = head(xtr_t[b])
            loss = sum(F.cross_entropy(out[c], ytr[b, j]) for j, c in enumerate(COMPONENTS))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        hist.append({"epoch": ep + 1, "train_loss": tot / len(xtr)})
    head.eval()
    preds, confs = {}, {}
    with torch.no_grad():
        for part, x in (("val_id", xvi), ("val_ood", xvo)):
            out = head(torch.from_numpy(x).to(device))
            preds[part], confs[part] = {}, {}
            for c in COMPONENTS:
                p = out[c].softmax(1)
                cf, ar = p.max(1)
                preds[part][c], confs[part][c] = ar.cpu().numpy(), cf.cpu().numpy()
    return preds, confs, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, nargs="+", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    device = get_device(cfg.get("device", "auto"))
    feats = meta = None
    for seed in a.seed or cfg.get("seeds", [42]):
        for fold in a.fold if a.fold is not None else cfg.get("folds", [0]):
            imgs, folds, labels, ix = load_fold_data(cfg, fold, in_memory=feats is None)
            if feats is None:
                feats, meta = get_features(cfg, imgs, ix, device)
            t = Timer()
            preds, confs, hist = train_probe(cfg, feats, labels, ix, seed, device)
            pdf = prediction_frame(folds, ix, preds, confs)
            extra = {"probe_sec": t(), **meta, "n_train": int(len(ix["train"]))}
            m = save_run(cfg, fold, seed, pdf, extra)
            from .runs import run_dir

            save_json(hist, run_dir(cfg, fold, seed) / "history.json")
            print(f"[{cfg['experiment']} fold {fold} seed {seed}]", {k: round(v, 4) for k, v in m.items() if k.endswith("score")})


if __name__ == "__main__":
    main()
