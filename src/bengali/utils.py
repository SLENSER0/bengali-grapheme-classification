"""Shared constants and small helpers (config loading, seeding, device, IO)."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

COMPONENTS = ["grapheme_root", "vowel_diacritic", "consonant_diacritic"]
SHORT = {"grapheme_root": "root", "vowel_diacritic": "vowel", "consonant_diacritic": "consonant"}
N_CLASSES = {"grapheme_root": 168, "vowel_diacritic": 11, "consonant_diacritic": 7}
# Competition metric weights: grapheme root counts twice.
WEIGHTS = {"grapheme_root": 2.0, "vowel_diacritic": 1.0, "consonant_diacritic": 1.0}

RAW_H, RAW_W = 137, 236


def resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _set_by_dotted_key(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            raise KeyError(f"--set {dotted}: no such section '{k}' in the config")
        node = node[k]
    if keys[-1] not in node:  # catches typos such as train.epoch=1 that would otherwise be silently ignored
        raise KeyError(f"--set {dotted}: unknown key (known: {sorted(node)})")
    node[keys[-1]] = value


def _parse_value(raw: str) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # YAML 1.1 reads '1e-3' as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. Supports `base:` inheritance and `key.sub=value` CLI overrides."""
    path = resolve(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = deep_update(base, cfg)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        _set_by_dotted_key(cfg, key, _parse_value(raw))
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def get_device(pref: str = "auto"):
    import torch

    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj: Any, path: str | Path) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(resolve(path)) as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def __call__(self) -> float:
        return time.perf_counter() - self.t0
