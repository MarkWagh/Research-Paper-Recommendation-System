"""Shared helpers: config loading with CLI overrides, seeding, logging, devices."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- logging
_FMT = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
_FILE_HANDLERS: list[logging.Handler] = []
_LOGGERS: list[logging.Logger] = []


def get_logger(name: str = "recsys", log_file: str | Path | None = None) -> logging.Logger:
    """Timestamped stdout logger. Passing ``log_file`` also mirrors every project logger
    (existing and future) into that file."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(_FMT)
        logger.addHandler(h)
        logger.propagate = False
        for fh in _FILE_HANDLERS:
            logger.addHandler(fh)
        _LOGGERS.append(logger)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(_FMT)
        _FILE_HANDLERS.append(fh)
        for lg in _LOGGERS:
            lg.addHandler(fh)
    return logger


class Timer:
    def __init__(self, logger: logging.Logger, msg: str):
        self.logger, self.msg = logger, msg

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info(f"{self.msg} ...")
        return self

    def __exit__(self, *exc):
        self.logger.info(f"{self.msg} done in {time.time() - self.t0:.1f}s")


# ---------------------------------------------------------------- seeding / device
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(pref: str = "auto") -> torch.device:
    """`auto` picks cuda > mps > cpu. MPS lacks some sparse ops, so callers that
    need sparse kernels should request `cpu` explicitly."""
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- config
def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def load_config(paths: list[str | Path] | None = None, overrides: list[str] | None = None) -> dict:
    """Merge configs/default.yaml, then each extra YAML, then `a.b=value` overrides
    (values parsed as YAML, so `x=0.5`, `x=[1,2]`, `x=null` all work)."""
    cfg: dict = {}
    for p in [ROOT / "configs" / "default.yaml", *(paths or [])]:
        with open(p) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must look like key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k.strip(), yaml.safe_load(v))
    return cfg


def base_argparser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", nargs="*", default=[], help="extra YAML configs merged over configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides", help="dotted overrides, e.g. split.history_frac=0.25")
    return ap


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def save_json(obj: Any, path: str | Path) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
