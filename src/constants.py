"""
src/constants.py
Single source of truth for class ordering, key indices, and the handful of
tiny stdlib-only helpers (config loading, log setup) shared across training,
evaluation and the testing/ analysis scripts, kept out of src/dataset_loader
so importing them doesn't pull in torch.

Bury et al. (2021) DOI: 10.1073/pnas.2106140118 defines four bifurcation types.
The ordering below is the canonical mapping used throughout this codebase.
"""
import logging
import sys
from pathlib import Path

import yaml

# Class index → bifurcation type, matching Bury's label CSV exactly.
CLASS_NAMES = ["fold", "hopf", "transcritical", "null"]

# Index of the "null" (no-transition) class — used to compute p(transition).
# p(transition) = 1 - probs[:, NULL_IDX]
NULL_IDX = 3


def _clean_dict(d):
    if isinstance(d, dict):
        return {k.strip() if isinstance(k, str) else k: _clean_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_clean_dict(i) for i in d]
    return d


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return _clean_dict(cfg)


def setup_log(log_path: Path, include_level: bool = True) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s" if include_level
        else "%(asctime)s | %(message)s", "%H:%M:%S")
    for h in [logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger
