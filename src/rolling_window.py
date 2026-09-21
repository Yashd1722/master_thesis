"""
src/rolling_window.py
Rolling window EWS engine for PANGAEA empirical testing.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from scipy.stats import kendalltau
from typing import List, Optional

logger = logging.getLogger(__name__)

# ponytail: redox proxies only — the anoxia bifurcation is a redox transition, so
# Mo and U are the physically meaningful channels (Bury 2021 anoxia used the same
# pair). Al/Ba/Ti are detrital/productivity proxies and only added noise to the
# type-classification consensus. Restore the full list if you need detrital EWS.
ELEMENTS        = ["Mo", "U"]
PRIMARY_ELEMENT = "Mo"


@dataclass
class RollingWindowResult:
    positions:     np.ndarray
    ages_kyr_bp:   np.ndarray
    variance:      np.ndarray
    lag1_ac:       np.ndarray
    ktau_variance: float
    ktau_lag1_ac:  float
    dl_inputs:     List[np.ndarray] = field(default_factory=list)
    dl_probs:      Optional[np.ndarray] = None
    p_transition:  Optional[np.ndarray] = None
    core_name:     str = ""
    sapropel_id:   str = ""
    element:       str = "Mo"
    segment_type:  str = "forced"
    n_series:      int = 0


def _variance(series: np.ndarray) -> float:
    return float(np.var(series, ddof=1)) if len(series) >= 2 else 0.0


def _lag1_ac(series: np.ndarray) -> float:
    if len(series) < 3:
        return 0.0
    x, y = series[:-1], series[1:]
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def prepare_dl_input(residuals: np.ndarray, position: int,
                     ts_len: int, pad_mode: str = "zero") -> np.ndarray:
    """Right-aligned, normalised, left-padded fixed window ending at `position`.

    Delegates to src.data_common.make_fixed_window so training and inference
    share one definition of the transform (normalise to mean|x|==1, then
    left-pad to ts_len). This is the exact shape the model is now trained on
    via Bury-style left-censoring.
    """
    from src.data_common import make_fixed_window
    return make_fixed_window(residuals, position, ts_len, pad_mode=pad_mode)


def compute_rolling_ews(
    residuals:    np.ndarray,
    ages_kyr_bp:  np.ndarray,
    element:      str,
    core_name:    str,
    sapropel_id:  str,
    segment_type: str,
    cfg:          dict,
    ts_len:       int,
    pad_mode:     str = "zero",
) -> RollingWindowResult:
    """Compute classical EWS indicators over a sliding window for ONE record.

    Also prepares the DL input sequence (normalised, left-padded to `ts_len`)
    for each window position, so prediction can run perfectly in sync.
    """
    inf_cfg = cfg["inference"]
    n       = len(residuals)
    win_frac = cfg.get("pangaea", {}).get("rolling_window_frac") or inf_cfg["rolling_window_frac"]
    win     = max(10, int(win_frac * n))

    n_steps   = inf_cfg["prediction_steps"]
    start_pos = win
    end_pos   = n

    positions = np.clip(
        np.linspace(start_pos, end_pos, n_steps, dtype=int), win, n
    )

    variances = np.zeros(n_steps)
    ac1s      = np.zeros(n_steps)
    step_ages = np.zeros(n_steps)
    dl_inputs = []

    for i, pos in enumerate(positions):
        win_seg       = residuals[pos - win: pos]
        variances[i]  = _variance(win_seg)
        ac1s[i]       = _lag1_ac(win_seg)
        step_ages[i]  = ages_kyr_bp[pos - 1] if pos - 1 < len(ages_kyr_bp) else np.nan
        dl_inputs.append(prepare_dl_input(residuals, pos, ts_len, pad_mode=pad_mode))

    step_idx     = np.arange(n_steps)
    ktau_var, _  = kendalltau(step_idx, variances)
    ktau_ac, _   = kendalltau(step_idx, ac1s)

    return RollingWindowResult(
        positions      = positions,
        ages_kyr_bp    = step_ages,
        variance       = variances,
        lag1_ac        = ac1s,
        ktau_variance  = float(ktau_var),
        ktau_lag1_ac   = float(ktau_ac),
        dl_inputs      = dl_inputs,
        core_name      = core_name,
        sapropel_id    = sapropel_id,
        element        = element,
        segment_type   = segment_type,
        n_series       = n,
    )
