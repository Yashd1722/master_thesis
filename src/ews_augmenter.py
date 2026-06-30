"""
src/ews_augmenter.py
Robust EWS Feature Augmenter.
"""
import numpy as np
import pandas as pd

def augment_ews_channels(X, window_frac=0.5):
    if X.ndim == 3:
        X = X.squeeze(1) 
    elif X.ndim != 2:
        raise ValueError(f"Expected 2D or 3D array, got {X.ndim}D")

    n_samples, n_timepoints = X.shape
    window_size = max(3, int(n_timepoints * window_frac))

    df = pd.DataFrame(X.T)
    
    var_ch = df.rolling(window=window_size, min_periods=1, center=True).var().T.values
    skew_ch = df.rolling(window=window_size, min_periods=1, center=True).skew().T.values
    
    df_lag = df.shift(1)
    df_lag.iloc[0] = df.iloc[0] 
    
    roll = lambda d: d.rolling(window=window_size, min_periods=1, center=True)
    mean_x = roll(df).mean()
    mean_y = roll(df_lag).mean()
    var_x = roll(df**2).mean() - mean_x**2
    var_y = roll(df_lag**2).mean() - mean_y**2
    cov_xy = roll(df * df_lag).mean() - (mean_x * mean_y)
    
    var_x = var_x.clip(lower=1e-10)
    var_y = var_y.clip(lower=1e-10)
    ac_ch = (cov_xy / np.sqrt(var_x * var_y)).T.values
    
    var_ch = np.nan_to_num(var_ch, nan=0.0)
    skew_ch = np.nan_to_num(skew_ch, nan=0.0)
    ac_ch = np.nan_to_num(ac_ch, nan=0.0)

    X_aug = np.stack([X, var_ch, ac_ch, skew_ch], axis=1)
    
    # Channel-level jitter to prevent aeon from rejecting constant derived channels
    rng = np.random.default_rng(42)
    X_aug[:, 1:, :] += rng.normal(0, 1e-2, X_aug[:, 1:, :].shape)
    
    return X_aug
