import torch


def add_realistic_trend(x, strength=1.0):
    """
    Add a climate-like nonlinear trend to a batch of sequences.

    Expected x shape:
        (batch, seq_len, features)
    """
    t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)

    a = 0.0003 * strength
    b = 0.0000008 * strength
    c = 0.05 * strength
    p_long = 120.0

    trend = a * t + b * (t ** 2) + c * torch.sin(2 * torch.pi * t / p_long)
    trend = trend.view(1, -1, 1)  # (1, seq_len, 1)

    return x + trend


def add_realistic_seasonality(x, amp=0.4, period=50):
    """
    Add multi-harmonic seasonality to a batch of sequences.

    Expected x shape:
        (batch, seq_len, features)
    """
    t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
    period = float(period)

    season = (
        amp * torch.sin(2 * torch.pi * t / period)
        + 0.5 * amp * torch.sin(4 * torch.pi * t / period)
    )

    noise = torch.randn_like(season) * 0.02
    season = (season + noise).view(1, -1, 1)  # (1, seq_len, 1)

    return x + season


def apply_forcing(
    x,
    trend=False,
    seasonality=False,
    trend_strength=1.0,
    season_amp=0.4,
    season_period=50,
):
    if trend:
        x = add_realistic_trend(x, trend_strength)

    if seasonality:
        x = add_realistic_seasonality(x, season_amp, season_period)

    return x
