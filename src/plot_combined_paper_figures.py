from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("src.plot_testing_results")


# ---------------------------------------------------------------------
# filesystem helpers
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_testing_output_dirs(run_dir: Path) -> Dict[str, Path]:
    """
    Create and return the standard output structure for one DL testing run.

    Structure
    ---------
    run_dir/
        logs/
        tables/
        plots/
            probability_curves/
            predicted_classes/
            signal_overview/
            overview/
    """
    logs_dir = run_dir / "logs"
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    probability_dir = plots_dir / "probability_curves"
    predicted_class_dir = plots_dir / "predicted_classes"
    signal_overview_dir = plots_dir / "signal_overview"
    overview_dir = plots_dir / "overview"

    for path in [
        logs_dir,
        tables_dir,
        plots_dir,
        probability_dir,
        predicted_class_dir,
        signal_overview_dir,
        overview_dir,
    ]:
        ensure_dir(path)

    return {
        "logs_dir": logs_dir,
        "tables_dir": tables_dir,
        "plots_dir": plots_dir,
        "probability_dir": probability_dir,
        "predicted_class_dir": predicted_class_dir,
        "signal_overview_dir": signal_overview_dir,
        "overview_dir": overview_dir,
    }


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------
def _safe_series_id(series_id: str) -> str:
    return str(series_id).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _sorted_probability_columns(df: pd.DataFrame) -> List[str]:
    return sorted([col for col in df.columns if col.startswith("prob_")])


def _extract_class_names_from_prob_cols(prob_cols: List[str]) -> List[str]:
    return [col.replace("prob_", "", 1) for col in prob_cols]


def _resolve_transition_index(series_df: pd.DataFrame) -> Optional[float]:
    if "transition_index" not in series_df.columns:
        return None

    vals = pd.to_numeric(series_df["transition_index"], errors="coerce").dropna().values
    if len(vals) == 0:
        return None
    return float(vals[0])


def _resolve_x_column(df: pd.DataFrame, requested_x_col: str) -> str:
    if requested_x_col in df.columns:
        return requested_x_col

    fallback_candidates = [
        "reveal_index",
        "prefix_length",
        "step_idx",
    ]
    for col in fallback_candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find x-axis column. Requested '{requested_x_col}', "
        f"available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------
# plot functions
# ---------------------------------------------------------------------
def plot_series_probability_curves(
    series_df: pd.DataFrame,
    save_path: Path,
    title: Optional[str] = None,
    transition_index: Optional[float] = None,
    x_col: str = "reveal_index",
) -> None:
    """
    Plot class probabilities over progressive reveal steps for one series.
    """
    if series_df.empty:
        LOGGER.warning("Skipping probability plot because series_df is empty: %s", save_path)
        return

    prob_cols = _sorted_probability_columns(series_df)
    if not prob_cols:
        LOGGER.warning("Skipping probability plot because no probability columns were found: %s", save_path)
        return

    x_col = _resolve_x_column(series_df, x_col)
    class_names = _extract_class_names_from_prob_cols(prob_cols)

    plt.figure(figsize=(10, 5))

    for prob_col, class_name in zip(prob_cols, class_names):
        y = pd.to_numeric(series_df[prob_col], errors="coerce").values
        x = pd.to_numeric(series_df[x_col], errors="coerce").values
        plt.plot(x, y, label=class_name)

    if transition_index is not None:
        plt.axvline(x=transition_index, linestyle="--", linewidth=1.5, label="transition")

    plt.xlabel("Local reveal index" if x_col == "reveal_index" else x_col)
    plt.ylabel("Probability")
    plt.ylim(-0.02, 1.02)
    plt.title(title or "Progressive class probabilities")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_series_predicted_classes(
    series_df: pd.DataFrame,
    save_path: Path,
    class_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    transition_index: Optional[float] = None,
    x_col: str = "reveal_index",
    y_col: str = "y_pred",
) -> None:
    """
    Plot predicted class index over progressive reveal steps for one series.
    """
    if series_df.empty:
        LOGGER.warning("Skipping predicted-class plot because series_df is empty: %s", save_path)
        return

    if y_col not in series_df.columns:
        LOGGER.warning("Skipping predicted-class plot because '%s' column is missing: %s", y_col, save_path)
        return

    x_col = _resolve_x_column(series_df, x_col)

    if class_names is None:
        prob_cols = _sorted_probability_columns(series_df)
        class_names = _extract_class_names_from_prob_cols(prob_cols) if prob_cols else []

    y_values = pd.to_numeric(series_df[y_col], errors="coerce").fillna(-1).astype(int).values
    x_values = pd.to_numeric(series_df[x_col], errors="coerce").values

    plt.figure(figsize=(10, 4))
    plt.step(x_values, y_values, where="post")

    if transition_index is not None:
        plt.axvline(x=transition_index, linestyle="--", linewidth=1.5, label="transition")

    plt.xlabel("Local reveal index" if x_col == "reveal_index" else x_col)
    plt.ylabel("Predicted class")
    plt.title(title or "Predicted class over time")

    if class_names:
        valid_ticks = list(range(len(class_names)))
        plt.yticks(valid_ticks, class_names)

    if transition_index is not None:
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_series_signal_with_prediction(
    series_df: pd.DataFrame,
    save_path: Path,
    title: Optional[str] = None,
    signal_col: str = "signal_value",
    x_signal_col: str = "signal_index",
    x_pred_col: str = "reveal_index",
    pred_col: str = "y_pred",
    transition_index: Optional[float] = None,
) -> None:
    """
    Optional combined plot if signal snapshots were saved into the progressive CSV.

    This is only used when the progressive CSV contains:
      - signal_value
      - signal_index
      - y_pred
      - reveal_index
    """
    if series_df.empty:
        LOGGER.warning("Skipping signal/prediction overview because series_df is empty: %s", save_path)
        return

    if signal_col not in series_df.columns or x_signal_col not in series_df.columns:
        LOGGER.warning("Skipping signal/prediction overview because signal columns are missing: %s", save_path)
        return

    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(
        pd.to_numeric(series_df[x_signal_col], errors="coerce").values,
        pd.to_numeric(series_df[signal_col], errors="coerce").values,
    )
    ax1.set_title(title or "Signal and predictions")
    ax1.set_xlabel("Signal index")
    ax1.set_ylabel("Signal")

    if transition_index is not None:
        ax1.axvline(x=transition_index, linestyle="--", linewidth=1.2)

    if pred_col in series_df.columns and x_pred_col in series_df.columns:
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.step(
            pd.to_numeric(series_df[x_pred_col], errors="coerce").values,
            pd.to_numeric(series_df[pred_col], errors="coerce").fillna(-1).astype(int).values,
            where="post",
        )
        ax2.set_xlabel("Local reveal index" if x_pred_col == "reveal_index" else x_pred_col)
        ax2.set_ylabel("Predicted class")

        if transition_index is not None:
            ax2.axvline(x=transition_index, linestyle="--", linewidth=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# batch plotting from CSVs
# ---------------------------------------------------------------------
def plot_all_series_from_progressive_csv(
    progressive_csv_path: Path,
    run_dir: Path,
    class_names: Optional[List[str]] = None,
    x_col: str = "reveal_index",
) -> Dict[str, int]:
    """
    Read the combined progressive prediction CSV and generate:
      - one probability plot per series
      - one predicted-class plot per series
      - optional signal overview plot per series if signal columns exist
    """
    output_dirs = get_testing_output_dirs(run_dir)

    if not progressive_csv_path.exists():
        raise FileNotFoundError(f"Progressive prediction CSV not found: {progressive_csv_path}")

    df = pd.read_csv(progressive_csv_path)
    if df.empty:
        LOGGER.warning("Progressive prediction CSV is empty: %s", progressive_csv_path)
        return {
            "num_series": 0,
            "num_probability_plots": 0,
            "num_predicted_class_plots": 0,
            "num_signal_overview_plots": 0,
        }

    if "series_id" not in df.columns:
        raise ValueError("Column 'series_id' not found in progressive prediction CSV.")

    num_probability_plots = 0
    num_predicted_class_plots = 0
    num_signal_overview_plots = 0

    for series_id, series_df in df.groupby("series_id", sort=True):
        try:
            resolved_x_col = _resolve_x_column(series_df, x_col)
        except ValueError as exc:
            LOGGER.warning("Skipping series_id=%s because x-axis could not be resolved: %s", series_id, exc)
            continue

        series_df = series_df.sort_values(resolved_x_col).reset_index(drop=True)
        safe_id = _safe_series_id(str(series_id))
        transition_index = _resolve_transition_index(series_df)

        prob_save_path = output_dirs["probability_dir"] / f"{safe_id}.png"
        pred_save_path = output_dirs["predicted_class_dir"] / f"{safe_id}.png"
        signal_save_path = output_dirs["signal_overview_dir"] / f"{safe_id}.png"

        plot_series_probability_curves(
            series_df=series_df,
            save_path=prob_save_path,
            title=f"Series {series_id} - class probabilities",
            transition_index=transition_index,
            x_col=resolved_x_col,
        )
        num_probability_plots += 1

        plot_series_predicted_classes(
            series_df=series_df,
            save_path=pred_save_path,
            class_names=class_names,
            title=f"Series {series_id} - predicted classes",
            transition_index=transition_index,
            x_col=resolved_x_col,
            y_col="y_pred",
        )
        num_predicted_class_plots += 1

        if "signal_value" in series_df.columns and "signal_index" in series_df.columns:
            plot_series_signal_with_prediction(
                series_df=series_df,
                save_path=signal_save_path,
                title=f"Series {series_id} - signal and predictions",
                signal_col="signal_value",
                x_signal_col="signal_index",
                x_pred_col=resolved_x_col,
                pred_col="y_pred",
                transition_index=transition_index,
            )
            num_signal_overview_plots += 1

    return {
        "num_series": int(df["series_id"].nunique()),
        "num_probability_plots": num_probability_plots,
        "num_predicted_class_plots": num_predicted_class_plots,
        "num_signal_overview_plots": num_signal_overview_plots,
    }


def plot_prediction_distribution_bar(
    final_predictions_csv: Path,
    save_path: Path,
    class_names: Optional[List[str]] = None,
    pred_col: str = "y_pred",
) -> None:
    """
    Plot final predicted-class distribution across all series.
    """
    if not final_predictions_csv.exists():
        LOGGER.warning("Final predictions CSV not found: %s", final_predictions_csv)
        return

    df = pd.read_csv(final_predictions_csv)
    if df.empty or pred_col not in df.columns:
        LOGGER.warning("Skipping final prediction distribution plot because data is missing: %s", final_predictions_csv)
        return

    counts = pd.to_numeric(df[pred_col], errors="coerce").dropna().astype(int).value_counts().sort_index()
    x_vals = counts.index.tolist()
    y_vals = counts.values.tolist()

    plt.figure(figsize=(8, 4))
    plt.bar(x_vals, y_vals)

    plt.xlabel("Predicted class")
    plt.ylabel("Count")
    plt.title("Final predicted-class distribution across series")

    if class_names:
        tick_labels = []
        for idx in x_vals:
            if 0 <= idx < len(class_names):
                tick_labels.append(class_names[idx])
            else:
                tick_labels.append(str(idx))
        plt.xticks(x_vals, tick_labels)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_basic_testing_tables(
    progressive_csv_path: Path,
    final_predictions_csv: Path,
    run_dir: Path,
) -> None:
    """
    Save a few simple tables into tables/.
    """
    output_dirs = get_testing_output_dirs(run_dir)

    if progressive_csv_path.exists():
        df_prog = pd.read_csv(progressive_csv_path)
        if not df_prog.empty and "series_id" in df_prog.columns:
            counts_df = (
                df_prog.groupby("series_id")
                .size()
                .reset_index(name="num_progressive_steps")
                .sort_values("series_id")
                .reset_index(drop=True)
            )
            counts_df.to_csv(output_dirs["tables_dir"] / "progressive_steps_per_series.csv", index=False)

    if final_predictions_csv.exists():
        df_final = pd.read_csv(final_predictions_csv)
        if not df_final.empty:
            df_final.to_csv(output_dirs["tables_dir"] / "final_series_predictions_copy.csv", index=False)


def run_all_testing_plots(
    run_dir: Path,
    class_names: Optional[List[str]] = None,
    progressive_csv_name: str = "all_series_progressive_predictions.csv",
    final_predictions_csv_name: str = "final_series_predictions.csv",
) -> Dict[str, int]:
    """
    Main entry point to be called from testing/test.py after inference finishes.
    """
    output_dirs = get_testing_output_dirs(run_dir)

    progressive_csv_path = run_dir / progressive_csv_name
    final_predictions_csv = run_dir / final_predictions_csv_name

    summary = plot_all_series_from_progressive_csv(
        progressive_csv_path=progressive_csv_path,
        run_dir=run_dir,
        class_names=class_names,
        x_col="reveal_index",
    )

    plot_prediction_distribution_bar(
        final_predictions_csv=final_predictions_csv,
        save_path=output_dirs["overview_dir"] / "final_predicted_class_distribution.png",
        class_names=class_names,
        pred_col="y_pred",
    )

    save_basic_testing_tables(
        progressive_csv_path=progressive_csv_path,
        final_predictions_csv=final_predictions_csv,
        run_dir=run_dir,
    )

    LOGGER.info(
        "Testing plots completed | series=%d | probability_plots=%d | predicted_class_plots=%d | signal_overview_plots=%d",
        summary["num_series"],
        summary["num_probability_plots"],
        summary["num_predicted_class_plots"],
        summary["num_signal_overview_plots"],
    )

    return summary
