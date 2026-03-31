# src/plot_testing_results.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("src.plot_testing_results")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_testing_output_dirs(run_dir: Path) -> Dict[str, Path]:
    """
    Create and return a dedicated directory structure for one testing run.

    Structure:
        run_dir/
            logs/
            tables/
            plots/
                probability_curves/
                predicted_classes/
                overview/
    """
    logs_dir = run_dir / "logs"
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    probability_dir = plots_dir / "probability_curves"
    predicted_class_dir = plots_dir / "predicted_classes"
    overview_dir = plots_dir / "overview"

    for path in [logs_dir, tables_dir, plots_dir, probability_dir, predicted_class_dir, overview_dir]:
        ensure_dir(path)

    return {
        "logs_dir": logs_dir,
        "tables_dir": tables_dir,
        "plots_dir": plots_dir,
        "probability_dir": probability_dir,
        "predicted_class_dir": predicted_class_dir,
        "overview_dir": overview_dir,
    }


def _sorted_probability_columns(df: pd.DataFrame) -> List[str]:
    return sorted([col for col in df.columns if col.startswith("prob_")])


def _extract_class_names_from_prob_cols(prob_cols: List[str]) -> List[str]:
    return [col.replace("prob_", "", 1) for col in prob_cols]


def _safe_series_id(series_id: str) -> str:
    return str(series_id).replace("/", "_").replace("\\", "_").replace(" ", "_")


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

    class_names = _extract_class_names_from_prob_cols(prob_cols)

    plt.figure(figsize=(10, 5))

    for prob_col, class_name in zip(prob_cols, class_names):
        plt.plot(series_df[x_col].values, series_df[prob_col].values, label=class_name)

    if transition_index is not None:
        plt.axvline(x=transition_index, linestyle="--", linewidth=1.5, label="transition")

    plt.xlabel(x_col)
    plt.ylabel("probability")
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

    y_values = series_df[y_col].astype(int).values

    if class_names is None:
        prob_cols = _sorted_probability_columns(series_df)
        class_names = _extract_class_names_from_prob_cols(prob_cols) if prob_cols else []

    plt.figure(figsize=(10, 4))
    plt.step(series_df[x_col].values, y_values, where="post")

    if transition_index is not None:
        plt.axvline(x=transition_index, linestyle="--", linewidth=1.5, label="transition")

    plt.xlabel(x_col)
    plt.ylabel("predicted class")
    plt.title(title or "Predicted class over time")

    if class_names and len(class_names) > 0:
        plt.yticks(list(range(len(class_names))), class_names)

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
) -> None:
    """
    Optional combined plot if signal snapshots were saved in the results.
    This will only work if test.py stores signal points into the progressive CSV.
    """
    if series_df.empty:
        LOGGER.warning("Skipping signal/prediction overview because series_df is empty: %s", save_path)
        return

    if signal_col not in series_df.columns or x_signal_col not in series_df.columns:
        LOGGER.warning("Skipping signal/prediction overview because signal columns are missing: %s", save_path)
        return

    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(series_df[x_signal_col].values, series_df[signal_col].values)
    ax1.set_title(title or "Signal and predictions")
    ax1.set_xlabel("signal index")
    ax1.set_ylabel("signal")

    if pred_col in series_df.columns and x_pred_col in series_df.columns:
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.step(series_df[x_pred_col].values, series_df[pred_col].astype(int).values, where="post")
        ax2.set_xlabel("reveal index")
        ax2.set_ylabel("predicted class")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


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
    """
    output_dirs = get_testing_output_dirs(run_dir)

    if not progressive_csv_path.exists():
        raise FileNotFoundError(f"Progressive prediction CSV not found: {progressive_csv_path}")

    df = pd.read_csv(progressive_csv_path)
    if df.empty:
        LOGGER.warning("Progressive prediction CSV is empty: %s", progressive_csv_path)
        return {"num_series": 0, "num_probability_plots": 0, "num_predicted_class_plots": 0}

    if "series_id" not in df.columns:
        raise ValueError("Column 'series_id' not found in progressive prediction CSV.")

    num_probability_plots = 0
    num_predicted_class_plots = 0

    grouped = df.groupby("series_id", sort=True)

    for series_id, series_df in grouped:
        series_df = series_df.sort_values(x_col).reset_index(drop=True)
        safe_id = _safe_series_id(str(series_id))

        transition_index = None
        if "transition_index" in series_df.columns:
            valid_vals = series_df["transition_index"].dropna().values
            if len(valid_vals) > 0:
                transition_index = float(valid_vals[0])

        prob_save_path = output_dirs["probability_dir"] / f"{safe_id}.png"
        pred_save_path = output_dirs["predicted_class_dir"] / f"{safe_id}.png"

        plot_series_probability_curves(
            series_df=series_df,
            save_path=prob_save_path,
            title=f"Series {series_id} - class probabilities",
            transition_index=transition_index,
            x_col=x_col,
        )
        num_probability_plots += 1

        plot_series_predicted_classes(
            series_df=series_df,
            save_path=pred_save_path,
            class_names=class_names,
            title=f"Series {series_id} - predicted classes",
            transition_index=transition_index,
            x_col=x_col,
            y_col="y_pred",
        )
        num_predicted_class_plots += 1

    return {
        "num_series": int(df["series_id"].nunique()),
        "num_probability_plots": num_probability_plots,
        "num_predicted_class_plots": num_predicted_class_plots,
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

    counts = df[pred_col].astype(int).value_counts().sort_index()

    x_vals = counts.index.tolist()
    y_vals = counts.values.tolist()

    plt.figure(figsize=(8, 4))
    plt.bar(x_vals, y_vals)

    plt.xlabel("predicted class")
    plt.ylabel("count")
    plt.title("Final predicted-class distribution across series")

    if class_names is not None and len(class_names) > 0:
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
    Save simple aggregated tables into the dedicated tables/ folder.
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
        "Testing plots completed | series=%d | probability_plots=%d | predicted_class_plots=%d",
        summary["num_series"],
        summary["num_probability_plots"],
        summary["num_predicted_class_plots"],
    )

    return summary
