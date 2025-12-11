#!/usr/bin/env python3
"""
scripts/plot_ti_results.py

Visualize trajectory inference (TI) + Waddington-OT benchmarks for
EGGFM and baselines.

Inputs
------
1) TI summary CSV (e.g. ti_ablation_summary6.csv)
   - Contains per-run pseudotime metrics, n_cells, metric_mode, etc.
2) Optional WOT summary CSV (from run_wot_benchmark.py)
   - Contains per-run WOT metrics (columns starting with 'wot_').

Outputs
-------
In --out-dir (default: out/figures):

  - ti_bar_spearman_N{fixed_n}.png
      Bar chart: pt_spearman_vs_time per method at n_cells == fixed_n

  - ti_scatter_spearman_vs_variance.png
      Scatter: pt_spearman_vs_time vs pt_variance, colored by method

  - ti_wot_bar_{metric}.png   (if WOT metrics present)
      Bar chart: chosen WOT metric per method at n_cells == fixed_n

  - ti_wot_scatter_spearman_vs_{metric}.png  (if WOT metrics present)
      Scatter: pt_spearman_vs_time vs chosen WOT metric

  - ti_best_per_method_table.csv / .tex
      Compact best-per-(method,n_cells) table, including any WOT metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def infer_method(row) -> str:
    mm = str(row.get("metric_mode", ""))
    if mm.startswith("baseline_"):
        return mm.replace("baseline_", "")
    if mm in ("", "nan", "None"):
        # Fall back on embedding_key
        ek = str(row.get("embedding_key", ""))
        if ek == "X_pca":
            return "pca"
        if ek == "X_diffmap":
            return "diffmap"
        if ek == "X_scvi":
            return "scvi"
        if ek == "X_eggfm_diffmap":
            return "eggfm_default"
        return "unknown"
    return f"eggfm_{mm}"



def load_ti_csv(metrics_csv: str) -> pd.DataFrame:
    """
    Load TI summary CSV and standardize core columns.
    """
    df = pd.read_csv(metrics_csv)

    # Ensure numeric columns are numeric if present
    num_cols = [
        "eval_n_cells", "max_cells",
        "pt_variance", "pt_min", "pt_max",
        "pt_spearman_vs_time", "pt_kendall_vs_time",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive n_cells from eval_n_cells (preferred) or max_cells
    if "eval_n_cells" in df.columns:
        df["n_cells"] = df["eval_n_cells"]
    else:
        df["n_cells"] = float("nan")
    if "max_cells" in df.columns:
        df["n_cells"] = df["n_cells"].fillna(df["max_cells"])

    df["n_cells"] = pd.to_numeric(df["n_cells"], errors="coerce")
    df["n_cells_int"] = df["n_cells"].round().astype("Int64")

    # Derive "method" column from metric_mode
    df["metric_mode"] = df.get("metric_mode", "")
    df["method"] = df.apply(infer_method, axis=1)

    # Make sure we have run_id as a string (for merging with WOT results)
    if "run_id" in df.columns:
        df["run_id"] = df["run_id"].astype(str)

    # Keep only rows with a valid Spearman metric
    if "pt_spearman_vs_time" in df.columns:
        df = df[~df["pt_spearman_vs_time"].isna()].copy()
    else:
        raise ValueError("TI CSV is missing 'pt_spearman_vs_time' column.")

    return df


def auto_find_wot_csv() -> Optional[Path]:
    """
    Try to auto-detect a WOT summary CSV in common locations.
    """
    candidates = [
        Path("out/metrics/wot/wot_benchmark_summary.csv"),
        Path("out/metrics/wot/wot_summary.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_wot_csv(wot_csv: Path) -> pd.DataFrame:
    """
    Load WOT benchmark CSV and keep any 'wot_*' columns + run_id.
    This is intentionally flexible: any columns starting with 'wot_'
    are treated as numeric metrics.
    """
    df = pd.read_csv(wot_csv)

    if "run_id" not in df.columns:
        # Fallback: allow base_run_id to serve as run_id if present
        if "base_run_id" in df.columns:
            df["run_id"] = df["base_run_id"]
        else:
            raise ValueError(
                f"WOT CSV {wot_csv} has no 'run_id' or 'base_run_id' column; "
                "cannot merge with TI summary."
            )

    df["run_id"] = df["run_id"].astype(str)

    # Coerce all wot_* columns to numeric
    wot_cols = [c for c in df.columns if c.startswith("wot_")]
    for col in wot_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # We only really need run_id + wot_* metrics for merging
    keep_cols = ["run_id"] + wot_cols
    df = df[keep_cols].copy()

    return df


def best_per_method_and_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (method, n_cells_int), keep the run with the highest Spearman.
    Returns the full rows, so run_id, WOT metrics, etc. are preserved.
    """
    idx = (
        df.groupby(["method", "n_cells_int"])["pt_spearman_vs_time"]
        .idxmax()
        .dropna()
    )
    best = df.loc[idx].reset_index(drop=True)
    return best


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def make_bar_chart(best_df: pd.DataFrame, out_dir: Path, fixed_n: int = 15000):
    """
    Bar chart: pt_spearman_vs_time per method at a fixed n_cells (e.g. 15000).
    """
    subset = best_df[best_df["n_cells_int"] == fixed_n].copy()
    if subset.empty:
        print(f"⚠️ No rows with n_cells_int == {fixed_n}, skipping TI bar chart.")
        return

    subset = subset.sort_values("pt_spearman_vs_time", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(subset["method"], subset["pt_spearman_vs_time"])
    ax.set_ylabel("Spearman vs true time")
    ax.set_title(f"Pseudotime Spearman correlation at N = {fixed_n}")
    ax.set_xticklabels(subset["method"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.7)
    fig.tight_layout()

    out_path = out_dir / f"ti_bar_spearman_N{fixed_n}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved TI bar chart to {out_path}")


def make_scatter(best_df: pd.DataFrame, out_dir: Path):
    """
    Scatter: pt_spearman_vs_time vs pt_variance, colored by method.
    One point per (method, n_cells_int) "best run".
    """
    if "pt_variance" not in best_df.columns:
        print("⚠️ 'pt_variance' missing, skipping TI scatter.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for method, sub in best_df.groupby("method"):
        ax.scatter(
            sub["pt_variance"],
            sub["pt_spearman_vs_time"],
            label=method,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xlabel("Pseudotime variance (pt_variance)")
    ax.set_ylabel("Spearman vs true time")
    ax.set_title("TI performance: variance vs temporal correlation")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.7)
    ax.legend(fontsize=7, bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)

    fig.tight_layout()

    out_path = out_dir / "ti_scatter_spearman_vs_variance.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved TI scatter plot to {out_path}")


def choose_wot_metric(best_df: pd.DataFrame) -> Optional[str]:
    """
    Pick a "primary" WOT metric column to plot, if available.
    Priority:
      1. wot_clone_emd
      2. wot_time_emd
      3. wot_clone_negloglik
      4. wot_clone_loglik
      5. any other wot_* column (first in sorted order)
    """
    candidates = [c for c in best_df.columns if c.startswith("wot_")]
    if not candidates:
        return None

    priority = [
        "wot_clone_emd",
        "wot_time_emd",
        "wot_clone_negloglik",
        "wot_clone_loglik",
    ]
    for p in priority:
        if p in candidates:
            return p

    # Fallback: alphabetically first wot_* column
    return sorted(candidates)[0]


def make_wot_bar_chart(
    best_df: pd.DataFrame,
    out_dir: Path,
    fixed_n: int = 15000,
    metric_col: Optional[str] = None,
):
    """
    Bar chart: chosen WOT metric per method at a fixed n_cells.
    Lower is usually better for distance / EMD metrics.
    """
    if metric_col is None:
        metric_col = choose_wot_metric(best_df)

    if metric_col is None or metric_col not in best_df.columns:
        print("⚠️ No WOT metric column found, skipping WOT bar chart.")
        return

    subset = best_df[best_df["n_cells_int"] == fixed_n].copy()
    if subset.empty:
        print(f"⚠️ No rows with n_cells_int == {fixed_n}, skipping WOT bar chart.")
        return

    subset = subset.sort_values(metric_col, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(subset["method"], subset[metric_col])
    ax.set_ylabel(f"{metric_col} (lower is better)")
    ax.set_title(f"Waddington-OT: {metric_col} at N = {fixed_n}")
    ax.set_xticklabels(subset["method"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.7)
    fig.tight_layout()

    out_path = out_dir / f"ti_wot_bar_{metric_col}_N{fixed_n}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved WOT bar chart to {out_path}")


def make_wot_scatter(
    best_df: pd.DataFrame,
    out_dir: Path,
    metric_col: Optional[str] = None,
):
    """
    Scatter: pt_spearman_vs_time vs chosen WOT metric.
    """
    if metric_col is None:
        metric_col = choose_wot_metric(best_df)

    if metric_col is None or metric_col not in best_df.columns:
        print("⚠️ No WOT metric column found, skipping WOT scatter.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for method, sub in best_df.groupby("method"):
        ax.scatter(
            sub[metric_col],
            sub["pt_spearman_vs_time"],
            label=method,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xlabel(f"{metric_col} (usually lower is better)")
    ax.set_ylabel("Spearman vs true time")
    ax.set_title(f"TI vs WOT: Spearman vs {metric_col}")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.7)
    ax.legend(fontsize=7, bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)

    fig.tight_layout()

    out_path = out_dir / f"ti_wot_scatter_spearman_vs_{metric_col}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved WOT scatter plot to {out_path}")


def save_summary_table(best_df: pd.DataFrame, out_dir: Path):
    """
    Save a compact table (CSV + LaTeX) of best runs per (method, n_cells_int),
    including any WOT metrics (columns starting with 'wot_').
    """
    base_cols = [
        "method",
        "n_cells_int",
        "pt_spearman_vs_time",
        "pt_kendall_vs_time",
        "pt_variance",
    ]
    wot_cols = [c for c in best_df.columns if c.startswith("wot_")]

    cols = [c for c in base_cols if c in best_df.columns] + wot_cols
    table = best_df[cols].copy()

    # Round metrics for readability
    for c in table.columns:
        if c not in ("method", "n_cells_int"):
            table[c] = pd.to_numeric(table[c], errors="coerce").round(3)

    csv_path = out_dir / "ti_best_per_method_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"✅ Saved summary table CSV to {csv_path}")

    # LaTeX version for posters/papers
    try:
        latex_path = out_dir / "ti_best_per_method_table.tex"
        latex_str = table.to_latex(index=False)
        latex_path.write_text(latex_str)
        print(f"✅ Saved LaTeX table to {latex_path}")
        print("\nLaTeX snippet (paste into your poster/paper):\n")
        print(latex_str)
    except Exception as e:
        print(f"⚠️ Could not write LaTeX table: {e}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-csv",
        type=str,
        required=True,
        help="Path to the combined TI metrics CSV (e.g. ti_ablation_summary6.csv).",
    )
    parser.add_argument(
        "--wot-csv",
        type=str,
        default="auto",
        help=(
            "Path to WOT benchmark summary CSV (from run_wot_benchmark.py). "
            "If 'auto' (default), will look for common paths under out/metrics/wot/."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out/figures",
        help="Directory to save plots and tables.",
    )
    parser.add_argument(
        "--fixed-n",
        type=int,
        default=15000,
        help="Cell count for the bar charts (default: 15000).",
    )

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    # ----- Load TI metrics -----
    print(f"📥 Loading TI metrics from {args.metrics_csv}")
    df = load_ti_csv(args.metrics_csv)
    print(f"📊 Total TI rows with valid Spearman: {len(df)}")

    # ----- Optionally merge WOT metrics -----
    ti_has_wot = any(c.startswith("wot_") for c in df.columns)
    wot_path: Optional[Path] = None

    if not ti_has_wot:
        if args.wot_csv == "auto":
            wot_path = auto_find_wot_csv()
            if wot_path is None:
                print("ℹ️ No WOT CSV found automatically; proceeding with TI-only plots.")
        else:
            candidate = Path(args.wot_csv)
            if candidate.exists():
                wot_path = candidate
            else:
                print(f"⚠️ Specified WOT CSV {candidate} does not exist; TI-only plots.")

        if wot_path is not None:
            print(f"📥 Loading WOT metrics from {wot_path}")
            wot_df = load_wot_csv(wot_path)
            print(f"📊 WOT rows: {len(wot_df)}")

            if "run_id" in df.columns:
                df = df.merge(wot_df, on="run_id", how="left")
                print("🔗 Merged WOT metrics into TI dataframe on 'run_id'.")
            else:
                print("⚠️ TI CSV has no 'run_id'; cannot merge WOT metrics.")

    # ----- Best per (method, n_cells) -----
    best_df = best_per_method_and_cells(df)
    print(f"📊 Best rows per (method, n_cells_int): {len(best_df)}")

    # ----- Plots -----
    make_bar_chart(best_df, out_dir, fixed_n=args.fixed_n)
    make_scatter(best_df, out_dir)

    # WOT plots (only if WOT columns are present after merge)
    wot_metric_col = choose_wot_metric(best_df)
    if wot_metric_col is not None:
        make_wot_bar_chart(best_df, out_dir, fixed_n=args.fixed_n, metric_col=wot_metric_col)
        make_wot_scatter(best_df, out_dir, metric_col=wot_metric_col)
    else:
        print("ℹ️ No WOT columns present in best_df; skipping WOT-specific figures.")

    # ----- Summary table -----
    save_summary_table(best_df, out_dir)


if __name__ == "__main__":
    main()
