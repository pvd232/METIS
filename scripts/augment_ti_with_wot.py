#!/usr/bin/env python3
"""
scripts/augment_ti_with_wot.py

Take an existing TI summary CSV (e.g. ti_ablation_summary6.csv) and
augment it with WOT + clone metrics from the per-run metrics.json
files written by ti_eval_extended.py.

Usage (from repo root):
    python scripts/augment_ti_with_wot.py \
        --input-csv out/metrics/ti_ablation_summary6.csv \
        --out-csv   out/metrics/ti_ablation_summary_wot.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv",
        required=True,
        help="Existing TI summary CSV (e.g. ti_ablation_summary6.csv).",
    )
    p.add_argument(
        "--out-csv",
        required=True,
        help="Path for augmented CSV with WOT/clone metrics.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)

    if "metrics_path" not in df.columns:
        raise ValueError(
            "Input CSV must have a 'metrics_path' column pointing to metrics.json files."
        )

    # Metrics added by ti_eval_extended.py
    extra_cols = [
        "wot_interp_mse",
        "clone_pt_var_within_mean",
        "clone_pt_var_within_median",
        "clone_pt_var_within_mean_norm",
        "clone_pt_var_within_median_norm",
        "clone_pt_kendall_vs_time_mean",
        "clone_pt_kendall_vs_time_median",
    ]

    # Ensure columns exist
    for col in extra_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Fill them from the per-run JSONs
    for idx, row in df.iterrows():
        mpath_str = row["metrics_path"]
        if not isinstance(mpath_str, str) or not mpath_str:
            continue

        mpath = Path(mpath_str)
        if not mpath.is_file():
            # Many ablations were evaluated with the old ti_eval.py;
            # only the runs re-evaluated with ti_eval_extended.py will have WOT metrics.
            continue

        try:
            with mpath.open("r") as f:
                metrics = json.load(f)
        except Exception:
            continue

        for col in extra_cols:
            if col in metrics and metrics[col] is not None:
                df.at[idx, col] = metrics[col]

    df.to_csv(args.out_csv, index=False)
    print(f"✅ Wrote augmented CSV with WOT/clone metrics to {args.out_csv}")


if __name__ == "__main__":
    main()
