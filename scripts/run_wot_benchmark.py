#!/usr/bin/env python3
"""
scripts/run_wot_benchmark.py

Run Waddington-OT–style interpolation benchmarks across multiple
embeddings (PCA, Diffmap, EGGFM, scVI, etc.) using ti_eval_extended.py.

Compared to the original version, this script now:

  * Adds the **global PCA** run (weinreb_pca_global_n60000__X_pca)
  * Adds all three scVI runs (local / meso / global)
  * Adds the global Diffmap run
  * Adds **all EGGFM Weinreb runs**:
        weinreb_eggfm_diffmap__X_eggfm_diffmap__r2 ... __r94

Rather than hard-coding every run by hand, we read the TI summary CSV
(typically out/metrics/ti_ablation_summary6.csv) and automatically
construct the WOT configs for the relevant run_ids.

It writes per-run configs under:
  configs/ti_wot_runs/<run_id>.yml

and metrics JSONs under:
  out/metrics/ti/<run_id>/metrics.json
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

# Where to find the existing TI summary CSV (without WOT columns).
# Update this path if your summary lives somewhere else.
TI_SUMMARY_CSV = Path("out/metrics/ti_ablation_summary6.csv")

# Where to write the per-run config YAMLs that ti_eval_extended.py expects.
RUN_DIR = Path("configs/ti_wot_runs")
RUN_DIR.mkdir(parents=True, exist_ok=True)


def sh(cmd: str) -> None:
    print(f"\n🔥 Running: {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def build_runs_from_ti(summary_csv: Path) -> List[Dict]:
    """
    Parse the TI summary CSV and build a list of WOT benchmark runs.

    We explicitly include:
      - Global PCA
      - Global Diffmap
      - All three scVI runs (local / meso / global)
      - All EGGFM Weinreb runs  (weinreb_eggfm_diffmap__X_eggfm_diffmap__r2..__r94)

    For each selected run_id we grab:
      - base_run_id
      - ad_path
      - embedding_key
      - metric_mode
      - a reasonable max_cells value for WOT (preferring eval_n_cells, then max_cells,
        and falling back to 60000 if everything is missing / zero).
    """
    if not summary_csv.exists():
        raise FileNotFoundError(f"TI summary CSV not found at: {summary_csv}")

    df = pd.read_csv(summary_csv)

    # Ensure numeric columns are numeric
    for col in ("eval_n_cells", "max_cells"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # 1) Explicit run_ids we care about, beyond the original script.
    # ------------------------------------------------------------------
    target_run_ids = {
        # PCA
        "weinreb_pca_global_n60000__X_pca",
        # Diffmap
        "weinreb_diffmap_global_n60000__X_diffmap",
        # scVI (local / meso / global)
        "weinreb_scvi_local_n5000__X_scvi",
        "weinreb_scvi_meso_n15000__X_scvi",
        "weinreb_scvi_global_n60000__X_scvi",
    }

    # ------------------------------------------------------------------
    # 2) Add *all* EGGFM Weinreb runs:
    #       weinreb_eggfm_diffmap__X_eggfm_diffmap__r2..__r94
    # ------------------------------------------------------------------
    eggfm_mask = df["run_id"].fillna("").str.startswith(
        "weinreb_eggfm_diffmap__X_eggfm_diffmap__r"
    )
    eggfm_run_ids = df.loc[eggfm_mask, "run_id"].dropna().unique().tolist()
    target_run_ids.update(eggfm_run_ids)

    runs: List[Dict] = []
    seen: set[tuple[str, int]] = set()

    for run_id in sorted(target_run_ids):
        rows = df[df["run_id"] == run_id]
        if rows.empty:
            # If a run_id isn't actually present in the summary, just skip it.
            continue

        # There may be multiple rows per run_id (e.g. if evaluated with different
        # eval_n_cells). For WOT we just create one config per (run_id, n_cells).
        for _, row in rows.iterrows():
            eval_n = row.get("eval_n_cells", math.nan)
            max_cells = row.get("max_cells", math.nan)

            # Prefer eval_n_cells if it's present and > 0; otherwise fall back to max_cells;
            # if both are missing/zero, use 60000 (full Weinreb dataset).
            n = None
            if isinstance(eval_n, (int, float)) and not math.isnan(eval_n) and eval_n > 0:
                n = eval_n
            elif isinstance(max_cells, (int, float)) and not math.isnan(max_cells) and max_cells > 0:
                n = max_cells
            else:
                n = 60000

            n_int = int(n)
            key = (run_id, n_int)
            if key in seen:
                continue
            seen.add(key)

            base_run_id = row.get("base_run_id", run_id)
            ad_path = row["ad_path"]
            embedding_key = row["embedding_key"]
            metric_mode = row.get("metric_mode", None)

            # Pandas will give NaN for missing metric_mode; convert that to None so
            # yaml.dump will emit 'null', which downstream code interprets as "default".
            if isinstance(metric_mode, float) and math.isnan(metric_mode):
                metric_mode = None

            runs.append(
                {
                    "run_id": run_id,
                    "base_run_id": base_run_id,
                    "ad_path": ad_path,
                    "embedding_key": embedding_key,
                    "metric_mode": metric_mode,
                    "max_cells": n_int,
                }
            )

    return runs


def main() -> None:
    runs = build_runs_from_ti(TI_SUMMARY_CSV)
    if not runs:
        print("No matching runs found in TI summary; nothing to do.")
        return

    print(f"Discovered {len(runs)} runs to evaluate with WOT.")

    for run in runs:
        run_id = run["run_id"]
        base_run_id = run["base_run_id"]
        ad_path = run["ad_path"]
        embedding_key = run["embedding_key"]
        metric_mode = run["metric_mode"]
        max_cells = int(run["max_cells"])

        # Minimal config for ti_eval_extended
        cfg = {
            "seed": 11,
            "eggfm_train": {
                # Only used for logging; doesn't affect evaluation
                "sigma": 0.15,
                "lr": 5.0e-3,
                "batch_size": 8192,
                "riemann_reg_type": "hess_smooth",
                "riemann_reg_weight": 0.1,
            },
            # No need to specify eggfm_diffmap for baselines;
            # ti_eval_extended will fall back to ti_eval.metric_mode when absent.
            "eggfm_diffmap": {},
            "ti_eval": {
                "ad_path": ad_path,
                "embedding_key": embedding_key,
                "time_key": "Time point",
                "cluster_key": "Cell type annotation",
                "fate_key": None,
                "baseline_embedding_key": None,
                "root_mask_key": None,
                "n_neighbors": 30,
                "n_dcs": 10,
                "max_cells": max_cells,
                "out_dir": "out/metrics/ti",
                "run_id": run_id,
                "base_run_id": base_run_id,
                "metric_mode": metric_mode,
            },
        }

        cfg_path = RUN_DIR / f"{run_id}.yml"
        cfg_path.write_text(yaml.dump(cfg, sort_keys=False))

        sh(f"python scripts/ti_eval_extended.py --params {cfg_path}")


if __name__ == "__main__":
    main()
