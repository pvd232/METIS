#!/usr/bin/env python3
"""
scripts/run_wot_benchmark.py

Run Waddington-OT–style interpolation benchmarks across multiple
embeddings (PCA, Diffmap, EGGFM, etc.) using ti_eval_extended.py.

This is analogous to run_baseline_manifolds.py but only runs TI eval
on precomputed embeddings (no training).

It writes per-run configs under:
  configs/ti_wot_runs/<run_id>.yml

and metrics JSONs under:
  out/metrics/ti/<run_id>/metrics.json
"""

from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import yaml


# --------------------------------------------------------------------------
# Define the runs you want to compare
# --------------------------------------------------------------------------
#
# Make sure these paths/keys match your existing embedding files.
# The names here mirror the run_ids visible in your CSV.
# Extend this list with additional EGGFM ablations or scVI runs if desired.
# --------------------------------------------------------------------------

RUNS: List[Dict] = [
    # PCA baselines
    {
        "run_id": "weinreb_pca_local_n5000__X_pca",
        "base_run_id": "weinreb_pca_local_n5000__X_pca",
        "ad_path": "data/embedding/weinreb_pca_local_n5000.h5ad",
        "embedding_key": "X_pca",
        "metric_mode": "baseline_pca",
        "max_cells": 5000,
    },
    {
        "run_id": "weinreb_pca_meso_n15000__X_pca",
        "base_run_id": "weinreb_pca_meso_n15000__X_pca",
        "ad_path": "data/embedding/weinreb_pca_meso_n15000.h5ad",
        "embedding_key": "X_pca",
        "metric_mode": "baseline_pca",
        "max_cells": 15000,
    },

    # Diffmap baselines
    {
        "run_id": "weinreb_diffmap_local_n5000__X_diffmap",
        "base_run_id": "weinreb_diffmap_local_n5000__X_diffmap",
        "ad_path": "data/embedding/weinreb_diffmap_local_n5000.h5ad",
        "embedding_key": "X_diffmap",
        "metric_mode": "baseline_diffmap",
        "max_cells": 5000,
    },
    {
        "run_id": "weinreb_diffmap_meso_n15000__X_diffmap",
        "base_run_id": "weinreb_diffmap_meso_n15000__X_diffmap",
        "ad_path": "data/embedding/weinreb_diffmap_meso_n15000.h5ad",
        "embedding_key": "X_diffmap",
        "metric_mode": "baseline_diffmap",
        "max_cells": 15000,
    },

    # Main EGGFM embedding (full 60k or 15k eval; adjust max_cells as you like)
    {
        "run_id": "weinreb_eggfm_diffmap__X_eggfm_diffmap",
        "base_run_id": "weinreb_eggfm_diffmap__X_eggfm_diffmap",
        "ad_path": "data/embedding/weinreb_eggfm_diffmap.h5ad",
        "embedding_key": "X_eggfm_diffmap",
        "metric_mode": "riem_normal",  # or scm / riem_curvature / etc
        "max_cells": 15000,
    },
]


# --------------------------------------------------------------------------
# Util
# --------------------------------------------------------------------------

RUN_DIR = Path("configs/ti_wot_runs")
RUN_DIR.mkdir(parents=True, exist_ok=True)


def sh(cmd: str) -> None:
    print(f"\n🔥 Running: {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    for run in RUNS:
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
            # No need to specify eggfm_diffmap for baselines; ti_eval_extended
            # will fall back to ti_eval.metric_mode when eggfm_diffmap is absent.
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
