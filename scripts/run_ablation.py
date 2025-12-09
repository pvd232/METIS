#!/usr/bin/env python3
"""
scripts/run_ablation.py

Perform ablations over:
  - SCM hyperparameters
  - Riemannian metric modes & hyperparameters

For each configuration:
  1) Write a params.yml
  2) Run embedding.py
  3) Run ti_eval.py
"""

from __future__ import annotations
import os
import yaml
import subprocess
from pathlib import Path
from copy import deepcopy

# --------------------------------------------------------------------------
#  Search space
# --------------------------------------------------------------------------

SCM_GRID = [
    {"metric_gamma": 5.0, "metric_lambda": 20.0, "energy_clip_abs": 6.0},
    {"metric_gamma": 3.0, "metric_lambda": 15.0, "energy_clip_abs": 4.0},
    {"metric_gamma": 8.0, "metric_lambda": 30.0, "energy_clip_abs": 8.0},
]

RIEM_MODES = ["riem_tangent", "riem_curvature", "riem_normal"]

RIEM_GRID = [
    {
        "tangent_dim": 10,
        "tangent_eps": 0.01,
        "tangent_k": 30,
        "curvature_k": 30,
        "curvature_scale": 0.5,
        "normal_k": 30,
        "normal_weight": 1.0,
    },
    {
        "tangent_dim": 20,
        "tangent_eps": 0.02,
        "tangent_k": 50,
        "curvature_k": 50,
        "curvature_scale": 1.0,
        "normal_k": 50,
        "normal_weight": 0.5,
    },
]

# --------------------------------------------------------------------------
#  Base params.yml template
# --------------------------------------------------------------------------

BASE_PARAMS = {
    "seed": 7,
    "hvg_n_top_genes": 2000,

    "spec": {
        "n_pcs": 30,
        "dcol_max_cells": 3000,
        "ari_label_key": "Cell type annotation",
        "ari_n_dims": 10,
        "ad_file": "data/interim/weinreb.h5ad"
    },

    "qc": {
        "min_cells": 500,
        "min_genes": 200,
        "max_pct_mt": 15,
    },

    "eggfm_model": {
        "hidden_dims": [512, 512, 512, 512],
        "latent_dim": 64,
    },

    "eggfm_train": {
        "batch_size": 8192,
        "num_epochs": 50,
        "lr": 5.0e-3,
        "sigma": 0.15,
        "device": "cuda",
        "latent_space": "pca",
        "early_stop_patience": 5,
        "early_stop_min_delta": 0.0,
        "n_cells_sample": 0,

        # riemannian training regularizer
        "riemann_reg_type": "hess_smooth",
        "riemann_reg_weight": 0.1,
        "riemann_eps": 0.01,
        "riemann_n_dirs": 4,
    },

    "eggfm_diffmap": {
        "geometry_source": "pca",
        "energy_source": "hvg",

        # NOTE: metric_mode is overridden inside ablation loops
        "metric_mode": "scm",

        "n_neighbors": 30,
        "n_comps": 30,
        "device": "cuda",
        "hvp_batch_size": 8192,
        "eps_trunc": "no",
        "distance_power": 2.0,
        "t": 3.0,
        "norm_type": "l2",

        # SCM defaults — overridden in SCM ablation loop
        "metric_gamma": 5.0,
        "metric_lambda": 20.0,
        "energy_clip_abs": 6.0,
        "energy_batch_size": 8192,

        # Riemannian geometry defaults
        "tangent_dim": 10,
        "tangent_eps": 0.01,
        "tangent_k": 30,
        "curvature_k": 30,
        "curvature_scale": 1.0,
        "normal_k": 30,
        "normal_weight": 1.0,
    },

    "embedding": {
        "ad_path": "data/interim/weinreb_qc.h5ad",
        "energy_ckpt": "out/models/eggfm/eggfm_energy_weinreb.pt",
        "out_dir": "data/embedding",
        "k": 30,
        "n_neighbors": 30,
        "n_cells_sample": 0,
        "seed": 7,
        "cfg_key": "eggfm_diffmap",
    },

    "ti_eval": {
        "ad_path": "data/embedding/weinreb_eggfm_diffmap.h5ad",
        "embedding_key": "X_eggfm_diffmap",
        "time_key": "Time point",
        "cluster_key": "Cell type annotation",
        "fate_key": None,
        "baseline_embedding_key": None,
        "root_mask_key": None,
        "n_neighbors": 30,
        "n_dcs": 10,
        "max_cells": 60000,
        "out_dir": "out/metrics/ti",
    },
}

# --------------------------------------------------------------------------
#  Make directory for generated configs
# --------------------------------------------------------------------------

RUN_DIR = Path("configs/ablation_runs")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
#  Helper: launch a shell command
# --------------------------------------------------------------------------

def sh(cmd: str):
    print(f"\n🔥 Running: {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, check=True)

# --------------------------------------------------------------------------
#  Main ablation loop
# --------------------------------------------------------------------------

def main():
    run_idx = 0

    # --------------------------------------------------------
    # 1. SCM Ablations
    # --------------------------------------------------------
    for scm_cfg in SCM_GRID:
        run_idx += 1
        params = deepcopy(BASE_PARAMS)

        params["eggfm_diffmap"]["metric_mode"] = "scm"
        params["eggfm_diffmap"].update(scm_cfg)

        cfg_path = RUN_DIR / f"ablation_scm_{run_idx}.yml"
        cfg_path.write_text(yaml.dump(params, sort_keys=False))

        print(f"\n=== SCM RUN {run_idx} ===")
        sh(f"python scripts/embedding.py --params {cfg_path}")
        sh(f"python scripts/ti_eval.py --params {cfg_path}")

    # --------------------------------------------------------
    # 2. Riemannian Ablations
    # --------------------------------------------------------
    for mode in RIEM_MODES:
        for riem_cfg in RIEM_GRID:
            run_idx += 1
            params = deepcopy(BASE_PARAMS)

            params["eggfm_diffmap"]["metric_mode"] = mode
            params["eggfm_diffmap"].update(riem_cfg)

            cfg_path = RUN_DIR / f"ablation_{mode}_{run_idx}.yml"
            cfg_path.write_text(yaml.dump(params, sort_keys=False))

            print(f"\n=== RIEM RUN {run_idx}: {mode} ===")
            sh(f"python scripts/embedding.py --params {cfg_path}")
            sh(f"python scripts/ti_eval.py --params {cfg_path}")

    print("\n✔ ALL ABLATIONS COMPLETE.\n")


if __name__ == "__main__":
    main()
