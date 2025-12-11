#!/usr/bin/env python3
"""
scripts/run_ablation_riem_xhard.py

Brutal, Riemannian-only ablation that pushes hard around your best
regimes:

  - riem_normal @ 15k cells (around r62)
  - riem_curvature @ 5k cells (around r63–r66)

This is *not* gentle. It cranks:
  - normal_weight and normal_k (for riem_normal)
  - curvature_beta and curvature_scale (for riem_curvature)
plus a few tangent_dim / tangent_eps tweaks.

All runs are still L4-safe (15k or 5k cells; hvp_batch_size chosen
from configs you already ran successfully).
"""

from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


# =============================================================================
#  SEARCH SPACES — AROUND BEST CONFIGS, BUT TURNED UP
# =============================================================================

# -------------------------------------------------------------------------
# Riemannian normal @ 15k
#
# Best so far (r62):
#   tangent_dim = 15
#   tangent_eps = 0.01
#   tangent_k   = 15
#   normal_k    = 40
#   normal_weight = 1.25
#
# Here we:
#   - push normal_weight to 1.5, 1.75, 2.0
#   - push normal_k to 50 and 60
#   - tweak tangent_dim (12, 18) and tangent_eps / tangent_k slightly
# -------------------------------------------------------------------------

RIEM_NORMAL_XHARD = [
]

RIEM_CURVATURE_XHARD = [
    # Mid-scale, stronger curvature than old 0.2
        {
        "name": "beta6.0_scale10.0",
        "curvature_beta": 6.0,
        "curvature_clip_std": 10.0,
        "curvature_k": 40,
        "curvature_scale": 10.0,
    },]

# =============================================================================
#  BASE PARAMS (same backbone as your current Riemannian ablation)
# =============================================================================
BASE = {
    "seed": 11,
    "hvg_n_top_genes": 2000,

    "spec": {
        "n_pcs": 30,
        "dcol_max_cells": 3000,
        "ari_label_key": "Cell type annotation",
        "ari_n_dims": 10,
        "ad_file": "data/interim/weinreb.h5ad",
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
        "riemann_reg_type": "hess_smooth",
        "riemann_reg_weight": 0.1,
        "riemann_eps": 0.01,
        "riemann_n_dirs": 4,
    },

    "eggfm_diffmap": {
        "geometry_source": "pca",
        "energy_source": "hvg",
        "metric_mode": "scm",  # overwritten per run

        "n_neighbors": 30,
        "n_comps": 30,
        "device": "cuda",
        "hvp_batch_size": 8192,
        "eps_trunc": "yes",
        "distance_power": 2.0,
        "t": 4.0,
        "norm_type": "l2",        

        # Riemann defaults (overwritten)
        "tangent_dim": 10,
        "tangent_eps": 0.01,
        "tangent_k": 30,
        "curvature_k": 30,
        "curvature_scale": 5.0,
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


# =============================================================================
#  UTIL
# =============================================================================

RUN_DIR = Path("configs/ablation_runs")
RUN_DIR.mkdir(parents=True, exist_ok=True)


def sh(cmd: str) -> None:
    print(f"\n🔥 Running: {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    idx = 0

    # ================================
    # 1. Riemannian normal — 15k cells
    # ================================
    for g in RIEM_NORMAL_XHARD:
        idx += 1
        cfg = deepcopy(BASE)

        cfg["eggfm_diffmap"]["metric_mode"] = "riem_normal"
        cfg["eggfm_diffmap"]["hvp_batch_size"] = 256   # safe for 15k
        cfg["eggfm_diffmap"]["n_neighbors"] = 10       # metric neighbors

        cfg["embedding"]["n_cells_sample"] = 15000

        # Apply core riem_normal defaults around best regime
        cfg["eggfm_diffmap"]["tangent_dim"] = g["tangent_dim"]
        cfg["eggfm_diffmap"]["tangent_eps"] = g["tangent_eps"]
        cfg["eggfm_diffmap"]["tangent_k"] = g["tangent_k"]
        cfg["eggfm_diffmap"]["normal_k"] = g["normal_k"]
        cfg["eggfm_diffmap"]["normal_weight"] = g["normal_weight"]

        run_name = g["name"]
        path = RUN_DIR / f"xhard_riem_normal_{idx}_{run_name}.yml"
        path.write_text(yaml.dump(cfg, sort_keys=False))

        sh(f"python scripts/embedding.py --params {path}")
        sh(f"python scripts/ti_eval.py --params {path}")

    # ================================
    # 2. Riemannian curvature — 5k cells
    # ================================
    for g in RIEM_CURVATURE_XHARD:
        idx += 1
        cfg = deepcopy(BASE)

        cfg["eggfm_diffmap"]["metric_mode"] = "riem_curvature"
        cfg["eggfm_diffmap"]["hvp_batch_size"] = 256   # you already ran 5k with this
        cfg["eggfm_diffmap"]["n_neighbors"] = 10         # curvature metric neighbors

        cfg["embedding"]["n_cells_sample"] = 15000

        # Fix tangent/normal around successful regime, but allow overrides
        cfg["eggfm_diffmap"]["tangent_dim"] = g.get("tangent_dim", 10)
        cfg["eggfm_diffmap"]["tangent_eps"] = g.get("tangent_eps", 0.01)
        cfg["eggfm_diffmap"]["tangent_k"] = g.get("tangent_k", 30)
        cfg["eggfm_diffmap"]["normal_k"] = 30
        cfg["eggfm_diffmap"]["normal_weight"] = g.get("normal_weight", 1.0)

        cfg["eggfm_diffmap"]["curvature_beta"] = g["curvature_beta"]
        cfg["eggfm_diffmap"]["curvature_clip_std"] = g["curvature_clip_std"]
        cfg["eggfm_diffmap"]["curvature_k"] = g["curvature_k"]
        cfg["eggfm_diffmap"]["curvature_scale"] = g["curvature_scale"]

        run_name = g["name"]
        path = RUN_DIR / f"xhard_riem_curvature_{idx}_{run_name}.yml"
        path.write_text(yaml.dump(cfg, sort_keys=False))

        sh(f"python scripts/embedding.py --params {path}")
        sh(f"python scripts/ti_eval.py --params {path}")

    print("\n🎉 ALL XHARD RIEMANNIAN ABLATIONS COMPLETE.\n")


if __name__ == "__main__":
    main()
