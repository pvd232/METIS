#!/usr/bin/env python3
"""
scripts/run_final_push.py

"Final boss" Riemannian ablation, centered on your best EGGFM configs:

  • riem_normal_push_15000
      – based on best riem_normal run at N=15000 (r72, Spearman ~0.419)
      – doubles effective cells to 30k and cranks normal penalties

  • riem_curvature_push_15000
      – based on best riem_curvature run at N=5000 (r63–66, Spearman ~0.421)
      – lifts to 15k cells and cranks curvature_scale / curvature_k / normal penalties

Both runs:
  – reuse the same BASE config as earlier ablations (weinreb, 2000 HVGs, latent_dim=64)
  – write configs under configs/ablation_runs/
  – then call:
        python scripts/embedding.py --params <cfg>
        python scripts/ti_eval.py    --params <cfg>

WARNING: These settings are intentionally heavy
         (n_cells_sample up to 30000, hvp_batch_size ~512, n_neighbors=40)
         and may OOM an L4 (24GB) depending on what else is running.

Usage:

  python scripts/run_final_push.py
"""

from __future__ import annotations
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


# =============================================================================
#  BASE CONFIG (mirrors your existing run_ablation.py BASE)
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

        # Riemannian regularizer (training-time only)
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
        "hvp_batch_size": 8192,  # overwritten per run
        "eps_trunc": "no",
        "distance_power": 2.0,
        "t": 3.0,
        "norm_type": "l2",

        # SCM defaults (ignored when metric_mode != "scm")
        "metric_gamma": 5.0,
        "metric_lambda": 20.0,
        "energy_clip_abs": 6.0,
        "energy_batch_size": 8192,

        # Riemann defaults (overwritten)
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
        "n_cells_sample": 0,  # overwritten per run
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

    # -------------------------------------------------------------------------
    # 1) riem_normal_push_15000
    #    – based on r72:
    #         metric_mode = riem_normal
    #         tangent_dim = 15, tangent_k = 15
    #         normal_k   = 40, normal_weight = 1.25
    #         N ≈ 15000
    #    – we push:
    #         n_cells_sample = 30000
    #         n_neighbors    = 40
    #         hvp_batch_size = 512
    #         tangent_dim    = 20
    #         tangent_k      = 30
    #         normal_k       = 60
    #         normal_weight  = 1.5
    # -------------------------------------------------------------------------
    idx += 1
    cfg = deepcopy(BASE)

    cfg["eggfm_diffmap"]["metric_mode"] = "riem_normal"
    cfg["eggfm_diffmap"]["hvp_batch_size"] = 512
    cfg["eggfm_diffmap"]["n_neighbors"] = 40

    cfg["eggfm_diffmap"]["tangent_dim"] = 20
    cfg["eggfm_diffmap"]["tangent_eps"] = 0.01
    cfg["eggfm_diffmap"]["tangent_k"] = 30

    cfg["eggfm_diffmap"]["normal_k"] = 60
    cfg["eggfm_diffmap"]["normal_weight"] = 1.5

    # keep curvature_k/scale at moderately safe defaults (normal metric ignores them)
    cfg["eggfm_diffmap"]["curvature_k"] = 30
    cfg["eggfm_diffmap"]["curvature_scale"] = 1.0

    cfg["embedding"]["n_cells_sample"] = 30000

    path_normal = RUN_DIR / f"riem_normal_final_push_{idx}.yml"
    path_normal.write_text(yaml.dump(cfg, sort_keys=False))

    sh(f"python scripts/embedding.py --params {path_normal}")
    sh(f"python scripts/ti_eval.py --params {path_normal}")

    # -------------------------------------------------------------------------
    # 2) riem_curvature_push_15000
    #    – based on r63 (best curvature run at 5k):
    #         tangent_dim = 10, tangent_k = 30
    #         curvature_k = 30, curvature_scale = 1.5
    #         normal_k   = 30, normal_weight = 1.0
    #         N ≈ 5000
    #    – we push:
    #         n_cells_sample = 15000
    #         n_neighbors    = 40
    #         hvp_batch_size = 512
    #         tangent_dim    = 15
    #         tangent_k      = 40
    #         curvature_k    = 50
    #         curvature_scale = 2.5
    #         normal_k       = 40
    #         normal_weight  = 1.5
    # -------------------------------------------------------------------------
    idx += 1
    cfg = deepcopy(BASE)

    cfg["eggfm_diffmap"]["metric_mode"] = "riem_curvature"
    cfg["eggfm_diffmap"]["hvp_batch_size"] = 512
    cfg["eggfm_diffmap"]["n_neighbors"] = 40

    cfg["eggfm_diffmap"]["tangent_dim"] = 15
    cfg["eggfm_diffmap"]["tangent_eps"] = 0.1
    cfg["eggfm_diffmap"]["tangent_k"] = 40

    cfg["eggfm_diffmap"]["curvature_k"] = 50
    cfg["eggfm_diffmap"]["curvature_scale"] = 2.5

    cfg["eggfm_diffmap"]["normal_k"] = 40
    cfg["eggfm_diffmap"]["normal_weight"] = 1.5

    cfg["embedding"]["n_cells_sample"] = 15000

    path_curv = RUN_DIR / f"riem_curvature_final_push_{idx}.yml"
    path_curv.write_text(yaml.dump(cfg, sort_keys=False))

    sh(f"python scripts/embedding.py --params {path_curv}")
    sh(f"python scripts/ti_eval.py --params {path_curv}")

    print("\n🎉 FINAL Riemannian PUSH ABLATIONS COMPLETE.\n")


if __name__ == "__main__":
    main()
