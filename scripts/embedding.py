#!/usr/bin/env python3
# scripts/embedding.py
"""
EGGFM → Diffmap dimension reduction for a QC AnnData.

This script builds an EGGFM-based diffusion map embedding (no PCA baselines)
using a trained EnergyMLP prior and a DiffusionConfig from params.yml.

It:
  - optionally subsamples cells for speed (with a hard cap of 60000)
  - builds an EGGFM-based diffusion embedding via `build_eggfm_geometry_from_config`  
  - stores:
      * X_eggfm_diffmap  (full EGGFM-based diffusion embedding)
      * X_diff_eggfm     (first k dims of EGGFM diffusion embedding)
  - writes .h5ad files inside the embedding output directory:
      * weinreb_eggfm_diffmap.h5ad
      * weinreb_eggfm_diffmap_<regime>_n<N>.h5ad
        where <regime> ∈ {global, meso, local} based on n_cells_sample
  - ALSO writes the effective YAML config for this embedding to:
      * configs/ablation_runs/embedding_eggfm_diffmap_<regime>_n<N>.yml

Config block (in configs/params.yml):

embedding:
  ad_path: data/interim/weinreb_qc.h5ad
  energy_ckpt: out/models/eggfm/eggfm_energy_weinreb.pt
  out_dir: data/embedding
  k: 30
  n_neighbors: 30
  n_cells_sample: 0   # 0 or missing => use all cells (subject to hard cap)
  seed: 7
  cfg_key: eggfm_diffmap

Usage:

  python scripts/embedding.py --params configs/params.yml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
import torch
import scanpy as sc  # type: ignore

from medit.eggfm.models import EnergyMLP
from medit.eggfm.config import EnergyModelConfig
from medit.diffusion.config import diffusion_config_from_params  # <<< CHANGED (removed EGGFMGeomConfig)
from medit.diffusion.embed import build_eggfm_geometry_from_config  # unchanged


# ---------------------------------------------------------------------------
# Subsampling helpers
# ---------------------------------------------------------------------------

def maybe_subsample_with_cap(
    ad_full: sc.AnnData,
    n_cells_sample: int,
    seed: int,
    hard_cap: int = 130800,
) -> sc.AnnData:
    """
    Optionally subsample cells for faster experimentation, with a hard cap.

    Rules:
      - If n_cells_sample <= 0: use all cells, but cap at `hard_cap`.
      - If n_cells_sample > 0: use min(n_cells_sample, hard_cap, n_obs).
    """
    n_obs = ad_full.n_obs

    if n_cells_sample <= 0:
        target = min(hard_cap, n_obs)
        reason = f"default (cap={hard_cap})"
    else:
        target = min(n_cells_sample, hard_cap, n_obs)
        reason = f"requested={n_cells_sample}, cap={hard_cap}"

    if target < n_obs:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_obs, size=target, replace=False))
        ad = ad_full[idx].copy()
        print(
            f"[EMBEDDING] Subsampled {target}/{n_obs} cells ({reason})",
            flush=True,
        )
    else:
        ad = ad_full
        print(
            f"[EMBEDDING] Using all {n_obs} cells (<= hard_cap={hard_cap})",
            flush=True,
        )

    return ad


def infer_regime(n_cells_sample: int, n_cells_actual: int, hard_cap: int = 130800) -> str:
    """
    Infer a regime label similar to the PCA baselines (global/meso/local).

    - global: n_cells_sample <= 0 (i.e. 'all up to cap') or >= hard_cap
    - meso:   15000 <= n_cells_sample < hard_cap
    - local:  5000  <= n_cells_sample < 15000
    - otherwise: return a generic 'sub<N>' label
    """
    if n_cells_sample <= 0 or n_cells_sample >= hard_cap:
        return "global"
    if n_cells_sample >= 15000:
        return "meso"
    if n_cells_sample >= 5000:
        return "local"
    return f"sub{n_cells_actual}"


# Directory for saving embedding configs
ABLAT_RUN_DIR = Path("configs/ablation_runs")
ABLAT_RUN_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---------- CLI: only params path ----------
    p = argparse.ArgumentParser(
        description="EGGFM→Diffmap dimension reductions (no PCA baselines)."
    )
    p.add_argument(
        "--params",
        required=True,
        help="Path to configs/params.yml",
    )
    args = p.parse_args()

    # ---------- load params + embedding config ----------
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())
    emb_cfg: Dict[str, Any] = params.get("embedding", {})

    ad_path = Path(emb_cfg["ad_path"])
    energy_ckpt = Path(emb_cfg["energy_ckpt"])
    out_dir = Path(emb_cfg.get("out_dir", "data/embedding"))
    out_dir.mkdir(parents=True, exist_ok=True)

    k = int(emb_cfg.get("k", 30))
    # n_neighbors kept for transparency; diffusion config typically carries neighbors
    n_neighbors = int(emb_cfg.get("n_neighbors", 30))
    n_cells_sample = int(emb_cfg.get("n_cells_sample", 0))
    seed = int(emb_cfg.get("seed", 7))
    cfg_key = emb_cfg.get("cfg_key", "eggfm_diffmap")

    # NOTE: store_kernel / store_eigvals / store_knn are no longer needed,
    # because we always store everything in the geometry builder.          # <<< CHANGED (conceptual)

    # ---------- echo resolved embedding config ----------
    print("[EMBEDDING] Resolved embedding config (from params.yml):", flush=True)
    print(f"  ad_path        = {ad_path}", flush=True)
    print(f"  energy_ckpt    = {energy_ckpt}", flush=True)
    print(f"  out_dir        = {out_dir}", flush=True)
    print(f"  k              = {k}", flush=True)
    print(f"  n_neighbors    = {n_neighbors}", flush=True)
    print(f"  n_cells_sample = {n_cells_sample}", flush=True)
    print(f"  seed           = {seed}", flush=True)
    print(f"  cfg_key        = {cfg_key}", flush=True)

    # ---------- seeds ----------
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---------- diffusion config ----------
    diff_cfg = diffusion_config_from_params(params, key=cfg_key)
    try:
        from dataclasses import asdict
        diff_cfg_dict = asdict(diff_cfg)
    except Exception:
        diff_cfg_dict = diff_cfg.__dict__
    print("[EMBEDDING] Using DiffusionConfig:", flush=True)
    for k_cfg, v_cfg in diff_cfg_dict.items():
        print(f"  {k_cfg}: {v_cfg}", flush=True)

    # ---------- load QC AnnData ----------
    ad_full = sc.read_h5ad(ad_path)
    print(
        f"[EMBEDDING] Loaded QC AnnData: "
        f"{ad_full.n_obs} cells × {ad_full.n_vars} genes",
        flush=True,
    )

    # ---------- optional subsample with hard cap ----------
    ad = maybe_subsample_with_cap(
        ad_full=ad_full,
        n_cells_sample=n_cells_sample,
        seed=seed,
        hard_cap=130800,
    )
    n_cells_actual = ad.n_obs
    print(
        f"[EMBEDDING] Using {n_cells_actual} cells for embedding "
        f"(total available={ad_full.n_obs})",
        flush=True,
    )

    regime = infer_regime(n_cells_sample=n_cells_sample, n_cells_actual=n_cells_actual)
    print(f"[EMBEDDING] Inferred regime: {regime}", flush=True)

    # ---------- EGGFM model ----------
    ckpt = torch.load(energy_ckpt, map_location="cpu", weights_only=False)
    model_cfg_dict = ckpt.get("model_cfg", {})
    n_genes = int(ckpt.get("n_genes", ad.n_vars))

    print("[EMBEDDING] Energy model config from checkpoint:", model_cfg_dict, flush=True)
    print(f"[EMBEDDING] n_genes from checkpoint / AnnData: {n_genes}", flush=True)

    model_cfg = EnergyModelConfig(**model_cfg_dict)
    latent_dim = getattr(model_cfg, "latent_dim", None)

    energy_model = EnergyMLP(
        n_genes=n_genes,
        hidden_dims=model_cfg.hidden_dims,
        latent_dim=latent_dim,
    )
    energy_model.load_state_dict(ckpt["state_dict"])
    energy_model.eval()

    # ---------- EGGFM → Diffmap ----------
    print("[EMBEDDING] Building EGGFM-based diffusion embedding", flush=True)

    # Directly pass the diffusion config into the geometry builder         
    ad = build_eggfm_geometry_from_config(
        ad=ad,
        energy_model=energy_model,
        diff_cfg=diff_cfg,            
        obsm_key="X_eggfm_diffmap",
    )

    # First k dims of EGGFM diffusion (analogous to X_diff_pca)
    X_diff_eggfm = ad.obsm["X_eggfm_diffmap"][:, :k].copy()
    ad.obsm["X_diff_eggfm"] = X_diff_eggfm

    # ---------- write outputs (.h5ad) ----------
    # 1) Generic output (backwards-compatible with existing scripts)
    eggfm_out = out_dir / "weinreb_eggfm_diffmap.h5ad"
    ad.write_h5ad(eggfm_out)
    print(f"[EMBEDDING] Wrote EGGFM embedding AnnData to {eggfm_out}")

    # 2) Regime-specific output, aligned with pca_global / pca_meso / pca_local style
    regime_out = out_dir / f"weinreb_eggfm_diffmap_{regime}_n{n_cells_actual}.h5ad"
    ad.write_h5ad(regime_out)
    print(f"[EMBEDDING] Wrote regime-specific EGGFM embedding AnnData to {regime_out}")

    # ---------- write effective YAML config ----------                      
    run_id = f"weinreb_eggfm_diffmap_{regime}_n{n_cells_actual}"
    params_to_save = dict(params)  # shallow copy is enough here

    # Attach a small meta block so we can later link config ↔ embedding file
    meta = {
        "run_id": run_id,
        "regime": regime,
        "n_cells_actual": int(n_cells_actual),
        "embedding_out": str(regime_out),
        "energy_ckpt": str(energy_ckpt),
        "cfg_key": cfg_key,
    }
    params_to_save["embedding_meta"] = meta

    # --- Make the ti_eval block self-consistent with this embedding ---
    ti_eval_cfg = params_to_save.get("ti_eval", {}) or {}
    ti_eval_cfg.update(
        {
            "ad_path": str(regime_out),
            "embedding_key": "X_eggfm_diffmap",
            "time_key": ti_eval_cfg.get("time_key", "Time point"),
            "cluster_key": ti_eval_cfg.get("cluster_key", "Cell type annotation"),
            "fate_key": ti_eval_cfg.get("fate_key", None),
            "baseline_embedding_key": ti_eval_cfg.get("baseline_embedding_key", None),
            "root_mask_key": ti_eval_cfg.get("root_mask_key", None),
            "n_neighbors": ti_eval_cfg.get("n_neighbors", 30),
            "n_dcs": ti_eval_cfg.get("n_dcs", 10),
            "max_cells": int(n_cells_actual),
            "out_dir": ti_eval_cfg.get("out_dir", "out/metrics/ti"),
        }
    )
    params_to_save["ti_eval"] = ti_eval_cfg

    # Also put run_id / base_run_id at the top level like the baselines do
    params_to_save["run_id"] = run_id
    params_to_save["base_run_id"] = run_id

    cfg_path = ABLAT_RUN_DIR / f"embedding_eggfm_diffmap_{regime}_n{n_cells_actual}.yml"
    cfg_path.write_text(yaml.dump(params_to_save, sort_keys=False))
    print(f"[EMBEDDING] Wrote embedding config to {cfg_path}")


if __name__ == "__main__":
    main()
