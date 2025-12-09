#!/usr/bin/env python3
# scripts/embedding.py
"""
PCA→Diffmap and EGGFM→Diffmap dimension reductions for a QC AnnData.

This script compares:
  - classical PCA → diffusion map
  - EGGFM-based diffusion map (using an energy prior)

It:
  - optionally subsamples cells for speed (with a hard cap of 60000)
  - computes PCA (if missing) and a PCA-based diffusion map
  - builds an EGGFM-based diffusion embedding via `EGGFMDiffusionEngine`
  - stores:
      * X_pca          (PCA embedding)
      * X_diff_pca     (PCA-based diffusion map, first k dims)
      * X_eggfm_diffmap (full EGGFM-based diffusion embedding)
      * X_diff_eggfm   (first k dims of EGGFM diffusion embedding)
  - writes two .h5ad files inside the embedding output directory:
      * weinreb_eggfm_diffmap.h5ad   (full AnnData with all views)
      * weinreb_pca_diffmap.h5ad     (PCA→Diffmap views only)

Config block (in configs/params.yml):

embedding:
  ad_path: data/interim/weinreb_qc.h5ad
  energy_ckpt: out/models/eggfm/eggfm_energy_weinreb.pt
  out_dir: data/interim/embedding
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
from medit.diffusion.config import diffusion_config_from_params
from medit.diffusion.embed import build_diffusion_embedding_from_config


def build_pca_diffmap(
    ad: sc.AnnData,
    k: int,
    n_neighbors: int,
) -> None:
    """
    Classical PCA → Diffmap.

    Populates:
      - ad.obsm["X_pca"]       (if missing)
      - ad.obsm["X_diff_pca"]  (first k diffusion components)
    """
    # PCA (compute if missing)
    if "X_pca" not in ad.obsm:
        n_comps = max(k, 30)
        print(f"[EMBEDDING] Computing PCA with n_comps={n_comps}", flush=True)
        sc.pp.scale(ad, max_value=10)
        sc.pp.pca(ad, n_comps=n_comps)
    else:
        print("[EMBEDDING] Using existing X_pca", flush=True)

    # Diffusion map on PCA space
    print(
        f"[EMBEDDING] PCA → Diffmap (n_neighbors={n_neighbors}, k={k})",
        flush=True,
    )
    sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.diffmap(ad, n_comps=k)

    X_diff_pca = ad.obsm["X_diffmap"][:, :k].copy()
    ad.obsm["X_diff_pca"] = X_diff_pca
    # Keep Scanpy's X_diffmap around; it can be useful later.


def maybe_subsample_with_cap(
    ad_full: sc.AnnData,
    n_cells_sample: int,
    seed: int,
    hard_cap: int = 60000,
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


def main() -> None:
    # ---------- CLI: only params path ----------
    p = argparse.ArgumentParser(
        description="PCA→Diffmap and EGGFM→Diffmap dimension reductions."
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
    out_dir = Path(emb_cfg.get("out_dir", "data/interim/embedding"))
    out_dir.mkdir(parents=True, exist_ok=True)

    k = int(emb_cfg.get("k", 30))
    n_neighbors = int(emb_cfg.get("n_neighbors", 30))
    n_cells_sample = int(emb_cfg.get("n_cells_sample", 0))
    seed = int(emb_cfg.get("seed", 7))
    cfg_key = emb_cfg.get("cfg_key", "eggfm_diffmap")

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
        hard_cap=60000,
    )
    print(
        f"[EMBEDDING] Using {ad.n_obs} cells for embedding "
        f"(total available={ad_full.n_obs})",
        flush=True,
    )

    # ---------- PCA → Diffmap ----------
    build_pca_diffmap(ad, k=k, n_neighbors=n_neighbors)

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
    ad = build_diffusion_embedding_from_config(
        ad=ad,
        energy_model=energy_model,
        diff_cfg=diff_cfg,
        obsm_key="X_eggfm_diffmap",
    )

    # First k dims of EGGFM diffusion
    X_diff_eggfm = ad.obsm["X_eggfm_diffmap"][:, :k].copy()
    ad.obsm["X_diff_eggfm"] = X_diff_eggfm

    # ---------- write outputs ----------
    pca_out = out_dir / "weinreb_pca_diffmap.h5ad"
    eggfm_out = out_dir / "weinreb_eggfm_diffmap.h5ad"

    # PCA-only view
    ad_pca_only = ad.copy()
    keep_keys = ["X_pca", "X_diff_pca"]
    ad_pca_only.obsm = {
        k_: v for k_, v in ad_pca_only.obsm.items() if k_ in keep_keys
    }
    ad_pca_only.write_h5ad(pca_out)
    print(f"[EMBEDDING] Wrote PCA Diffmap embedding AnnData to {pca_out}")

    # Full EGGFM embedding (includes PCA + EGGFM views)
    ad.write_h5ad(eggfm_out)
    print(f"[EMBEDDING] Wrote EGGFM embedding AnnData to {eggfm_out}")


if __name__ == "__main__":
    main()
