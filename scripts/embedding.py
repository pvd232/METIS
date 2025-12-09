#!/usr/bin/env python3
# scripts/embedding.py
"""
PCA→Diffmap and EGGFM→Diffmap dimension reductions for a QC AnnData.

This script compares:
  - classical PCA → diffusion map
  - EGGFM-based diffusion map (using an energy prior)

It:
  - optionally subsamples cells for speed
  - computes PCA (if missing) and a PCA-based diffusion map
  - builds an EGGFM-based diffusion embedding via `EGGFMDiffusionEngine`
  - stores:
      * X_diff_pca      (PCA-based diffusion map, first k dims)
      * X_eggfm_diffmap (full EGGFM-based diffusion embedding)
      * X_diff_eggfm    (first k dims of EGGFM diffusion embedding)
  - writes three .h5ad files:
      * <stem>.h5ad                 (combined embeddings; path = --out)
      * <stem>_pca_diffmap.h5ad     (PCA→Diffmap only)
      * <stem>_eggfm_diffmap.h5ad   (EGGFM→Diffmap only)

Usage:

  python scripts/embedding.py \\
      --params configs/params.yml \\
      --ad data/interim/weinreb_qc.h5ad \\
      --energy-ckpt out/models/eggfm/eggfm_energy_weinreb.pt \\
      --out data/interim/weinreb_embedding.h5ad \\
      --k 10 \\
      --n-neighbors 30 \\
      --n-cells-sample 5000 \\
      --seed 7
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
from scripts.manifest_utils import write_embedding_manifest


def build_pca_diffmap(
    ad: sc.AnnData,
    k: int,
    n_neighbors: int,
) -> None:
    """
    Classical PCA → Diffmap.

    Populates:
      - ad.obsm["X_pca"]        (if missing)
      - ad.obsm["X_diff_pca"]  (first k diffusion components)
    """
    # PCA (compute if missing)
    if "X_pca" not in ad.obsm:
        n_comps = max(k, 20)
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


def maybe_subsample(
    ad: sc.AnnData,
    n_cells_sample: int,
    seed: int | None = None,
) -> sc.AnnData:
    """
    Optionally subsample cells for faster experimentation.

    If n_cells_sample <= 0 or >= n_obs, returns ad unchanged.
    Otherwise returns a new AnnData view with n_cells_sample cells.
    """
    if n_cells_sample <= 0 or ad.n_obs <= n_cells_sample:
        return ad

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(ad.n_obs, size=n_cells_sample, replace=False))
    print(
        f"[EMBEDDING] Subsampling {n_cells_sample}/{ad.n_obs} cells "
        f"(seed={seed})",
        flush=True,
    )
    return ad[idx, :].copy()


def main() -> None:
    p = argparse.ArgumentParser(
        description="PCA→Diffmap and EGGFM→Diffmap dimension reductions."
    )
    p.add_argument(
        "--params",
        required=True,
        help="Path to configs/params.yml",
    )
    p.add_argument(
        "--ad",
        required=True,
        help="QC .h5ad (e.g. data/interim/weinreb_qc.h5ad)",
    )
    p.add_argument(
        "--energy-ckpt",
        required=True,
        help="EGGFM checkpoint (e.g. out/models/eggfm/eggfm_energy_weinreb.pt)",
    )
    p.add_argument(
        "--out",
        required=True,
        help=(
            "Base output .h5ad (e.g. data/interim/weinreb_embedding.h5ad). "
            "Also writes *_pca_diffmap.h5ad and *_eggfm_diffmap.h5ad."
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of diffusion components to keep (default: 10)",
    )
    p.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="k for kNN graph in PCA space (default: 30)",
    )
    p.add_argument(
        "--n-cells-sample",
        type=int,
        default=0,
        help="Optional cell subsample size (0 = use all cells)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for subsampling / torch (default: 7)",
    )
    p.add_argument(
        "--cfg-key",
        default="eggfm_diffmap",
        help="YAML block key for diffusion config (default: eggfm_diffmap)",
    )
    args = p.parse_args()

    # ---------- seeds ----------
    if int(args.seed) is not None:
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    # ---------- params + config ----------
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())
    diff_cfg = diffusion_config_from_params(params, key=args.cfg_key)

    # ---------- load QC AnnData ----------
    ad_full = sc.read_h5ad(args.ad)
    print(
        f"[EMBEDDING] Loaded QC AnnData: {ad_full.n_obs} cells × {ad_full.n_vars} genes",
        flush=True,
    )

    # work on a copy (and optionally subsample)
    ad = maybe_subsample(
        ad_full,
        n_cells_sample=int(args.n_cells_sample),
        seed=int(int(args.seed)),
    )

    # ---------- PCA → Diffmap ----------
    build_pca_diffmap(ad, k=int(args.k), n_neighbors=int(args.n_neighbors))

    # ---------- EGGFM model ----------
    ckpt = torch.load(args.energy_ckpt, map_location="cpu")
    model_cfg_dict = ckpt.get("model_cfg", {})
    n_genes = int(ckpt.get("n_genes", ad.n_vars))

    model_cfg = EnergyModelConfig(**model_cfg_dict)
    energy_model = EnergyMLP(n_genes=n_genes, hidden_dims=model_cfg.hidden_dims)
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

    # First k dims of the EGGFM diffusion embedding
    X_diff_eggfm = ad.obsm["X_eggfm_diffmap"][:, : int(args.k)].copy()
    ad.obsm["X_diff_eggfm"] = X_diff_eggfm

    # ---------- compute output paths ----------
    out_base = Path(args.out)
    if out_base.suffix != ".h5ad":
        out_base = out_base.with_suffix(".h5ad")

    out_dir = out_base.parent
    stem = out_base.stem

    out_combined = out_base
    out_pca = out_dir / f"{stem}_pca_diffmap.h5ad"
    out_eggfm = out_dir / f"{stem}_eggfm_diffmap.h5ad"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- write combined ----------
    ad.write_h5ad(out_combined)
    print(f"[EMBEDDING] Wrote combined embeddings to {out_combined}")

    # ---------- write PCA-only ----------
    ad_pca = ad.copy()
    for key in ("X_eggfm_diffmap", "X_diff_eggfm"):
        if key in ad_pca.obsm:
            del ad_pca.obsm[key]
    ad_pca.write_h5ad(out_pca)
    print(f"[EMBEDDING] Wrote PCA→Diffmap embedding to {out_pca}")

    # ---------- write EGGFM-only ----------
    ad_eggfm = ad.copy()
    if "X_diff_pca" in ad_eggfm.obsm:
        del ad_eggfm.obsm["X_diff_pca"]
    ad_eggfm.write_h5ad(out_eggfm)
    print(f"[EMBEDDING] Wrote EGGFM→Diffmap embedding to {out_eggfm}")

    # ---------- write combined ----------
    ad.write_h5ad(out_combined)
    print(f"[EMBEDDING] Wrote combined embeddings to {out_combined}")

    # ---------- write PCA-only ----------
    ad_pca = ad.copy()
    for key in ("X_eggfm_diffmap", "X_diff_eggfm"):
        if key in ad_pca.obsm:
            del ad_pca.obsm[key]
    ad_pca.write_h5ad(out_pca)
    print(f"[EMBEDDING] Wrote PCA→Diffmap embedding to {out_pca}")

    # ---------- write EGGFM-only ----------
    ad_eggfm = ad.copy()
    if "X_diff_pca" in ad_eggfm.obsm:
        del ad_eggfm.obsm["X_diff_pca"]
    ad_eggfm.write_h5ad(out_eggfm)
    print(f"[EMBEDDING] Wrote EGGFM→Diffmap embedding to {out_eggfm}")

    # ---------- manifest ----------
    manifest_path = write_embedding_manifest(
        qc_path=Path(args.ad),
        out_path=out_path,
        pca_path=out_path_pca,      # or None if you pack both in one file
        eggfm_path=out_path_eggfm,  # or out_path if everything is in that file
        params=params,
        params_path=Path(args.params),
        cfg_key=args.cfg_key,
        k=args.k,
        n_neighbors=args.n_neighbors,
        n_cells_sample=args.n_cells_sample,
        seed=args.seed,
    )
    print(f"[EMBEDDING] Wrote manifest to {manifest_path}")

if __name__ == "__main__":
    main()
