#!/usr/bin/env python3
"""
scripts/run_baseline_manifolds.py

Run non-EGGFM baseline embeddings (PCA, DiffMap, PHATE, scVI) on the
Weinreb QC dataset, at the same cell-count regimes as your EGGFM
ablations (60k / 15k / 5k), and evaluate with scripts/ti_eval.py.

Outputs metrics under out/metrics/ti/** just like EGGFM runs, with
metric_mode tagged as "baseline_{method}" so your summarizer can pick
them up cleanly.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess

import numpy as np
import scanpy as sc
import yaml

# Optional methods: handled via lazy import
HAS_PHATE = True
try:
    import phate  # type: ignore
except Exception:
    HAS_PHATE = False

HAS_SCVI = True
try:
    import scvi  # type: ignore
except Exception:
    HAS_SCVI = False


# =============================================================================
#  CONFIG
# =============================================================================

# Cell-count regimes, mirroring your EGGFM ablations
CELL_REGIMES = {
    "global": 60000,
    "meso": 15000,
    "local": 5000,
}

# Minimal ti_eval config template, matching your existing setup
BASE_CFG = {
    "ti_eval": {
        "ad_path": "",               # filled in per run
        "embedding_key": "",         # filled in per run
        "time_key": "Time point",
        "cluster_key": "Cell type annotation",
        "fate_key": None,
        "baseline_embedding_key": None,
        "root_mask_key": None,
        "n_neighbors": 30,
        "n_dcs": 10,
        "max_cells": 60000,         # overwritten per run
        "out_dir": "out/metrics/ti",
    },
    # just to keep your summarizer happy; we use metric_mode as "baseline_*"
    "eggfm_diffmap": {
        "metric_mode": "baseline",
    },
}

RUN_DIR = Path("configs/ablation_baselines")
RUN_DIR.mkdir(parents=True, exist_ok=True)


def sh(cmd: str):
    print(f"\n🔥 Running: {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# =============================================================================
#  EMBEDDING HELPERS
# =============================================================================

def subsample_ad(ad, n_cells: int, seed: int = 11):
    """Subsample adata to n_cells without replacement (or return full if smaller)."""
    if n_cells <= 0 or n_cells >= ad.n_obs:
        return ad.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(ad.n_obs, size=n_cells, replace=False)
    return ad[idx].copy()


def embed_pca(ad):
    """Compute PCA embedding; returns embedding key."""
    if "X_pca" not in ad.obsm:
        sc.tl.pca(ad, n_comps=30, use_highly_variable=True, svd_solver="arpack")
    return "X_pca"


def embed_diffmap(ad):
    """Diffusion Map using PCA as input; returns embedding key."""
    if "X_pca" not in ad.obsm:
        sc.tl.pca(ad, n_comps=30, use_highly_variable=True, svd_solver="arpack")
    sc.pp.neighbors(ad, use_rep="X_pca", n_neighbors=30)
    sc.tl.diffmap(ad)
    return "X_diffmap"


def embed_phate(ad):
    """PHATE embedding; returns embedding key."""
    if not HAS_PHATE:
        raise RuntimeError("phate is not installed; pip install phate to use this baseline.")
    op = phate.PHATE(n_components=10)
    ad.obsm["X_phate"] = op.fit_transform(ad.X)
    return "X_phate"


def embed_scvi(ad):
    """scVI latent embedding; returns embedding key."""
    if not HAS_SCVI:
        raise RuntimeError("scvi-tools is not installed; pip install scvi-tools to use this baseline.")

    # Setup; adjust batch_key if needed (e.g. "batch" in .obs)
    scvi.model.SCVI.setup_anndata(ad)
    model = scvi.model.SCVI(ad, n_latent=30)
    model.train(max_epochs=200)
    ad.obsm["X_scvi"] = model.get_latent_representation()
    return "X_scvi"


METHODS = {
    "pca": embed_pca,
    "diffmap": embed_diffmap,
    "phate": embed_phate,
    "scvi": embed_scvi,
}


# =============================================================================
#  MAIN DRIVER
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adata",
        type=str,
        default="data/interim/weinreb_qc.h5ad",
        help="Path to QC'ed Weinreb AnnData (log-normalized, HVGs ready).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["pca", "diffmap", "scvi"],
        help="Which baselines to run: subset of {pca, diffmap, phate, scvi}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed for subsampling.",
    )
    args = parser.parse_args()

    ad_full = sc.read_h5ad(args.adata)
    print(f"Loaded {args.adata} with {ad_full.n_obs} cells and {ad_full.n_vars} genes.")

    for method in args.methods:
        if method not in METHODS:
            print(f"⚠️ Skipping unknown method '{method}'")
            continue

        embed_fn = METHODS[method]

        for regime, target_n in CELL_REGIMES.items():
            print(f"\n=== Method={method}  Regime={regime}  N={target_n} ===")

            # 1) Subsample
            ad_sub = subsample_ad(ad_full, n_cells=target_n, seed=args.seed)
            n_actual = ad_sub.n_obs
            print(f"Subsampled to {n_actual} cells.")

            # 2) Compute embedding
            emb_key = embed_fn(ad_sub)
            print(f"Embedding computed: obsm['{emb_key}'] with shape {ad_sub.obsm[emb_key].shape}")

            # 3) Save AnnData
            method_tag = method.lower()
            out_ad_path = Path("data/embedding") / f"weinreb_{method_tag}_{regime}_n{n_actual}.h5ad"
            out_ad_path.parent.mkdir(parents=True, exist_ok=True)
            ad_sub.write_h5ad(out_ad_path)
            print(f"Saved embedding AnnData to {out_ad_path}")

            # 4) Build ti_eval config
            cfg = deepcopy(BASE_CFG)
            cfg["ti_eval"]["ad_path"] = str(out_ad_path)
            cfg["ti_eval"]["embedding_key"] = emb_key
            cfg["ti_eval"]["max_cells"] = int(n_actual)

            # Tag method for summarizer
            cfg["eggfm_diffmap"]["metric_mode"] = f"baseline_{method_tag}"

            # Optional: if your ti_eval expects these for naming
            cfg["run_id"] = f"weinreb_{method_tag}_{regime}__{emb_key}"
            cfg["base_run_id"] = cfg["run_id"]

            cfg_path = RUN_DIR / f"ti_baseline_{method_tag}_{regime}_n{n_actual}.yml"
            cfg_path.write_text(yaml.dump(cfg, sort_keys=False))
            print(f"Wrote TI config to {cfg_path}")

            # 5) Run TI evaluation
            sh(f"python scripts/ti_eval.py --params {cfg_path}")


if __name__ == "__main__":
    main()
