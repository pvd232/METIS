#!/usr/bin/env python3
"""
scripts/ti_eval_extended.py

Drop-in trajectory evaluation with:
  - Diffusion pseudotime metrics (variance, range, Spearman/Kendall vs time)
  - A Waddington-OT–style interpolation error metric, computed *in the
    chosen embedding space*.
  - Clone-aware pseudotime metrics using ad.obsm["X_clone_membership"].

Usage
-----
  python scripts/ti_eval_extended.py --params configs/some_run.yml

Expected YAML structure (minimal)
---------------------------------
root:
  seed: 11                      # optional; used for subsampling
  eggfm_train:                  # optional; only used to log training params
    sigma: 0.15
    lr: 0.005
    batch_size: 8192
    riemann_reg_type: hess_smooth
    riemann_reg_weight: 0.1

  eggfm_diffmap:                # optional; used to log metric hyperparams
    metric_mode: riem_normal    # or riem_curvature, scm, baseline_*, ...
    geometry_source: pca
    energy_source: hvg
    t: 3.0
    distance_power: 2.0
    metric_gamma: 5.0
    metric_lambda: 20.0
    energy_clip_abs: 6.0
    energy_batch_size: 8192
    tangent_dim: 10
    tangent_eps: 0.01
    tangent_k: 30
    curvature_k: 30
    curvature_scale: 1.0
    normal_k: 30
    normal_weight: 1.0

  ti_eval:
    ad_path: data/embedding/weinreb_eggfm_diffmap.h5ad
    embedding_key: X_eggfm_diffmap
    time_key: Time point          # or "time_point" if you use WEINREB.time_key
    cluster_key: Cell type annotation
    fate_key: null                # currently unused
    baseline_embedding_key: null
    root_mask_key: null
    n_neighbors: 30
    n_dcs: 10
    max_cells: 60000
    out_dir: out/metrics/ti

    # Optional IDs / modes
    run_id: weinreb_eggfm_diffmap__X_eggfm_diffmap__r62
    base_run_id: weinreb_eggfm_diffmap__X_eggfm_diffmap
    metric_mode: riem_normal      # used when eggfm_diffmap.metric_mode is absent

    # Optional: clone membership key. If omitted, defaults to "X_clone_membership"
    # when present in ad.obsm.
    clone_membership_key: X_clone_membership
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from scipy import sparse
from scipy.stats import kendalltau, spearmanr


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
 
import numpy as np
import pandas as pd
import scanpy as sc

def ensure_dpt_pseudotime(
    ad,
    embedding_key: str,
    time_key: str,
    n_neighbors: int = 30,
    n_dcs: int = 10,
) -> None:
    """
    Make sure ad.obs['dpt_pseudotime'] exists.

    If missing, compute neighbors + diffusion map on the given embedding,
    pick a root cell from the earliest time point, and run scanpy.tl.dpt.
    """

    if "dpt_pseudotime" in ad.obs.columns:
        return

    # 1) Build graph on the embedding
    sc.pp.neighbors(ad, use_rep=embedding_key, n_neighbors=n_neighbors)
    sc.tl.diffmap(ad, n_comps=n_dcs)

    # 2) Turn time_key into numeric values
    if time_key not in ad.obs.columns:
        raise KeyError(
            f"time_key='{time_key}' not found in ad.obs. "
            f"Available columns: {list(ad.obs.columns)}"
        )

    time_raw = ad.obs[time_key]
    time_num = pd.to_numeric(time_raw, errors="coerce")

    if time_num.isna().all():
        # Fallback: use categorical codes if time is not numeric
        time_num = time_raw.astype("category").cat.codes.astype(float)

    # 3) Choose root as earliest time point
    root_idx = int(np.argmin(time_num.values))
    ad.uns["iroot"] = root_idx

    # 4) Run DPT
    sc.tl.dpt(ad)

def pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances between rows of a and b.

    a: [n, d], b: [m, d] -> [n, m]
    """
    diff = a[:, None, :] - b[None, :, :]
    return np.einsum("ijk,ijk->ij", diff, diff)


def compute_pseudotime_metrics(
    adata,
    embedding_key: str,
    time_key: str,
    n_neighbors: int,
    n_dcs: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Compute Diffusion Pseudotime (DPT) on top of the provided embedding,
    and correlate with the experimental time key.

    Returns
    -------
    metrics: dict
        Dict with PT variance, min, max, Spearman/Kendall vs time.
    pt: np.ndarray
        Pseudotime for each cell, shape [n_cells].
    time_numeric: np.ndarray
        Numeric encoding of experimental time, shape [n_cells].
    """
    sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=n_neighbors)
    sc.tl.diffmap(adata, n_comps=n_dcs)
    sc.tl.dpt(adata, n_dcs=n_dcs)
    bb = adata.obs
    for col in bb:
        print("col", col)
        
    pt = adata.obs["dpt_pseudotime"].to_numpy()
    pt_var = float(np.var(pt))
    pt_min = float(np.min(pt))
    pt_max = float(np.max(pt))

    t_raw = adata.obs[time_key]
    if np.issubdtype(t_raw.dtype, np.number):
        time_numeric = t_raw.to_numpy().astype(float)
    else:
        cats = pd.Categorical(t_raw)
        time_numeric = cats.codes.astype(float)

    rho, _ = spearmanr(pt, time_numeric)
    tau, _ = kendalltau(pt, time_numeric)

    metrics = {
        "pt_variance": float(pt_var),
        "pt_min": float(pt_min),
        "pt_max": float(pt_max),
        "pt_spearman_vs_time": float(rho),
        "pt_kendall_vs_time": float(tau),
    }
    return metrics, pt, time_numeric


def compute_wot_style_interp_error(
    adata,
    embedding_key: str,
    time_key: str,
    max_cells_per_time: int = 1000,
    seed: int = 0,
) -> float:
    """
    Waddington-OT–style interpolation error computed *in embedding space*.

    Idea (simplified):
      - Treat each adjacent triplet of timepoints (t0, t1, t2).
      - Using the embedding as geometry, construct a soft OT-like coupling
        between t0 and t2 via an RBF kernel (no external OT dependency).
      - For each cell x_i at t0, compute a barycentric target ŷ_i at t2.
      - Interpolate halfway along the geodesic: ẑ_i = 0.5 * x_i + 0.5 * ŷ_i.
      - Compare the cloud {ẑ_i} to the actual cells at t1 via symmetric NN MSE.
      - Average this error across all valid (t0, t1, t2) triplets.

    Returns
    -------
    float
        Mean symmetric NN MSE; lower is better. np.nan if not computable.
    """
    obs_time = adata.obs[time_key]
    if obs_time.nunique() < 3:
        return float("nan")

    cats = pd.Categorical(obs_time)
    levels = list(cats.categories)
    time_values = obs_time.to_numpy()

    Z = adata.obsm[embedding_key].astype(np.float64)
    rng = np.random.default_rng(seed)

    # Map each time level to indices
    time_to_idx = {lvl: np.where(time_values == lvl)[0] for lvl in levels}

    errors = []

    for i in range(len(levels) - 2):
        t0, t1, t2 = levels[i], levels[i + 1], levels[i + 2]
        idx0 = time_to_idx[t0]
        idx1 = time_to_idx[t1]
        idx2 = time_to_idx[t2]

        if min(len(idx0), len(idx1), len(idx2)) < 20:
            # Too few cells in at least one timepoint
            continue

        # Subsample to keep OT-like computations manageable
        n = min(len(idx0), len(idx2), max_cells_per_time)
        idx0_sub = rng.choice(idx0, size=n, replace=False)
        idx2_sub = rng.choice(idx2, size=n, replace=False)

        Z0 = Z[idx0_sub]  # [n, d]
        Z2 = Z[idx2_sub]  # [n, d]

        # Cost matrix in embedding space
        C02 = pairwise_sq_dists(Z0, Z2)  # [n, n]

        med = np.median(C02)
        if med <= 0:
            continue

        K = np.exp(-C02 / (2.0 * med))
        row_sums = K.sum(axis=1, keepdims=True)
        K = np.divide(
            K,
            row_sums,
            out=np.zeros_like(K),
            where=row_sums > 0,
        )

        # Barycentric targets at t2
        Y_hat = K @ Z2  # [n, d]

        # Half-way interpolation along OT geodesic
        alpha = 0.5
        Z_hat = (1.0 - alpha) * Z0 + alpha * Y_hat  # [n, d]

        # Subsample t1 as comparison cloud
        m = min(len(idx1), n, max_cells_per_time)
        idx1_sub = rng.choice(idx1, size=m, replace=False)
        Z1 = Z[idx1_sub]  # [m, d]

        # Symmetric NN MSE between Z_hat and Z1
        D = pairwise_sq_dists(Z_hat, Z1)  # [n, m]
        err1 = float(D.min(axis=1).mean())
        err2 = float(D.min(axis=0).mean())
        errors.append(0.5 * (err1 + err2))

    if not errors:
        return float("nan")

    return float(np.mean(errors))


def compute_clone_metrics(
    adata,
    pt: np.ndarray,
    time_numeric: np.ndarray,
    clone_membership_key: str,
    min_cells_per_clone: int = 5,
) -> Dict[str, float]:
    """
    Clone-aware pseudotime metrics using ad.obsm[clone_membership_key].

    For each clone with at least min_cells_per_clone cells:
      - Compute variance of pseudotime within the clone.
      - Compute Kendall tau between pseudotime and experimental time
        *within the clone*.
    Aggregate these across clones via mean/median.

    Returns
    -------
    dict with:
      - clone_pt_var_within_mean
      - clone_pt_var_within_median
      - clone_pt_var_within_mean_norm (divided by global pt variance)
      - clone_pt_var_within_median_norm
      - clone_pt_kendall_vs_time_mean
      - clone_pt_kendall_vs_time_median
    """
    if clone_membership_key not in adata.obsm:
        return {
            "clone_pt_var_within_mean": None,
            "clone_pt_var_within_median": None,
            "clone_pt_var_within_mean_norm": None,
            "clone_pt_var_within_median_norm": None,
            "clone_pt_kendall_vs_time_mean": None,
            "clone_pt_kendall_vs_time_median": None,
        }

    M = adata.obsm[clone_membership_key]
    n_cells = adata.n_obs

    if sparse.issparse(M):
        M_csc = M.tocsc()
        _, n_clones = M_csc.shape
        get_indices = lambda j: M_csc.indices[M_csc.indptr[j] : M_csc.indptr[j + 1]]
    else:
        M_arr = np.asarray(M)
        if M_arr.shape[0] != n_cells:
            raise ValueError(
                f"Clone membership matrix has shape {M_arr.shape}, "
                f"but AnnData has {n_cells} cells."
            )
        _, n_clones = M_arr.shape

        def get_indices(j):
            return np.where(M_arr[:, j] > 0)[0]

    global_pt_var = float(np.var(pt)) if n_cells > 1 else 0.0

    vars_within = []
    taus = []

    for j in range(n_clones):
        idx = np.array(get_indices(j), dtype=int)
        if idx.size < min_cells_per_clone:
            continue

        pt_j = pt[idx]
        t_j = time_numeric[idx]

        if np.allclose(pt_j, pt_j[0]):
            # Zero variance, skip tau (undefined)
            var_j = 0.0
            tau_j = np.nan
        else:
            var_j = float(np.var(pt_j))
            if np.allclose(t_j, t_j[0]):
                tau_j = np.nan
            else:
                tau_j, _ = kendalltau(pt_j, t_j)

        vars_within.append(var_j)
        if np.isfinite(tau_j):
            taus.append(float(tau_j))

    if not vars_within:
        return {
            "clone_pt_var_within_mean": None,
            "clone_pt_var_within_median": None,
            "clone_pt_var_within_mean_norm": None,
            "clone_pt_var_within_median_norm": None,
            "clone_pt_kendall_vs_time_mean": None,
            "clone_pt_kendall_vs_time_median": None,
        }

    vars_within_arr = np.array(vars_within, dtype=float)
    mean_var = float(vars_within_arr.mean())
    median_var = float(np.median(vars_within_arr))

    if global_pt_var > 0:
        mean_var_norm = mean_var / global_pt_var
        median_var_norm = median_var / global_pt_var
    else:
        mean_var_norm = float("nan")
        median_var_norm = float("nan")

    if taus:
        taus_arr = np.array(taus, dtype=float)
        mean_tau = float(taus_arr.mean())
        median_tau = float(np.median(taus_arr))
    else:
        mean_tau = float("nan")
        median_tau = float("nan")

    return {
        "clone_pt_var_within_mean": mean_var,
        "clone_pt_var_within_median": median_var,
        "clone_pt_var_within_mean_norm": mean_var_norm,
        "clone_pt_var_within_median_norm": median_var_norm,
        "clone_pt_kendall_vs_time_mean": mean_tau,
        "clone_pt_kendall_vs_time_median": median_tau,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extended TI eval (DPT + WOT-style + clone-aware metrics)."
    )
    p.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to YAML config file (same style as other MEDIT scripts).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.params)

    with cfg_path.open("r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    seed = int(cfg.get("seed", 0))

    t_cfg: Dict[str, Any] = cfg["ti_eval"]
    diff_cfg: Dict[str, Any] = cfg.get("eggfm_diffmap", {})
    train_cfg: Dict[str, Any] = cfg.get("eggfm_train", {})

    ad_path = t_cfg["ad_path"]
    embedding_key = t_cfg["embedding_key"]
    time_key = t_cfg["time_key"]

    n_neighbors = int(t_cfg.get("n_neighbors", 30))
    n_dcs = int(t_cfg.get("n_dcs", 10))
    max_cells_raw = t_cfg.get("max_cells", 0)
    max_cells = int(max_cells_raw) if max_cells_raw is not None else 0

    out_dir = Path(t_cfg.get("out_dir", "out/metrics/ti"))

    # Clone membership key (optional)
    clone_membership_key = t_cfg.get("clone_membership_key", "X_clone_membership")

    # Base/run IDs: allow config override; otherwise derive from path + embedding
    base_run_id = t_cfg.get("base_run_id")
    if base_run_id is None:
        base_run_id = f"{Path(ad_path).stem}__{embedding_key}"

    run_id = t_cfg.get("run_id", base_run_id)

    # Metric mode / geometry / energy
    metric_mode = diff_cfg.get("metric_mode", t_cfg.get("metric_mode", "baseline"))
    geometry_source = diff_cfg.get("geometry_source", t_cfg.get("geometry_source"))
    energy_source = diff_cfg.get("energy_source", t_cfg.get("energy_source"))

    # Load embedding AnnData
    sc.settings.verbosity = 0
    ad = sc.read_h5ad(ad_path)
    ensure_dpt_pseudotime(ad, embedding_key, time_key, n_neighbors, n_dcs)

    # Optional subsampling for eval
    rng = np.random.default_rng(seed)
    if max_cells > 0 and ad.n_obs > max_cells:
        idx = rng.choice(ad.n_obs, size=max_cells, replace=False)
        ad = ad[idx].copy()

    eval_n_cells = int(ad.n_obs)

    if embedding_key not in ad.obsm:
        raise KeyError(
            f"Embedding key '{embedding_key}' not found in ad.obsm. "
            f"Available keys: {list(ad.obsm.keys())}"
        )

    # --- 1) Pseudotime metrics
    pt_metrics, pt, time_numeric = compute_pseudotime_metrics(
        ad,
        embedding_key=embedding_key,
        time_key=time_key,
        n_neighbors=n_neighbors,
        n_dcs=n_dcs,
    )

    # --- 2) WOT-style interpolation error
    wot_err = compute_wot_style_interp_error(
        ad,
        embedding_key=embedding_key,
        time_key=time_key,
        max_cells_per_time=min(1000, eval_n_cells),
        seed=seed,
    )

    # --- 3) Clone-aware pseudotime metrics (Weinreb-style lineage check)
    clone_metrics = compute_clone_metrics(
        ad,
        pt=pt,
        time_numeric=time_numeric,
        clone_membership_key=clone_membership_key,
        min_cells_per_clone=5,
    )

    # Collect metadata + metrics in the same style as your existing CSV
    metrics: Dict[str, Any] = {
        "metrics_path": None,  # filled in below
        "run_id": run_id,
        "base_run_id": base_run_id,
        "ad_path": ad_path,
        "embedding_key": embedding_key,
        "eval_n_cells": eval_n_cells,
        "n_neighbors": n_neighbors,
        "n_dcs": n_dcs,
        "max_cells": max_cells if max_cells > 0 else eval_n_cells,
        "metric_mode": metric_mode,
        "geometry_source": geometry_source,
        "energy_source": energy_source,
        "t": diff_cfg.get("t"),
        "distance_power": diff_cfg.get("distance_power"),
        "metric_gamma": diff_cfg.get("metric_gamma"),
        "metric_lambda": diff_cfg.get("metric_lambda"),
        "energy_clip_abs": diff_cfg.get("energy_clip_abs"),
        "energy_batch_size": diff_cfg.get("energy_batch_size"),
        "tangent_dim": diff_cfg.get("tangent_dim"),
        "tangent_eps": diff_cfg.get("tangent_eps"),
        "tangent_k": diff_cfg.get("tangent_k"),
        "curvature_k": diff_cfg.get("curvature_k"),
        "curvature_scale": diff_cfg.get("curvature_scale"),
        "normal_k": diff_cfg.get("normal_k"),
        "normal_weight": diff_cfg.get("normal_weight"),
        "train_sigma": train_cfg.get("sigma"),
        "train_lr": train_cfg.get("lr"),
        "train_batch_size": train_cfg.get("batch_size"),
        "train_riemann_reg_type": train_cfg.get("riemann_reg_type"),
        "train_riemann_reg_weight": train_cfg.get("riemann_reg_weight"),
    }

    # Add PT metrics
    metrics.update(pt_metrics)

    # Add WOT-style metric
    metrics["wot_interp_mse"] = float(wot_err) if np.isfinite(wot_err) else None

    # Add clone-aware metrics
    metrics.update(clone_metrics)

    # Write out
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"

    metrics["metrics_path"] = str(metrics_path)

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
