# src/medit/ti/eval.py

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import numpy as np
import scanpy as sc
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


def compute_pseudotime_metrics(
    ad: sc.AnnData,
    embedding_key: str,
    time_key: Optional[str] = None,
    root_mask_key: Optional[str] = None,
    n_neighbors: int = 30,
    n_dcs: int = 10,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Compute diffusion pseudotime on a given embedding, and compare it
    to a ground-truth temporal variable if available.

    Parameters
    ----------
    ad
        AnnData with obsm[embedding_key].
    embedding_key
        Key in ad.obsm for the embedding to evaluate.
    time_key
        Optional obs column with ground-truth time / stage values.
    root_mask_key
        Optional boolean obs column to choose root cells.
        If provided, the first True cell is used as DPT root.
        If None, and time_key is provided, the earliest time is used.
        Otherwise, cell 0 is used.
    n_neighbors
        Number of neighbors for kNN graph.
    n_dcs
        Number of diffusion components to use for DPT.

    Returns
    -------
    metrics
        Dictionary of pseudotime-related metrics.
    pt
        Array of pseudotime values, shape [n_cells].
    """
    ad = ad.copy()  # avoid mutating caller
    if embedding_key not in ad.obsm:
        raise KeyError(f"Embedding key '{embedding_key}' not found in ad.obsm")

    ad.obsm["X_use_ti"] = ad.obsm[embedding_key]

    # Build kNN graph
    sc.pp.neighbors(ad, use_rep="X_use_ti", n_neighbors=n_neighbors)

    # Diffusion maps + DPT
    sc.tl.diffmap(ad)
    if root_mask_key is not None:
        if root_mask_key not in ad.obs:
            raise KeyError(f"root_mask_key '{root_mask_key}' not in ad.obs")
        roots = np.where(ad.obs[root_mask_key].values.astype(bool))[0]
        if roots.size == 0:
            raise ValueError(
                f"root_mask_key '{root_mask_key}' has no True entries."
            )
        root_idx = int(roots[0])
    else:
        if time_key is not None and time_key in ad.obs:
            t_vals = ad.obs[time_key].to_numpy()
            root_idx = int(np.nanargmin(t_vals))
        else:
            root_idx = 0
    ad.uns['iroot'] = root_idx
    sc.tl.dpt(ad, n_dcs=n_dcs)
    pt = ad.obs["dpt_pseudotime"].to_numpy()

    metrics: Dict[str, float] = {
        "pt_variance": float(np.nanvar(pt)),
        "pt_min": float(np.nanmin(pt)),
        "pt_max": float(np.nanmax(pt)),
    }

    if time_key is not None:
        if time_key not in ad.obs:
            raise KeyError(f"time_key '{time_key}' not found in ad.obs")
        t_true = ad.obs[time_key].to_numpy().astype(float)
        valid = np.isfinite(t_true) & np.isfinite(pt)
        if valid.sum() > 10:
            rho, _ = spearmanr(t_true[valid], pt[valid])
            tau, _ = kendalltau(t_true[valid], pt[valid])
            metrics["pt_spearman_vs_time"] = float(rho)
            metrics["pt_kendall_vs_time"] = float(tau)
        else:
            metrics["pt_spearman_vs_time"] = np.nan
            metrics["pt_kendall_vs_time"] = np.nan

    return metrics, pt


def compute_branch_metrics(
    ad: sc.AnnData,
    embedding_key: str,
    cluster_key: str,
    fate_key: str,
    n_neighbors: int = 30,
) -> Dict[str, float]:
    """
    Simple branch / fate metric: how well do clusters derived from
    this embedding agree with known fate labels?

    This is not a full TI evaluation, but gives a quick sense of whether
    the embedding separates fates cleanly.

    Parameters
    ----------
    ad
        AnnData with obsm[embedding_key] and obs[cluster_key], obs[fate_key].
    embedding_key
        Key in ad.obsm used to build kNN graph (for PAGA, if desired).
    cluster_key
        obs column with discrete cluster labels (e.g. Louvain).
    fate_key
        obs column with fate / lineage labels.

    Returns
    -------
    metrics
        Dictionary with ARI(cluster, fate) and basic checks.
    """
    ad = ad.copy()
    if embedding_key not in ad.obsm:
        raise KeyError(f"Embedding key '{embedding_key}' not found in ad.obsm")
    if cluster_key not in ad.obs:
        raise KeyError(f"cluster_key '{cluster_key}' not found in ad.obs")
    if fate_key not in ad.obs:
        raise KeyError(f"fate_key '{fate_key}' not found in ad.obs")

    ad.obsm["X_use_ti"] = ad.obsm[embedding_key]
    sc.pp.neighbors(ad, use_rep="X_use_ti", n_neighbors=n_neighbors)

    # Optional: compute PAGA on these clusters (for downstream plotting / analysis)
    sc.tl.paga(ad, groups=cluster_key)

    y_cluster = ad.obs[cluster_key].astype("category").cat.codes.to_numpy()
    y_fate = ad.obs[fate_key].astype("category").cat.codes.to_numpy()

    valid = np.isfinite(y_cluster) & np.isfinite(y_fate)
    if valid.sum() == 0:
        ari_cf = np.nan
    else:
        ari_cf = adjusted_rand_score(y_fate[valid], y_cluster[valid])

    metrics: Dict[str, float] = {
        "branch_ari_cluster_vs_fate": float(ari_cf),
        "branch_n_clusters": int(np.unique(y_cluster[valid]).size),
        "branch_n_fates": int(np.unique(y_fate[valid]).size),
    }
    return metrics


def knn_graph_overlap(
    X_base: np.ndarray,
    X_embed: np.ndarray,
    n_neighbors: int = 30,
) -> Dict[str, float]:
    """
    Quantify how much the local kNN graph has changed between a baseline
    embedding and a candidate embedding, using Jaccard overlap of
    neighbor sets.

    Parameters
    ----------
    X_base
        Baseline embedding, shape [n_cells, d_base].
    X_embed
        Candidate embedding, shape [n_cells, d_embed].
    n_neighbors
        k for kNN (excluding self).

    Returns
    -------
    metrics
        mean and std of per-cell Jaccard overlap.
    """
    if X_base.shape[0] != X_embed.shape[0]:
        raise ValueError(
            f"X_base and X_embed must have same number of rows; "
            f"got {X_base.shape[0]} and {X_embed.shape[0]}"
        )

    nn_base = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_base)
    nn_emb = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_embed)

    _, idx_base = nn_base.kneighbors(X_base)
    _, idx_emb = nn_emb.kneighbors(X_embed)

    idx_base = idx_base[:, 1:]  # drop self
    idx_emb = idx_emb[:, 1:]

    overlaps = []
    for nb, ne in zip(idx_base, idx_emb):
        set_b = set(nb.tolist())
        set_e = set(ne.tolist())
        inter = len(set_b & set_e)
        union = len(set_b | set_e)
        if union > 0:
            overlaps.append(inter / union)

    overlaps_arr = np.asarray(overlaps, dtype=float)
    return {
        "graph_knn_jaccard_mean": float(np.nanmean(overlaps_arr)),
        "graph_knn_jaccard_std": float(np.nanstd(overlaps_arr)),
    }


def evaluate_embedding_for_ti(
    ad: sc.AnnData,
    embedding_key: str,
    time_key: Optional[str] = None,
    cluster_key: Optional[str] = None,
    fate_key: Optional[str] = None,
    baseline_embedding_key: Optional[str] = None,
    root_mask_key: Optional[str] = None,
    n_neighbors: int = 30,
    n_dcs: int = 10,
) -> Dict[str, Any]:
    """
    High-level TI evaluation entry point.

    For a given embedding, compute:
      - pseudotime metrics (variance, correlation with time),
      - optional branch/fate metrics,
      - optional kNN graph overlap vs a baseline embedding.

    Parameters
    ----------
    ad
        AnnData with obsm[embedding_key].
    embedding_key
        Key in ad.obsm to evaluate (e.g. "X_diff_pca", "X_eggfm_diffmap").
    time_key
        Optional obs column with ground-truth time or stage.
    cluster_key
        Optional obs column with cluster labels for this embedding.
    fate_key
        Optional obs column with fate / lineage labels.
    baseline_embedding_key
        Optional obsm key for a baseline embedding to compare kNN graph against.
    root_mask_key
        Optional obs boolean column marking candidate root cells.
    n_neighbors
        k for kNN graph construction.
    n_dcs
        Number of diffusion components for DPT.

    Returns
    -------
    metrics
        Dictionary aggregating all available TI-relevant metrics.
    """
    metrics: Dict[str, Any] = {}

    # 1) Pseudotime metrics
    pt_metrics, pt = compute_pseudotime_metrics(
        ad,
        embedding_key=embedding_key,
        time_key=time_key,
        root_mask_key=root_mask_key,
        n_neighbors=n_neighbors,
        n_dcs=n_dcs,
    )
    metrics.update(pt_metrics)
    metrics["pt_values"] = pt  # optional: caller can drop this if undesired

    # 2) Branch / fate metrics
    if cluster_key is not None and fate_key is not None:
        branch_metrics = compute_branch_metrics(
            ad,
            embedding_key=embedding_key,
            cluster_key=cluster_key,
            fate_key=fate_key,
            n_neighbors=n_neighbors,
        )
        metrics.update(branch_metrics)

    # 3) Graph overlap vs baseline
    if baseline_embedding_key is not None:
        if baseline_embedding_key not in ad.obsm:
            raise KeyError(
                f"baseline_embedding_key '{baseline_embedding_key}' not found in ad.obsm"
            )
        X_base = ad.obsm[baseline_embedding_key]
        X_emb = ad.obsm[embedding_key]
        graph_metrics = knn_graph_overlap(
            X_base=X_base,
            X_embed=X_emb,
            n_neighbors=n_neighbors,
        )
        metrics.update(graph_metrics)

    return metrics
