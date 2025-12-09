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
    """
    print("[TI.DEBUG] ---- compute_pseudotime_metrics ----", flush=True)
    print(f"[TI.DEBUG] ad.n_obs={ad.n_obs}, ad.n_vars={ad.n_vars}", flush=True)
    print(f"[TI.DEBUG] embedding_key={embedding_key}", flush=True)
    print(f"[TI.DEBUG] available obsm keys={list(ad.obsm.keys())}", flush=True)
    print(f"[TI.DEBUG] time_key={time_key}, root_mask_key={root_mask_key}", flush=True)

    ad = ad.copy()  # avoid mutating caller
    if embedding_key not in ad.obsm:
        raise KeyError(f"Embedding key '{embedding_key}' not found in ad.obsm")

    X = ad.obsm[embedding_key]
    print(f"[TI.DEBUG] embedding shape from obsm[{embedding_key}] = {X.shape}", flush=True)

    ad.obsm["X_use_ti"] = X

    # Build kNN graph
    print(f"[TI.DEBUG] calling sc.pp.neighbors with n_neighbors={n_neighbors}", flush=True)
    sc.pp.neighbors(ad, use_rep="X_use_ti", n_neighbors=n_neighbors)
    print("[TI.DEBUG] sc.pp.neighbors finished", flush=True)

    # Diffusion maps + DPT
    print("[TI.DEBUG] calling sc.tl.diffmap", flush=True)
    sc.tl.diffmap(ad)
    print("[TI.DEBUG] sc.tl.diffmap finished", flush=True)

    if root_mask_key is not None:
        print(f"[TI.DEBUG] root_mask_key provided: {root_mask_key}", flush=True)
        if root_mask_key not in ad.obs:
            raise KeyError(f"root_mask_key '{root_mask_key}' not in ad.obs")
        roots = np.where(ad.obs[root_mask_key].values.astype(bool))[0]
        print(f"[TI.DEBUG] roots indices from root_mask_key: {roots[:10]}, total={roots.size}", flush=True)
        if roots.size == 0:
            raise ValueError(
                f"root_mask_key '{root_mask_key}' has no True entries."
            )
        root_idx = int(roots[0])
    else:
        print("[TI.DEBUG] root_mask_key is None", flush=True)
        if time_key is not None and time_key in ad.obs:
            t_vals = ad.obs[time_key].to_numpy()
            print(f"[TI.DEBUG] time_key present, first 10 t_vals={t_vals[:10]}", flush=True)
            root_idx = int(np.nanargmin(t_vals))
            print(f"[TI.DEBUG] root_idx from min(time) = {root_idx}", flush=True)
        else:
            print("[TI.DEBUG] time_key is None or not in ad.obs, using root_idx=0", flush=True)
            root_idx = 0

    ad.uns["iroot"] = root_idx
    print(f"[TI.DEBUG] set ad.uns['iroot']={root_idx}", flush=True)

    print(f"[TI.DEBUG] calling sc.tl.dpt with n_dcs={n_dcs}", flush=True)
    sc.tl.dpt(ad, n_dcs=n_dcs)
    print("[TI.DEBUG] sc.tl.dpt finished", flush=True)

    pt = ad.obs["dpt_pseudotime"].to_numpy()
    print(f"[TI.DEBUG] pt.shape={pt.shape}, first 10 pt={pt[:10]}", flush=True)

    metrics: Dict[str, float] = {
        "pt_variance": float(np.nanvar(pt)),
        "pt_min": float(np.nanmin(pt)),
        "pt_max": float(np.nanmax(pt)),
    }
    print(f"[TI.DEBUG] pt_variance={metrics['pt_variance']}, pt_min={metrics['pt_min']}, pt_max={metrics['pt_max']}", flush=True)

    if time_key is not None:
        print(f"[TI.DEBUG] computing correlation vs time_key={time_key}", flush=True)
        if time_key not in ad.obs:
            raise KeyError(f"time_key '{time_key}' not found in ad.obs")
        t_true = ad.obs[time_key].to_numpy().astype(float)
        print(f"[TI.DEBUG] t_true.shape={t_true.shape}, first 10 t_true={t_true[:10]}", flush=True)
        valid = np.isfinite(t_true) & np.isfinite(pt)
        print(f"[TI.DEBUG] valid.sum={valid.sum()}", flush=True)
        if valid.sum() > 10:
            rho, _ = spearmanr(t_true[valid], pt[valid])
            tau, _ = kendalltau(t_true[valid], pt[valid])
            metrics["pt_spearman_vs_time"] = float(rho)
            metrics["pt_kendall_vs_time"] = float(tau)
            print(f"[TI.DEBUG] spearman={rho}, kendall={tau}", flush=True)
        else:
            metrics["pt_spearman_vs_time"] = np.nan
            metrics["pt_kendall_vs_time"] = np.nan
            print("[TI.DEBUG] not enough valid points for correlation; setting NaN", flush=True)

    print("[TI.DEBUG] ---- compute_pseudotime_metrics done ----", flush=True)
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
    """
    print("[TI.DEBUG] ---- compute_branch_metrics ----", flush=True)
    print(f"[TI.DEBUG] ad.n_obs={ad.n_obs}, ad.n_vars={ad.n_vars}", flush=True)
    print(f"[TI.DEBUG] embedding_key={embedding_key}, cluster_key={cluster_key}, fate_key={fate_key}", flush=True)
    print(f"[TI.DEBUG] available obsm keys={list(ad.obsm.keys())}", flush=True)
    print(f"[TI.DEBUG] available obs columns={list(ad.obs.columns)}", flush=True)

    ad = ad.copy()
    if embedding_key not in ad.obsm:
        raise KeyError(f"Embedding key '{embedding_key}' not found in ad.obsm")
    if cluster_key not in ad.obs:
        raise KeyError(f"cluster_key '{cluster_key}' not found in ad.obs")
    if fate_key not in ad.obs:
        raise KeyError(f"fate_key '{fate_key}' not found in ad.obs")

    ad.obsm["X_use_ti"] = ad.obsm[embedding_key]
    print(f"[TI.DEBUG] X_use_ti.shape={ad.obsm['X_use_ti'].shape}", flush=True)

    print(f"[TI.DEBUG] calling sc.pp.neighbors for branch metrics with n_neighbors={n_neighbors}", flush=True)
    sc.pp.neighbors(ad, use_rep="X_use_ti", n_neighbors=n_neighbors)
    print("[TI.DEBUG] sc.pp.neighbors finished (branch)", flush=True)

    print(f"[TI.DEBUG] calling sc.tl.paga with groups={cluster_key}", flush=True)
    sc.tl.paga(ad, groups=cluster_key)
    print("[TI.DEBUG] sc.tl.paga finished", flush=True)

    y_cluster = ad.obs[cluster_key].astype("category").cat.codes.to_numpy()
    y_fate = ad.obs[fate_key].astype("category").cat.codes.to_numpy()
    print(f"[TI.DEBUG] y_cluster.shape={y_cluster.shape}, y_fate.shape={y_fate.shape}", flush=True)
    print(f"[TI.DEBUG] first 10 y_cluster={y_cluster[:10]}, first 10 y_fate={y_fate[:10]}", flush=True)

    valid = np.isfinite(y_cluster) & np.isfinite(y_fate)
    print(f"[TI.DEBUG] valid.sum={valid.sum()}", flush=True)
    if valid.sum() == 0:
        ari_cf = np.nan
        print("[TI.DEBUG] no valid entries for ARI; setting NaN", flush=True)
    else:
        ari_cf = adjusted_rand_score(y_fate[valid], y_cluster[valid])
        print(f"[TI.DEBUG] ARI(cluster, fate)={ari_cf}", flush=True)

    metrics: Dict[str, float] = {
        "branch_ari_cluster_vs_fate": float(ari_cf),
        "branch_n_clusters": int(np.unique(y_cluster[valid]).size),
        "branch_n_fates": int(np.unique(y_fate[valid]).size),
    }
    print(f"[TI.DEBUG] branch_n_clusters={metrics['branch_n_clusters']}, branch_n_fates={metrics['branch_n_fates']}", flush=True)
    print("[TI.DEBUG] ---- compute_branch_metrics done ----", flush=True)
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
    """
    print("[TI.DEBUG] ---- knn_graph_overlap ----", flush=True)
    print(f"[TI.DEBUG] X_base.shape={X_base.shape}, X_embed.shape={X_embed.shape}", flush=True)
    print(f"[TI.DEBUG] n_neighbors={n_neighbors}", flush=True)

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
    print(f"[TI.DEBUG] idx_base.shape={idx_base.shape}, idx_emb.shape={idx_emb.shape}", flush=True)

    overlaps = []
    for i, (nb, ne) in enumerate(zip(idx_base, idx_emb)):
        set_b = set(nb.tolist())
        set_e = set(ne.tolist())
        inter = len(set_b & set_e)
        union = len(set_b | set_e)
        if union > 0:
            overlaps.append(inter / union)
        if i < 3:
            print(f"[TI.DEBUG] cell {i}: inter={inter}, union={union}, jaccard={overlaps[-1] if union > 0 else 'NA'}", flush=True)

    overlaps_arr = np.asarray(overlaps, dtype=float)
    mean_val = float(np.nanmean(overlaps_arr))
    std_val = float(np.nanstd(overlaps_arr))
    print(f"[TI.DEBUG] graph_knn_jaccard_mean={mean_val}, std={std_val}", flush=True)
    print("[TI.DEBUG] ---- knn_graph_overlap done ----", flush=True)
    return {
        "graph_knn_jaccard_mean": mean_val,
        "graph_knn_jaccard_std": std_val,
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
    random_state: int = 0,
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
    random_state
        RNG seed (currently unused, reserved for future extensions).

    Returns
    -------
    metrics
        Dictionary aggregating all available TI-relevant metrics.
        (No raw pt_values included — only scalars.)
    """
    metrics: Dict[str, Any] = {}

    # 1) Pseudotime metrics (we ignore the raw pt array)
    pt_metrics, _ = compute_pseudotime_metrics(
        ad,
        embedding_key=embedding_key,
        time_key=time_key,
        root_mask_key=root_mask_key,
        n_neighbors=n_neighbors,
        n_dcs=n_dcs,
    )
    metrics.update(pt_metrics)

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