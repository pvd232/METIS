# src/medit/diffusion/metrics_riemann.py

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch
from scipy import sparse as sp_sparse
from sklearn.neighbors import NearestNeighbors

import scanpy as sc  # type: ignore

from .metrics import (
    BaseMetric,
    pairwise_norm,
    _hessian_quadratic_form_batched,
)


# ---------------------------------------------------------------------
# Helper: estimate local tangent bases in geometry space
# ---------------------------------------------------------------------


def _estimate_tangent_bases(
    X_geom: np.ndarray,
    k: int,
    tangent_dim: int,
) -> list[np.ndarray]:
    """
    Estimate a local PCA-based tangent basis U_i for each point i
    in the *geometry* space X_geom.

    Parameters
    ----------
    X_geom : (n_cells, d_geom)
        Geometry matrix (typically PCA embedding).
    k : int
        Number of neighbors used to define the local patch.
    tangent_dim : int
        Number of leading PCs to treat as tangent directions.

    Returns
    -------
    U_list : list of (d_geom, tangent_dim) arrays
        For each i, U[i] has orthonormal columns spanning the tangent space.
    """
    n_cells, d_geom = X_geom.shape
    tangent_dim = max(1, min(tangent_dim, d_geom))
    k = max(1, min(k, n_cells - 1))

    print(
        f"[Riemann] estimating tangent bases in geometry space: "
        f"n_cells={n_cells}, d_geom={d_geom}, k={k}, tangent_dim={tangent_dim}",
        flush=True,
    )

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_geom)
    _, idx = nn.kneighbors(X_geom)  # (n_cells, k+1)

    U_list: list[np.ndarray] = []

    for i in range(n_cells):
        neigh_idx = idx[i, 1:]  # drop self
        local = X_geom[neigh_idx] - X_geom[i]

        if local.shape[0] == 0:
            # Fallback: use first tangent_dim std-basis vectors
            Ui = np.eye(d_geom, tangent_dim, dtype=np.float32)
            U_list.append(Ui)
            continue

        # Local covariance and eigen-decomposition
        C = local.T @ local / max(local.shape[0], 1)
        evals, evecs = np.linalg.eigh(C)  # returns in ascending order

        order = np.argsort(evals)[::-1]  # largest first (max variance)
        Ui = evecs[:, order[:tangent_dim]]  # (d_geom, tangent_dim)

        U_list.append(Ui.astype(np.float32))

        if (i + 1) % 5000 == 0 or i == n_cells - 1:
            print(
                f"[Riemann] tangent bases: finished {i+1}/{n_cells}",
                flush=True,
            )

    return U_list


# ---------------------------------------------------------------------
# 1. Tangent-alignment metric
# ---------------------------------------------------------------------


class RiemannTangentAlignmentMetric(BaseMetric):
    """
    Metric that favors motion along the local tangent space and
    penalizes motion in the normal directions.

    Geometry:
      - Work entirely in geometry space X_geom (typically PCA).
      - For each point i, estimate a local tangent basis U_i using PCA
        on its k neighbors.
      - For each edge (i, j), decompose v_ij into tangent + normal:

            v = x_j - x_i
            v_tan  = U_i U_i^T v
            v_norm = v - v_tan

        Then define a Riemannian distance

            d_ij^2 = alpha * ||v_tan||^2 + beta * ||v_norm||^2

        with alpha < 1, beta > 1 so that edges aligned with the tangent
        are relatively shorter and those leaving the manifold are longer.
    """

    def prepare(
        self,
        ad_prep: sc.AnnData,
        energy_model,
        device: str,
    ) -> Dict[str, Any]:
        # Nothing to precompute from ad_prep here; everything is in geometry space.
        # We keep the signature for compatibility.
        return {"device": device}

    def edge_distances(
        self,
        X_geom: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        state: Dict[str, Any],
        norm_type: str = "l2",
    ) -> np.ndarray:
        n_cells, d_geom = X_geom.shape
        n_edges = rows.shape[0]

        # Hyperparameters
        tangent_k = int(self.diff_cfg.get("tangent_k", self.diff_cfg.get("n_neighbors", 30)))
        tangent_dim = int(self.diff_cfg.get("tangent_dim", min(10, d_geom)))

        alpha = float(self.diff_cfg.get("riem_tan_alpha", 0.5))  # weight for tangent
        beta = float(self.diff_cfg.get("riem_norm_beta", self.diff_cfg.get("normal_weight", 2.0)))

        # 1) Estimate tangent bases in geometry space
        U_list = _estimate_tangent_bases(X_geom, k=tangent_k, tangent_dim=tangent_dim)

        # 2) Compute edge-wise distances
        dist = np.empty(n_edges, dtype=np.float64)

        for idx, (i, j) in enumerate(zip(rows, cols)):
            xi = X_geom[i]
            xj = X_geom[j]
            v = xj - xi  # (d_geom,)

            Ui = U_list[i]  # (d_geom, t_dim)

            # Project to tangent and normal
            v_tan = Ui @ (Ui.T @ v)
            v_norm = v - v_tan

            tan_norm2 = float(np.dot(v_tan, v_tan))
            nor_norm2 = float(np.dot(v_norm, v_norm))

            # Weighted quadratic combination
            d2 = alpha * tan_norm2 + beta * nor_norm2
            if d2 < 1e-12:
                d2 = 1e-12
            dist[idx] = np.sqrt(d2)

            if (idx + 1) % 200000 == 0 or idx == n_edges - 1:
                print(
                    f"[RiemannTangent] processed {idx+1}/{n_edges} edges",
                    flush=True,
                )

        dist[dist < 1e-12] = 1e-12
        return dist


# ---------------------------------------------------------------------
# 2. Curvature-aware metric
# ---------------------------------------------------------------------


class RiemannCurvatureMetric(BaseMetric):
    """
    Curvature-aware metric based on energy Hessian.

    Geometry:
      - Base distances are computed in geometry space X_geom.
      - For each edge (i, j), we take the direction v_ij in *energy* space,
        compute q_ij = v̂^T H_E(x_i) v̂ via Hessian-vector products where v̂ is
        the unit direction, and then form a multiplicative factor:

            factor_ij = exp(beta * q_std_ij)

        where q_std_ij is median/MAD-normalized and clipped. Final distance:

            d_ij = base_ij * factor_ij
    """

    def prepare(
        self,
        ad_prep: sc.AnnData,
        energy_model,
        device: str,
    ) -> Dict[str, Any]:
        # Choose energy space to match SCM / Hessian-mixed conventions
        energy_source = self.diff_cfg.get("energy_source", "hvg").lower()
        if energy_source == "hvg":
            X_energy = ad_prep.X
        elif energy_source == "pca":
            if "X_pca" not in ad_prep.obsm:
                raise ValueError(
                    "[RiemannCurvature] energy_source='pca' but 'X_pca' missing "
                    "in ad_prep.obsm."
                )
            X_energy = ad_prep.obsm["X_pca"]
        else:
            raise ValueError(f"[RiemannCurvature] Unknown energy_source: {energy_source!r}")

        if sp_sparse.issparse(X_energy):
            X_energy = X_energy.toarray()
        X_energy = np.asarray(X_energy, dtype=np.float32)

        energy_model = energy_model.to(device)
        energy_model.eval()

        return {
            "X_energy": X_energy,
            "energy_model": energy_model,
            "device": device,
        }

    def edge_distances(
        self,
        X_geom: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        state: Dict[str, Any],
        norm_type: str = "l2",
    ) -> np.ndarray:
        X_energy = state["X_energy"]
        energy_model = state["energy_model"]
        device = state["device"]

        n_edges = rows.shape[0]

        # Base distances in geometry space
        Xi_geom = X_geom[rows]
        Xj_geom = X_geom[cols]
        base = pairwise_norm(Xi_geom, Xj_geom, norm=norm_type).astype(np.float64)

        # Hyperparams for curvature factor
        edge_batch_size = int(self.diff_cfg.get("hvp_batch_size", 1024))
        beta = float(self.diff_cfg.get("curvature_beta", 0.3))
        clip_std = float(self.diff_cfg.get("curvature_clip_std", 2.0))

        # Compute q_ij = v̂^T H v̂ in energy space
        q_all = np.empty(n_edges, dtype=np.float64)

        print(
            "[RiemannCurvature] computing Hessian-based curvature along edges...",
            flush=True,
        )

        for b in range(0, n_edges, edge_batch_size):
            start = b
            end = min(b + edge_batch_size, n_edges)
            if start >= end:
                break

            i_batch = rows[start:end]
            j_batch = cols[start:end]

            Xi_energy = X_energy[i_batch]
            Xj_energy = X_energy[j_batch]
            V = Xj_energy - Xi_energy  # (B, D)

            norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
            V_unit = V / norms

            q_batch = _hessian_quadratic_form_batched(
                energy_model=energy_model,
                X_batch=Xi_energy,
                V_batch=V_unit,
                device=device,
            )
            q_all[start:end] = q_batch

            if (end % (10 * edge_batch_size)) == 0 or end == n_edges:
                print(
                    f"[RiemannCurvature] processed edges {end}/{n_edges}",
                    flush=True,
                )

        # Robust normalize q_all (median + MAD), then clip
        med_q = np.median(q_all)
        mad_q = np.median(np.abs(q_all - med_q)) + 1e-8
        q_std = (q_all - med_q) / mad_q
        q_std = np.clip(q_std, -clip_std, clip_std)

        # Multiplicative curvature factor
        factor = np.exp(beta * q_std).astype(np.float64)

        dist = base * factor
        dist[dist < 1e-12] = 1e-12
        return dist


# ---------------------------------------------------------------------
# 3. Normal-suppression metric
# ---------------------------------------------------------------------


class RiemannNormalSuppressMetric(BaseMetric):
    """
    Metric that strongly penalizes motion in the normal directions
    while preserving the base distance in tangent directions.

    Geometry:
      - As in RiemannTangentAlignmentMetric, we estimate a local tangent
        basis U_i in geometry space.
      - For each edge (i, j), decompose v_ij into tangent + normal:

            v = x_j - x_i
            v_tan  = U_i U_i^T v
            v_norm = v - v_tan

        and set

            d_ij^2 = ||v||^2 + beta * ||v_norm||^2

        so that purely tangent motion leaves d_ij ≈ ||v||, while motion
        with a large normal component gets stretched.
    """

    def prepare(
        self,
        ad_prep: sc.AnnData,
        energy_model,
        device: str,
    ) -> Dict[str, Any]:
        # No precomputation needed beyond geometry; keep signature for compatibility.
        return {"device": device}

    def edge_distances(
        self,
        X_geom: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        state: Dict[str, Any],
        norm_type: str = "l2",
    ) -> np.ndarray:
        n_cells, d_geom = X_geom.shape
        n_edges = rows.shape[0]

        # Hyperparameters
        normal_k = int(self.diff_cfg.get("normal_k", self.diff_cfg.get("n_neighbors", 30)))
        tangent_dim = int(self.diff_cfg.get("normal_dim", self.diff_cfg.get("tangent_dim", min(10, d_geom))))
        beta = float(self.diff_cfg.get("riem_norm_beta", self.diff_cfg.get("normal_weight", 2.0)))

        # 1) Estimate tangent bases in geometry space (reuse helper)
        U_list = _estimate_tangent_bases(X_geom, k=normal_k, tangent_dim=tangent_dim)

        # 2) Compute edge-wise distances
        dist = np.empty(n_edges, dtype=np.float64)

        for idx, (i, j) in enumerate(zip(rows, cols)):
            xi = X_geom[i]
            xj = X_geom[j]
            v = xj - xi  # (d_geom,)

            Ui = U_list[i]  # (d_geom, t_dim)
            v_tan = Ui @ (Ui.T @ v)
            v_norm = v - v_tan

            base2 = float(np.dot(v, v))
            nor2 = float(np.dot(v_norm, v_norm))

            d2 = base2 + beta * nor2
            if d2 < 1e-12:
                d2 = 1e-12
            dist[idx] = np.sqrt(d2)

            if (idx + 1) % 200000 == 0 or idx == n_edges - 1:
                print(
                    f"[RiemannNormal] processed {idx+1}/{n_edges} edges",
                    flush=True,
                )

        dist[dist < 1e-12] = 1e-12
        return dist
