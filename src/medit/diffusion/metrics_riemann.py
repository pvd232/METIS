from __future__ import annotations
import numpy as np
import torch
from sklearn.decomposition import PCA
from typing import Dict, Any

from .metrics import BaseMetric, pairwise_norm
from .metrics import _hessian_quadratic_form_batched


# -------------------------------------------------------------
# Utility: local PCA tangent space estimator
# -------------------------------------------------------------
def estimate_tangent_space(X: np.ndarray, n_neighbors: int, n_tangent: int):
    """
    Compute local PCA tangent space for each point.

    Returns:
        U[i] → first n_tangent unit vectors of local PCA basis (tangent)
        N[i] → remaining normal-space basis
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, idx = nn.kneighbors(X)

    U_list = []
    N_list = []
    for i in range(X.shape[0]):
        local = X[idx[i, 1:]]
        p = PCA().fit(local)
        comps = p.components_
        U_list.append(comps[:n_tangent])
        N_list.append(comps[n_tangent:])
    return np.array(U_list, dtype=object), np.array(N_list, dtype=object)


# -------------------------------------------------------------
# 1. Tangent-alignment metric
# -------------------------------------------------------------
class RiemannTangentAlignmentMetric(BaseMetric):

    def prepare(self, ad_prep, energy_model, device: str):
        X = np.asarray(ad_prep.X.toarray() if hasattr(ad_prep.X, "toarray") else ad_prep.X)
        k = int(self.diff_cfg.get("tangent_k", 30))
        d = int(self.diff_cfg.get("tangent_dim", 10))
        print(f"[Riemann] estimating tangent dimension={d} with kNN={k}")

        U, N = estimate_tangent_space(X, n_neighbors=k, n_tangent=d)
        return {"U": U, "N": N}

    def edge_distances(self, X_geom, rows, cols, state, norm_type="l2"):
        U = state["U"]
        N = state["N"]

        Xi = X_geom[rows]
        Xj = X_geom[cols]
        v = Xj - Xi

        # projection onto tangent + normal
        dist = np.zeros(len(rows))
        for idx, (vi, i, j) in enumerate(zip(v, rows, cols)):
            Ui = U[i]      # tangent basis
            Ni = N[i]      # normal basis

            v_tan = Ui.T @ (Ui @ vi)
            v_norm = vi - v_tan

            tan_norm = np.linalg.norm(v_tan)
            nor_norm = np.linalg.norm(v_norm)

            # weight tangent lower, normal higher
            alpha = float(self.diff_cfg.get("riem_tan_alpha", 0.7))
            beta  = float(self.diff_cfg.get("riem_norm_beta", 2.0))

            dist[idx] = alpha * tan_norm + beta * nor_norm

        dist[dist < 1e-12] = 1e-12
        return dist
        

# -------------------------------------------------------------
# 2. Curvature-aware metric
# -------------------------------------------------------------
class RiemannCurvatureMetric(BaseMetric):

    def prepare(self, ad_prep, energy_model, device: str):
        X = np.asarray(ad_prep.X.toarray() if hasattr(ad_prep.X, "toarray") else ad_prep.X)
        self.energy_model = energy_model.to(device)
        self.device = device
        self.X = X
        return {"dummy": True}

    def edge_distances(self, X_geom, rows, cols, state, norm_type="l2"):
        Xi = self.X[rows]
        Xj = self.X[cols]
        V = Xj - Xi
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
        V_unit = V / norms

        # second derivative estimate: v^T H v
        q = _hessian_quadratic_form_batched(
            energy_model=self.energy_model,
            X_batch=Xi,
            V_batch=V_unit,
            device=self.device,
        )

        q = np.clip(q, -20, 20)
        curvature = np.exp(0.3 * q)

        base = pairwise_norm(Xi, Xj, norm=norm_type)
        dist = base * curvature
        dist[dist < 1e-12] = 1e-12
        return dist


# -------------------------------------------------------------
# 3. Normal-suppression metric
# -------------------------------------------------------------
class RiemannNormalSuppressMetric(BaseMetric):

    def prepare(self, ad_prep, energy_model, device: str):
        X = np.asarray(ad_prep.X.toarray() if hasattr(ad_prep.X, "toarray") else ad_prep.X)
        k = int(self.diff_cfg.get("normal_k", 30))
        d = int(self.diff_cfg.get("normal_dim", 10))
        U, N = estimate_tangent_space(X, n_neighbors=k, n_tangent=d)
        return {"U": U, "N": N}

    def edge_distances(self, X_geom, rows, cols, state, norm_type="l2"):
        U = state["U"]
        N = state["N"]

        Xi = X_geom[rows]
        Xj = X_geom[cols]
        v = Xj - Xi

        dist = np.zeros(len(rows))
        beta = float(self.diff_cfg.get("riem_norm_beta", 2.0))

        for idx, (vi, i) in enumerate(zip(v, rows)):
            Ni = N[i]  # normal basis

            v_norm = Ni.T @ (Ni @ vi)
            d_norm = np.linalg.norm(v_norm)
            d_base = np.linalg.norm(vi)

            dist[idx] = d_base * (1 + beta * d_norm)

        dist[dist < 1e-12] = 1e-12
        return dist
