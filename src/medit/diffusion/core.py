# EGGFM/diffusion_core.py

from typing import Dict, Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs


class DiffusionMapBuilder:
    """
    Metric- and norm-agnostic diffusion map constructor.
    """

    def __init__(self, diff_cfg: Dict[str, Any]):
        self.diff_cfg = diff_cfg
    def build_from_distances(
        self,
        n_cells: int,
        rows: np.ndarray,
        cols: np.ndarray,
        dist_vals: np.ndarray,
        return_kernel: bool = False,
        return_eigvals: bool = False,
    ):
        """
        Build diffusion coordinates from pairwise distances.

        If return_kernel and return_eigvals are both False (default),
        returns only the diffusion coordinates (backwards compatible).

        Otherwise returns a tuple:
            (diff_coords, kernel_or_None, eigvals_or_None)
        """
        diff_cfg = self.diff_cfg
        n_comps = diff_cfg.get("n_comps", 30)
        t = diff_cfg.get("t", 3.0)
        distance_power = diff_cfg.get("distance_power", 1.0)

        # 1) kernel W
        d_p = dist_vals ** distance_power
        W = sparse.csr_matrix(
            (np.exp(-d_p), (rows, cols)),
            shape=(n_cells, n_cells),
        )

        # 2) Markov matrix P
        d = np.array(W.sum(axis=1)).ravel()
        d_safe = np.where(d > 0, d, 1.0)
        D_inv = sparse.diags(1.0 / d_safe)
        P = D_inv @ W

        # 3) eigen-decomp
        k_eigs = n_comps + 1
        eigvals, eigvecs = eigs(P.T, k=k_eigs, which="LR")
        eigvals = eigvals.real
        eigvecs = eigvecs.real

        order = np.argsort(-eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        lambdas = eigvals[1 : n_comps + 1]
        phis = eigvecs[:, 1 : n_comps + 1]

        diff_coords = phis * (lambdas**t)
        diff_coords = diff_coords.astype(np.float32)

        # --- Backwards-compatible return shape ---
        if not (return_kernel or return_eigvals):
            # Old behavior: only embedding
            return diff_coords

        kernel_out = P if return_kernel else None
        eig_out = lambdas if return_eigvals else None
        return diff_coords, kernel_out, eig_out
