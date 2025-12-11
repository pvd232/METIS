# src/medit/diffusion/geom.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import sparse  # type: ignore

from .config import DiffusionConfig

@dataclass
class EGGFMGeometry:
    """
    Full geometry bundle produced by EGGFM + diffusion maps.

    This is what TI code should consume.
    """
    # raw geometry used to build kNN graph (PCA or HVG)
    X_geom: np.ndarray               # [N, d_geom]

    # final low-dim embedding (diffusion coords, or later flow coords)
    X_embed: np.ndarray              # [N, n_comps]

    # kNN graph in index form
    knn_indices: np.ndarray          # [N, k]
    knn_distances: np.ndarray        # [N, k]

    # flattened edge list and metric-aware distances
    rows: np.ndarray                 # [E]
    cols: np.ndarray                 # [E]
    edge_distances: np.ndarray       # [E]

    # optional:
    kernel: Optional[sparse.spmatrix] = None  # P or W, if you want it
    eigvals: Optional[np.ndarray] = None      # [n_comps]
    diff_cfg: Optional[DiffusionConfig] = None
