# src/medit/diffusion/config.py
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Literal


GeometrySource = Literal["pca", "hvg"]
EnergySource = Literal["hvg", "pca"]
MetricMode = Literal["euclidean", "scm", "hessian_mixed"]
NormType = Literal["linf", "l2", "l1", "l0"]
HessianMixMode = Literal["multiplicative", "additive", "none"]

@dataclass
class RiemannianConfig:
    use_hessian_fro: bool = False
    use_directional_smoothness: bool = False
    use_tangent_penalty: bool = False
    use_true_riemannian_distance: bool = False

    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    n_dirs: int = 4

@dataclass
class DiffusionConfig:
    geometry_source: GeometrySource = "pca"
    energy_source: EnergySource = "hvg"
    metric_mode: MetricMode = "scm"
    n_neighbors: int = 30
    n_comps: int = 30
    hvp_batch_size: int = 8192
    eps_trunc: str = "no"
    t: float = 3.0
    norm_type: NormType = "l2"

    distance_power: float = 1.0

    metric_gamma: float = 5.0
    metric_lambda: float = 20.0
    energy_clip_abs: float = 6.0
    energy_batch_size: int = 8192
    hessian_mix_mode: HessianMixMode = "none"
    hessian_mix_alpha: float = 0.5
    hessian_beta: float = 1.0
    hessian_clip_std: float = 1.0
    hessian_use_neg: bool = True
    tangent_k: int = 30
    tangent_dim: int = 5
    normal_k: int = 30
    normal_dim: int = 5
    device: str = "auto"  # "auto", "cuda", "cpu"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EGGFMGeomConfig:
    diff: DiffusionConfig
    store_kernel: bool = False     # whether to return P
    store_eigvals: bool = True     # cache eigenvalues
    store_knn: bool = True         # knn_indices, knn_distances

def diffusion_config_from_params(
    params: Mapping[str, Any],
    key: str = "eggfm_diffmap",   # ← match your params.yml
) -> DiffusionConfig:
    """
    Build a DiffusionConfig from a params.yml-style dict.

    Looks up params[key], filters unknown keys, and applies defaults
    from the dataclass for anything not specified.
    """
    block = dict(params.get(key, {}))

    valid_fields = {f.name for f in fields(DiffusionConfig)}
    filtered = {k: v for k, v in block.items() if k in valid_fields}

    return DiffusionConfig(**filtered)