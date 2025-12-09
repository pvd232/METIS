# src/medit/diffusion/config.py
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Literal


GeometrySource = Literal["pca", "hvg"]
EnergySource = Literal["hvg", "pca"]
MetricMode = Literal["euclidean", "scm", "hessian_mixed"]

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
    """
    Config for EGGFM-based diffusion maps.

    Mirrors the `eggfm_diffusion` (or `eggfm_diffmap`) block in configs/params.yml.
    """
    # geometry / energy views
    geometry_source: GeometrySource = "pca"
    energy_source: EnergySource = "hvg"

    # kNN graph
    n_neighbors: int = 30

    # metric choice
    metric_mode: MetricMode = "hessian_mixed"
    norm_type: str = "l2"

    # diffusion map parameters
    n_comps: int = 30
    t: float = 1.0
    eps_trunc: str | None = None  # "yes" or None

    # Hessian / SCM hyperparams (only used if relevant)
    metric_gamma: float = 0.2
    metric_lambda: float = 4.0
    energy_batch_size: int = 2048
    energy_clip_abs: float = 3.0

    hessian_mix_mode: str = "multiplicative"
    hessian_mix_alpha: float = 0.3
    hessian_beta: float = 0.3
    hessian_clip_std: float = 2.0
    hessian_use_neg: bool = True
    hvp_batch_size: int = 1024
    
    tangent_k: int = 30
    tangent_dim: int = 5
    normal_k: int = 30
    normal_dim: int = 5

    # device
    device: str = "auto"  # "auto", "cuda", "cpu"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

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