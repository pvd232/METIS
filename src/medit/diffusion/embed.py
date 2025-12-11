# src/medit/diffusion/embed.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Mapping   # ← add Mapping

import yaml
import torch
import scanpy as sc  # type: ignore

from medit.eggfm import EnergyMLP
from .engine import EGGFMDiffusionEngine
from .config import DiffusionConfig


def _ensure_diff_cfg(
    cfg: DiffusionConfig | Mapping[str, Any],
) -> DiffusionConfig:
    if isinstance(cfg, DiffusionConfig):
        return cfg
    return DiffusionConfig(**dict(cfg))


def build_diffusion_embedding_from_config(
    ad: sc.AnnData,
    energy_model: EnergyMLP,                       # ← optional but nicer typing
    diff_cfg: DiffusionConfig | Mapping[str, Any],
    obsm_key: str,
) -> sc.AnnData:
    """
    Construct an EGGFM-based diffusion embedding and store it in ad.obsm[obsm_key].
    """
    diff_cfg = _ensure_diff_cfg(diff_cfg)
    engine = EGGFMDiffusionEngine(
    energy_model=energy_model,
    diff_cfg={
        "geometry_source": diff_cfg.geometry_source,
        "energy_source": diff_cfg.energy_source,
        "n_neighbors": diff_cfg.n_neighbors,
        "metric_mode": diff_cfg.metric_mode,
        "norm_type": diff_cfg.norm_type,
        "n_comps": diff_cfg.n_comps,
        "t": diff_cfg.t,
        "eps_trunc": diff_cfg.eps_trunc,
        "distance_power": getattr(diff_cfg, "distance_power", 1.0),

        "metric_gamma": diff_cfg.metric_gamma,
        "metric_lambda": diff_cfg.metric_lambda,
        "energy_batch_size": diff_cfg.energy_batch_size,
        "energy_clip_abs": diff_cfg.energy_clip_abs,

        "hessian_mix_mode": diff_cfg.hessian_mix_mode,
        "hessian_mix_alpha": diff_cfg.hessian_mix_alpha,
        "hessian_beta": diff_cfg.hessian_beta,
        "hessian_clip_std": diff_cfg.hessian_clip_std,
        "hessian_use_neg": diff_cfg.hessian_use_neg,
        "hvp_batch_size": diff_cfg.hvp_batch_size,

        # NEW: Riemannian geometry knobs
        "tangent_k": diff_cfg.tangent_k,
        "tangent_dim": diff_cfg.tangent_dim,
        "normal_k": diff_cfg.normal_k,
        "normal_dim": diff_cfg.normal_dim,
        "riem_tan_alpha": getattr(diff_cfg, "riem_tan_alpha", 1.0),
        "riem_norm_beta": getattr(diff_cfg, "riem_norm_beta", 1.0),
        "riem_beta": getattr(diff_cfg, "riem_beta", 0.3),

        "device": diff_cfg.resolve_device(),
    },
)

    embedding = engine.build_embedding(ad)
    ad.obsm[obsm_key] = embedding
    return ad
