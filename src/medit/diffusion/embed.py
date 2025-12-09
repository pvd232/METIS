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
            "eps_value": diff_cfg.eps_value,
            "eps_trunc": diff_cfg.eps_trunc,
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
            "device": diff_cfg.resolve_device(),
        },
    )
    embedding = engine.build_embedding(ad)
    ad.obsm[obsm_key] = embedding
    return ad
