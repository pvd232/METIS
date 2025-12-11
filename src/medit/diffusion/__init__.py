# src/medit/diffusion/__init__.py
"""
Diffusion / geometry components for MEDIT.

This subpackage contains:
  - core:         generic diffusion map construction
  - engine:       EGGFMDiffusionEngine and friends
  - metrics:      metric layer (Euclidean, SCM, Hessian-mixed, ...)
  - data_sources: AnnDataViewProvider and related views
  - admr:         AlternatingDiffuser (ADMR) for layered diffusion
  - utils:        small helpers used by the diffusion stack
"""

from __future__ import annotations

from .core import DiffusionMapBuilder  
from .engine import EGGFMDiffusionEngine  
from .data_sources import AnnDataViewProvider  

from .metrics import (  
    BaseMetric,
    EuclideanMetric,
    SCMMetric,
    HessianMixedMetric,
)

from .admr import AlternatingDiffuser  
from .embed import build_eggfm_geometry_from_config
