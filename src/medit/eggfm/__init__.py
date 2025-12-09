# src/medit/eggfm/__init__.py
"""
EGGFM — Energy-Guided Geometric Flow Model (energy component) for MEDIT.

Public API:
  - EnergyMLP
  - AnnDataExpressionDataset
  - train_energy_model        (core DSM loop)
  - train_energy_from_config  (config + h5ad → checkpoint)
"""

from __future__ import annotations

from .models import EnergyMLP  # noqa: F401
from .dataset import AnnDataExpressionDataset  # noqa: F401
from .trainer import train_energy_model  # noqa: F401
