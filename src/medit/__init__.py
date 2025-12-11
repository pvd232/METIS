# src/medit/__init__.py
from __future__ import annotations

# Dataset config
from .config import WEINREB, WeinrebDatasetConfig

# EGGFM
from medit.eggfm import (
    EnergyMLP,
    AnnDataExpressionDataset,
    train_energy_model,
)
from medit.eggfm.config import (
    EnergyModelConfig,
    EnergyTrainConfig,
    EnergyModelBundle,
)

# Diffusion
from medit.diffusion import (
    DiffusionMapBuilder,
    EGGFMDiffusionEngine,
    # AlternatingDiffuser,  # uncomment only if you keep admr.py around
)
from medit.diffusion.config import DiffusionConfig, diffusion_config_from_params
from medit.diffusion.embed import build_eggfm_geometry_from_config
