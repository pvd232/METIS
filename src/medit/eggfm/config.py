# src/medit/eggfm/config.py
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Sequence, Mapping, Any, Dict


@dataclass
class EnergyModelConfig:
    """
    Architecture for the EnergyMLP.
    Mirrors the expected keys in `eggfm_model` in params.yml.
    """
    hidden_dims: Sequence[int] = (512, 512)


@dataclass
class EnergyTrainConfig:
    batch_size: int = 256
    num_epochs: int = 100
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    sigma: float = 0.15
    device: str | None = None

    # from your YAML
    latent_space: str = "hvg"
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    base_lr: float | None = None

    # extras that trainer may use
    max_grad_norm: float = 5.0
    seed: int | None = None

@dataclass
class EnergyModelBundle:
    """
    Canonical checkpoint payload for an EGGFM model.
    Not required by training, but useful for saving / loading.
    """
    model_cfg: EnergyModelConfig
    train_cfg: EnergyTrainConfig
    n_genes: int
    var_names: Sequence[str]
    space: str  # e.g., "hvg"
    state_dict: Dict[str, Any]
    mean: Sequence[float]
    std: Sequence[float]

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "model_cfg": asdict(self.model_cfg),
            "train_cfg": asdict(self.train_cfg),
            "n_genes": int(self.n_genes),
            "var_names": list(self.var_names),
            "space": self.space,
            "state_dict": self.state_dict,
            "mean": list(self.mean),
            "std": list(self.std),
        }


def _filter_fields(cls, data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Drop any keys not in the dataclass fields.
    Prevents crashes if params.yml has extra keys.
    """
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


def energy_configs_from_params(
    params: Mapping[str, Any],
) -> tuple[EnergyModelConfig, EnergyTrainConfig]:
    """
    Helper to build EnergyModelConfig / EnergyTrainConfig from a params.yml dict.

    Expects keys:
      - "eggfm_model"
      - "eggfm_train"

    This is a convenience; it's optional to use.
    """
    model_block = _filter_fields(EnergyModelConfig, dict(params.get("eggfm_model", {})))
    train_block = _filter_fields(EnergyTrainConfig, dict(params.get("eggfm_train", {})))
    return EnergyModelConfig(**model_block), EnergyTrainConfig(**train_block)
