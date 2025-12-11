from __future__ import annotations

from typing import Dict, Any, Mapping
from pathlib import Path
from datetime import datetime
import json
import subprocess

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import scanpy as sc  # type: ignore

from .models import EnergyMLP
from .dataset import AnnDataExpressionDataset
from .config import EnergyModelConfig, EnergyTrainConfig
from .riemann import hessian_smoothness_penalty

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _to_plain(obj: Any) -> Any:
    """
    Convert config objects / Paths into JSON-serializable Python types.
    """
    from dataclasses import is_dataclass, asdict

    if obj is None:
        return None

    # Dataclasses (common pattern for configs)
    if is_dataclass(obj):
        return asdict(obj)

    # Pydantic / similar
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()

    # Generic objects with __dict__
    if hasattr(obj, "__dict__"):
        return {
            k: _to_plain(v)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }

    # pathlib.Path
    if isinstance(obj, Path):
        return str(obj)

    # Containers
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]

    return obj


def _next_run_dir(manifest_root: Path) -> tuple[str, Path]:
    """
    Find the next run id (run1, run2, ...) under manifest_root and create it.
    """
    manifest_root.mkdir(parents=True, exist_ok=True)

    existing = [
        p for p in manifest_root.iterdir()
        if p.is_dir() and p.name.startswith("run")
    ]
    max_idx = 0
    for p in existing:
        suffix = p.name[3:]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))

    next_idx = max_idx + 1
    run_id = f"run{next_idx}"
    run_dir = manifest_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def _git_commit_or_none() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha
    except Exception:
        return None

def log_config(
    model_cfg: EnergyModelConfig,
    train_cfg: EnergyTrainConfig,
    extra: Dict[str, Any] | None = None,
) -> Path:
    """
    Create ./manifest/runX/manifest.json in the CWD.

    Returns the created run directory path.
    """
    manifest_root = Path.cwd() / "manifest"
    run_id, run_dir = _next_run_dir(manifest_root)
    manifest_path = run_dir / "manifest.json"

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit_or_none(),
        "model_cfg": _to_plain(model_cfg),
        "train_cfg": _to_plain(train_cfg),
    }

    if extra:
        payload["extra"] = _to_plain(extra)

    with manifest_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"[Energy DSM] Wrote manifest to {manifest_path}", flush=True)
    return run_dir

def _ensure_model_cfg(cfg: EnergyModelConfig | dict) -> EnergyModelConfig:
    if isinstance(cfg, EnergyModelConfig):
        return cfg
    return EnergyModelConfig.from_dict(dict(cfg))


def _ensure_train_cfg(cfg: EnergyTrainConfig | Mapping[str, Any]) -> EnergyTrainConfig:
    if isinstance(cfg, EnergyTrainConfig):
        return cfg
    return EnergyTrainConfig(**dict(cfg))


def _dsm_loss(
    model: nn.Module,
    x: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    noise = torch.randn_like(x) * sigma
    x_noisy = x + noise
    x_noisy.requires_grad_(True)

    energy = model(x_noisy)
    if energy.ndim > 1:
        energy = energy.squeeze(-1)

    score = -torch.autograd.grad(
        outputs=energy.sum(),
        inputs=x_noisy,
        create_graph=True,
    )[0]
    target = -noise / (sigma**2)

    return 0.5 * (score - target).pow(2).sum(dim=1).mean()


# ---------------------------------------------------------------------
# The EnergyTrainer class — restored
# ---------------------------------------------------------------------

class EnergyTrainer:
    """
    Full trainer class wrapping DSM + optional Riemannian regularizers.
    """

    def __init__(
        self,
        model: EnergyMLP,
        dataset: AnnDataExpressionDataset,
        train_cfg: EnergyTrainConfig,
    ):
        train_cfg = _ensure_train_cfg(train_cfg)

        # device
        if train_cfg.device is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = train_cfg.device
        self.device = torch.device(device_str)

        # model
        self.model = model.to(self.device)

        # dataloader
        self.loader = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )

        self.cfg = train_cfg

        # Riemannian regularizer hyperparams
        self.riem_type = train_cfg.riemann_reg_type
        self.riem_weight = float(train_cfg.riemann_reg_weight)
        self.riem_eps = float(train_cfg.riemann_eps)
        self.riem_n_dirs = int(train_cfg.riemann_n_dirs)

    # ------------------------------------------------------------------
    def train(self) -> EnergyMLP:
        best_loss = float("inf")
        best_state = None
        patience = 0

        for epoch in range(self.cfg.num_epochs):
            epoch_loss = 0.0
            batches = 0

            for batch in self.loader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()

                # DSM loss
                loss = _dsm_loss(
                    model=self.model,
                    x=batch,
                    sigma=self.cfg.sigma,
                )

                # Riemannian regularizer
                if self.riem_weight > 0.0 and self.riem_type != "none":
                    if self.riem_type == "hess_smooth":
                        reg = hessian_smoothness_penalty(
                            x=batch,
                            energy_model=self.model,
                            eps=self.riem_eps,
                            n_dirs=self.riem_n_dirs,
                        )
                    else:
                        reg = 0.0
                    loss_ratio = (self.riem_weight * reg) / loss
                    print(
                        f"[Energy DSM] Epoch {epoch+1}/{self.cfg.num_epochs}  "                        
                        f"Loss ratio={loss_ratio:.6e}",
                        flush=True
                    )
                    
                    loss = loss + self.riem_weight * reg
                
                loss.backward()

                # grad clipping
                if self.cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.max_grad_norm,
                    )

                self.optimizer.step()

                epoch_loss += float(loss.item())
                batches += 1

            epoch_loss /= max(batches, 1)

            print(
                f"[Energy DSM] Epoch {epoch+1}/{self.cfg.num_epochs}  "
                f"loss={epoch_loss:.6e}",
                flush=True,
            )

            # early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = self.model.state_dict()
                patience = 0
            else:
                patience += 1
            if self.cfg.early_stop_patience > 0 and patience >= self.cfg.early_stop_patience:
                print(f"[Energy DSM] Early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)


    if log_manifest:
        # A few useful extras (totally optional)
        extra = {
            "data": {
                "n_cells": int(ad_prep.n_obs),
                "n_genes": int(ad_prep.n_vars),
            },
            "riemann": {
                "type": train_cfg.riemann_reg_type,
                "weight": float(train_cfg.riemann_reg_weight),
                "eps": float(train_cfg.riemann_eps),
                "n_dirs": int(train_cfg.riemann_n_dirs),
            },
            "training": {
                "num_epochs": int(train_cfg.num_epochs),
                "sigma": float(train_cfg.sigma),
                "batch_size": int(train_cfg.batch_size),
                "device": train_cfg.device,
            },
        }
        log_config(model_cfg=model_cfg, train_cfg=train_cfg, extra=extra)

        return self.model


# ---------------------------------------------------------------------
# Functional API, still supported
# ---------------------------------------------------------------------

def train_energy_model(
    ad_prep: sc.AnnData,
    model_cfg: EnergyModelConfig | Mapping[str, Any],
    train_cfg: EnergyTrainConfig | Mapping[str, Any],
) -> EnergyMLP:

    model_cfg = _ensure_model_cfg(model_cfg)
    train_cfg = _ensure_train_cfg(train_cfg)

    dataset = AnnDataExpressionDataset(ad_prep.X)

    model = EnergyMLP(
        n_genes=ad_prep.n_vars,
        hidden_dims=model_cfg.hidden_dims,
        latent_dim=model_cfg.latent_dim,
    )

    trainer = EnergyTrainer(model, dataset, train_cfg)
    return trainer.train()
