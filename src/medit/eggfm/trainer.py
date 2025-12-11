from __future__ import annotations

from typing import Dict, Any, Mapping
from pathlib import Path
from datetime import datetime
import json
import subprocess

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.cluster import KMeans
import scanpy as sc 

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
    *,
    dataset_n_cells: int | None = None,
    dataset_n_genes: int | None = None,
    extra: Dict[str, Any] | None = None,
) -> Path:
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

    if dataset_n_cells is not None or dataset_n_genes is not None:
        payload["data"] = {
            "n_cells": dataset_n_cells,
            "n_genes": dataset_n_genes,
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
        if train_cfg.device in (None, "auto"):
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = train_cfg.device
        self.device = torch.device(device_str)

        # model
        self.model = model.to(self.device)

        self.dataset = dataset
        
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

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------

    def _set_uniform_loader(self) -> None:
        """Original behavior: uniform sampling with shuffle=True."""
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def _compute_energies(self, batch_size: int = 4096) -> np.ndarray:
        """Compute E(x_i) for all cells under the current model."""
        self.model.eval()
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        energies = []
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                e = self.model(x)
                if e.ndim > 1:
                    e = e.squeeze(-1)
                energies.append(e.cpu().numpy())
        self.model.train()
        return np.concatenate(energies, axis=0)  # [N]

    def _make_weights_for_step(self, step_idx: int, n_steps: int) -> torch.Tensor:
        """
        Simple SNIS-style weights:
        - cluster with k-means
        - within each cluster, upweight high-energy points
        - anneal beta over refinement steps
        """
        N = len(self.dataset)

        # anneal beta from start -> end
        beta0 = self.cfg.refinement_beta_start
        beta1 = self.cfg.refinement_beta_end
        beta = beta0 + (beta1 - beta0) * (step_idx / max(n_steps - 1, 1))

        # energy and clipping
        E = self._compute_energies()
        E = np.clip(E, -self.cfg.refinement_energy_clip, self.cfg.refinement_energy_clip)
        E = (E - E.mean()) / (E.std() + 1e-6)

        # cluster for stratification
        X = self.dataset.X  # assuming AnnDataExpressionDataset exposes .X
        k = self.cfg.refinement_n_clusters
        km = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = km.fit_predict(X)

        weights = np.zeros(N, dtype=np.float64)
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                continue
            E_j = E[mask]
            w_j = np.exp(beta * E_j)
            w_j = np.maximum(w_j, self.cfg.refinement_weight_floor)
            w_j /= w_j.sum()
            weights[mask] = w_j

        weights /= weights.sum()
        return torch.from_numpy(weights)

    def _set_weighted_loader(self, step_idx: int, n_steps: int) -> None:
        """Rebuild self.loader with a WeightedRandomSampler."""
        weights = self._make_weights_for_step(step_idx, n_steps)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=False,
        )
    # ------------------------------------------------------------------
    def train(self) -> EnergyMLP:
        best_loss = float("inf")
        best_state = None
        patience = 0
        
         # Fallback to 1 if the field is missing
        n_steps = max(int(getattr(self.cfg, "n_refinement_steps", 1)), 1)
        epochs_per_step = max(self.cfg.num_epochs // n_steps, 1)

        for step in range(n_steps):
            if n_steps == 1:
                self._set_uniform_loader()
            else:
                self._set_weighted_loader(step, n_steps)

            print(f"[Energy DSM] Refinement step {step+1}/{n_steps}", flush=True)
            
            for local_epoch in range(epochs_per_step):
                # Optional: global epoch index, if you want consistent logging
                global_epoch = step * epochs_per_step + local_epoch

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
                    f"[Energy DSM] Epoch {global_epoch+1}/{self.cfg.num_epochs}  "
                    f"loss={epoch_loss:.6e}",
                    flush=True,
                )

                # early stopping
                if epoch_loss < best_loss - self.cfg.early_stop_min_delta:
                    best_loss = epoch_loss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    patience = 0
                else:
                    patience += 1

                if (
                    self.cfg.early_stop_patience > 0
                    and patience >= self.cfg.early_stop_patience
                ):
                    print(f"[Energy DSM] Early stop at epoch {global_epoch+1}")
                    break

            # propagate early stop out of refinement loop
            if (
                self.cfg.early_stop_patience > 0
                and patience >= self.cfg.early_stop_patience
            ):
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.model

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
    trained_model = trainer.train()
    log_config(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        dataset_n_cells=int(ad_prep.n_obs),
        dataset_n_genes=int(ad_prep.n_vars),
    )
    return trained_model
