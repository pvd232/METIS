# src/medit/eggfm/trainer.py
from __future__ import annotations

from typing import Dict, Any, Mapping

import torch
from torch import optim
from torch.utils.data import DataLoader
import scanpy as sc  # type: ignore

from .models import EnergyMLP
from .dataset import AnnDataExpressionDataset
from .config import EnergyModelConfig, EnergyTrainConfig

def _ensure_model_cfg(cfg: EnergyModelConfig | Mapping[str, Any]) -> EnergyModelConfig:
    if isinstance(cfg, EnergyModelConfig):
        return cfg
    return EnergyModelConfig(**dict(cfg))


def _ensure_train_cfg(cfg: EnergyTrainConfig | Mapping[str, Any]) -> EnergyTrainConfig:
    if isinstance(cfg, EnergyTrainConfig):
        return cfg
    return EnergyTrainConfig(**dict(cfg))

def _dsm_loss(
    model: nn.Module,
    x: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Standard denoising score matching loss.

    x: [B, G] batch on the correct device (does NOT need requires_grad).
    """
    # sample Gaussian noise
    noise = torch.randn_like(x) * sigma

    # noisy input
    x_noisy = x + noise
    x_noisy.requires_grad_(True)

    # energy: shape [B] or [B, 1]
    energy = model(x_noisy)
    if energy.ndim > 1:
        energy = energy.squeeze(-1)

    # score = -∂E/∂x
    score = -torch.autograd.grad(
        outputs=energy.sum(),
        inputs=x_noisy,
        create_graph=True,
    )[0]

    target = -noise / (sigma ** 2)

    loss = 0.5 * (score - target).pow(2).sum(dim=1).mean()
    return loss
def train_energy_model(
    ad_prep: sc.AnnData,
    model_cfg: EnergyModelConfig | Mapping[str, Any],
    train_cfg: EnergyTrainConfig | Mapping[str, Any],
) -> EnergyMLP:
    """
    Train an EnergyMLP on preprocessed AnnData using DSM.

    Parameters
    ----------
    ad_prep
        QC'd + log-normalized AnnData (e.g. weinreb_qc.h5ad).
    model_cfg
        Either:
          - EnergyModelConfig dataclass, or
          - dict with model hyperparameters, e.g.:
              hidden_dims: [256, 256]
    train_cfg
        Either:
          - EnergyTrainConfig dataclass, or
          - dict with training hyperparameters, e.g.:
              batch_size, num_epochs, lr, sigma, etc.

    Behavior is identical to the original implementation when
    passed dicts; the dataclasses are thin wrappers.
    """
    # Normalize configs to dataclasses but keep defaults identical
    model_cfg = _ensure_model_cfg(model_cfg)
    train_cfg = _ensure_train_cfg(train_cfg)

    # ---------------- device ----------------
    # OLD BEHAVIOR:
    # device_str = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if train_cfg.device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = train_cfg.device
    device = torch.device(device_str)

    # ---------------- dataset / loader ----------------
    # IMPORTANT: preserve original behavior: dataset gets ad_prep.X,
    # not the full AnnData. AnnDataExpressionDataset standardizes internally.
    dataset = AnnDataExpressionDataset(ad_prep.X)
    loader = DataLoader(
        dataset,
        batch_size=int(getattr(train_cfg, "batch_size", 256)),
        shuffle=True,
        drop_last=False,
    )

    n_genes = ad_prep.n_vars
    hidden_dims = list(getattr(model_cfg, "hidden_dims", [512, 512]))
    model = EnergyMLP(n_genes=n_genes, hidden_dims=hidden_dims).to(device)

    lr = float(getattr(train_cfg, "lr", 1e-3))
    weight_decay = float(getattr(train_cfg, "weight_decay", 0.0))
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    num_epochs = int(getattr(train_cfg, "num_epochs", 100))
    sigma = float(getattr(train_cfg, "sigma", 0.15))
    max_grad_norm = float(getattr(train_cfg, "max_grad_norm", 5.0))
    early_stop_patience = int(getattr(train_cfg, "early_stop_patience", 0))

    best_loss = float("inf")
    best_state_dict = None
    epochs_without_improve = 0

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            loss = _dsm_loss(model, batch, sigma=sigma)
            loss.backward()

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        print(
            f"[Energy DSM] Epoch {epoch+1}/{num_epochs}  loss={epoch_loss:.6e}",
            flush=True,
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state_dict = model.state_dict()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
            print(
                f"[Energy DSM] Early stopping at epoch {epoch+1} "
                f"(best_loss={best_loss:.6e})",
                flush=True,
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model
