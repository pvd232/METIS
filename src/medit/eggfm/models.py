from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class EnergyMLP(nn.Module):
    """
    MLP energy model with an optional latent bottleneck.

    If `latent_dim` is provided, the network is:

        x → hidden MLP → z (latent_dim) → scalar energy

    and `encode(x)` returns `z`.

    If `latent_dim` is None, the last hidden layer is used as the "latent"
    (so `encode` still works), and energy is predicted directly from it.
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dims: Sequence[int],
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()

        in_dim = n_genes
        layers: list[nn.Module] = []

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h

        self.backbone = nn.Sequential(*layers)

        # Optional bottleneck
        self._has_latent = latent_dim is not None
        if latent_dim is not None:
            self.latent_dim = int(latent_dim)
            self.latent_layer = nn.Linear(in_dim, self.latent_dim)
            head_in = self.latent_dim
        else:
            # "latent" is just the last hidden layer
            self.latent_dim = in_dim
            self.latent_layer = None  # type: ignore[assignment]
            head_in = in_dim

        # Final energy head
        self.energy_head = nn.Linear(head_in, 1)

    # ----- core API -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar energy E(x).

        Returns shape: [B] (squeezed last dimension).
        """
        h = self.backbone(x)
        if self._has_latent:
            z = self.latent_layer(h)
        else:
            z = h
        E = self.energy_head(z).squeeze(-1)
        return E

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute latent representation z(x).

        - If latent_dim is set, returns the bottleneck output.
        - Otherwise returns the final hidden layer.
        """
        h = self.backbone(x)
        if self._has_latent:
            return self.latent_layer(h)
        return h
