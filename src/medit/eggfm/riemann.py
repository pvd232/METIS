from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def hessian_smoothness_penalty(
    x: torch.Tensor,
    energy_model: nn.Module,
    eps: float = 1e-2,
    n_dirs: int = 4,
) -> torch.Tensor:
    """
    Very simple Riemannian-ish regularizer: penalize squared directional
    second derivatives of the energy along random directions.

    For each batch of points x, we:
      - sample random directions v
      - compute finite-difference approx of second directional derivative:
          d^2E/dv^2 ≈ (E(x+eps v) - 2E(x) + E(x-eps v)) / eps^2
      - penalize its squared magnitude.

    This encourages smoother curvature of the learned energy landscape
    along typical directions, i.e. a softer Riemannian metric.
    """
    B, D = x.shape
    device = x.device

    # sample n_dirs random directions per cell
    # shape: [n_dirs, B, D]
    v = torch.randn(n_dirs, B, D, device=device)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    # expand x to [n_dirs, B, D]
    x_exp = x.unsqueeze(0).expand(n_dirs, B, D)

    # compute energies at x, x+eps v, x-eps v
    x_plus = x_exp + eps * v
    x_minus = x_exp - eps * v

    # Flatten batch for model calls
    x_flat = x_exp.reshape(-1, D)
    x_plus_flat = x_plus.reshape(-1, D)
    x_minus_flat = x_minus.reshape(-1, D)

    # Assumes energy_model(x) -> [N] or [N,1]
    E0 = energy_model(x_flat).view(n_dirs, B)
    Ep = energy_model(x_plus_flat).view(n_dirs, B)
    Em = energy_model(x_minus_flat).view(n_dirs, B)

    second_dir = (Ep - 2.0 * E0 + Em) / (eps ** 2)  # [n_dirs, B]
    penalty = (second_dir ** 2).mean()
    return penalty
