#!/usr/bin/env python3
# scripts/diffusion_embed.py
"""
Compute an EGGFM-based diffusion embedding for a QC'd AnnData.

Thin wrapper around `medit.diffusion.build_diffusion_embedding_from_config`.

Usage:

  python scripts/diffusion_embed.py \
      --params configs/params.yml \
      --ad data/interim/weinreb_qc.h5ad \
      --energy-ckpt out/models/eggfm/eggfm_energy_weinreb.pt \
      --out data/interim/weinreb_eggfm_diffmap.h5ad \
      --obsm-key X_eggfm_diffmap
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
import torch
import scanpy as sc  # type: ignore

from medit.eggfm.models import EnergyMLP
from medit.eggfm.config import EnergyModelConfig
from medit.diffusion.config import diffusion_config_from_params
from medit.diffusion.embed import build_diffusion_embedding_from_config


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build an EGGFM-based diffusion embedding (MEDIT / Weinreb)."
    )
    p.add_argument(
        "--params",
        required=True,
        help="Path to configs/params.yml",
    )
    p.add_argument(
        "--ad",
        required=True,
        help="QC .h5ad (e.g. data/interim/weinreb_qc.h5ad)",
    )
    p.add_argument(
        "--energy-ckpt",
        required=True,
        help="Path to EGGFM checkpoint (e.g. out/models/eggfm/eggfm_energy_weinreb.pt)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output .h5ad with embedding (e.g. data/interim/weinreb_eggfm_diffmap.h5ad)",
    )
    p.add_argument(
        "--obsm-key",
        default="X_eggfm_diffmap",
        help="obsm key to store embedding (default: X_eggfm_diffmap)",
    )
    p.add_argument(
        "--cfg-key",
        default="eggfm_diffmap",
        help="YAML block key for diffusion config (default: eggfm_diffmap)",
    )

    args = p.parse_args()

    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())
    diff_cfg = diffusion_config_from_params(params, key=args.cfg_key)

    ad = sc.read_h5ad(args.ad)
    print(
        f"[MEDIT.DIFF] Loaded QC AnnData: {ad.n_obs} cells × {ad.n_vars} genes",
        flush=True,
    )

    ckpt = torch.load(args.energy_ckpt, map_location="cpu")
    model_cfg_dict = ckpt.get("model_cfg", {})
    n_genes = int(ckpt.get("n_genes", ad.n_vars))

    model_cfg = EnergyModelConfig(**model_cfg_dict)
    energy_model = EnergyMLP(n_genes=n_genes, hidden_dims=model_cfg.hidden_dims)
    energy_model.load_state_dict(ckpt["state_dict"])
    energy_model.eval()

    ad_out = build_diffusion_embedding_from_config(
        ad=ad,
        energy_model=energy_model,
        diff_cfg=diff_cfg,
        obsm_key=args.obsm_key,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ad_out.write_h5ad(out_path)
    print(f"[MEDIT.DIFF] Wrote embedded AnnData to {out_path}")


if __name__ == "__main__":
    main()