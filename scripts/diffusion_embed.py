#!/usr/bin/env python3
# scripts/diffusion_embed.py
"""
Compute an EGGFM-based diffusion embedding for a QC'd AnnData.

Thin wrapper around `medit.diffusion.build_diffusion_embedding_from_config`.

Usage (Weinreb example):

  python scripts/diffusion_embed.py \
      --params configs/params.yml \
      --ad data/interim/weinreb_qc.h5ad \
      --energy-ckpt out/models/eggfm/eggfm_energy_weinreb.pt \
      --out data/embedding/weinreb_eggfm_diffmap.h5ad \
      --obsm-key X_eggfm_diffmap
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
import torch
import scanpy as sc  # type: ignore

from medit.eggfm.models import EnergyMLP
from medit.eggfm.config import EnergyModelConfig
from medit.diffusion.config import diffusion_config_from_params
from medit.diffusion.embed import build_diffusion_embedding_from_config


def _load_manifest(manifest_path: Path) -> Any:
    """Load an existing manifest.json (if present), else return a fresh structure."""
    if not manifest_path.exists():
        return []  # default: list of records

    try:
        with manifest_path.open("r") as f:
            data = json.load(f)
    except Exception:
        # Corrupt or unreadable? Start fresh rather than crash.
        return []

    return data


def _update_manifest(
    manifest_path: Path,
    embedding_record: Dict[str, Any],
) -> None:
    """
    Update manifest.json to include / overwrite this embedding record.

    Supports:
      - list-of-records style: [ {...}, {...} ]
      - dict-with-'embeddings' style: {"embeddings": [ {...}, ... ], ...}
    """
    manifest = _load_manifest(manifest_path)

    def records_from_manifest(m: Any) -> List[Dict[str, Any]]:
        if isinstance(m, list):
            return m
        if isinstance(m, dict):
            emb = m.get("embeddings", [])
            if isinstance(emb, list):
                return emb
        return []

    records = records_from_manifest(manifest)
    # Use `id` as primary key for dedup
    new_id = embedding_record.get("id")

    # Replace existing record with same id, or append
    updated = False
    for i, rec in enumerate(records):
        if rec.get("id") == new_id:
            records[i] = embedding_record
            updated = True
            break
    if not updated:
        records.append(embedding_record)

    # Write back in the "natural" structure
    if isinstance(manifest, list):
        out_obj: Any = records
    elif isinstance(manifest, dict):
        manifest["embeddings"] = records
        out_obj = manifest
    else:
        out_obj = records

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(out_obj, f, indent=2)


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
        help=(
            "Path to EGGFM checkpoint "
            "(e.g. out/models/eggfm/eggfm_energy_weinreb.pt)"
        ),
    )
    p.add_argument(
        "--out",
        required=True,
        help=(
            "Output .h5ad with embedding "
            "(e.g. data/embedding/weinreb_eggfm_diffmap.h5ad)"
        ),
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

    # -----------------------------
    # 1) Load params and diffusion config
    # -----------------------------
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())
    diff_cfg = diffusion_config_from_params(params, key=args.cfg_key)

    # -----------------------------
    # 2) Load QC AnnData
    # -----------------------------
    ad = sc.read_h5ad(args.ad)
    print(
        f"[MEDIT.DIFF] Loaded QC AnnData: {ad.n_obs} cells × {ad.n_vars} genes",
        flush=True,
    )

    # -----------------------------
    # 3) Load energy checkpoint + rebuild model
    # -----------------------------
    ckpt = torch.load(args.energy_ckpt, map_location="cpu", weights_only=False)

    model_cfg_dict = ckpt.get("model_cfg", {})
    n_genes_ckpt = ckpt.get("n_genes", None)
    if n_genes_ckpt is None:
        n_genes = ad.n_vars
        print(
            f"[MEDIT.DIFF] Checkpoint missing 'n_genes'; "
            f"assuming n_genes = ad.n_vars = {n_genes}",
            flush=True,
        )
    else:
        n_genes_ckpt = int(n_genes_ckpt)
        if n_genes_ckpt != ad.n_vars:
            raise ValueError(
                f"Mismatch between checkpoint n_genes={n_genes_ckpt} "
                f"and AnnData n_vars={ad.n_vars}. "
                "Make sure you are using the QC/HVG-prepped AnnData "
                "that matches this checkpoint."
            )
        n_genes = n_genes_ckpt

    model_cfg = EnergyModelConfig(**model_cfg_dict)
    energy_model = EnergyMLP(n_genes=n_genes, hidden_dims=model_cfg.hidden_dims)

    state_dict = ckpt.get("state_dict", None)
    if state_dict is None:
        raise KeyError(
            "Checkpoint is missing 'state_dict'. "
            "Expected keys: ['state_dict', 'model_cfg', 'train_cfg', ...]"
        )

    energy_model.load_state_dict(state_dict)
    energy_model.eval()

    # Move model to requested device (if available)
    device_str = diff_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    energy_model.to(device)
    print(f"[MEDIT.DIFF] Using device: {device}", flush=True)

    # -----------------------------
    # 4) Build diffusion embedding
    # -----------------------------
    with torch.no_grad():
        ad_out = build_diffusion_embedding_from_config(
            ad=ad,
            energy_model=energy_model,
            diff_cfg=diff_cfg,
            obsm_key=args.obsm_key,
        )

    # -----------------------------
    # 5) Write output .h5ad
    # -----------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ad_out.write_h5ad(out_path)
    print(f"[MEDIT.DIFF] Wrote embedded AnnData to {out_path}", flush=True)

    # -----------------------------
    # 6) Update manifest in the same directory
    # -----------------------------
    manifest_path = out_path.parent / "manifest.json"
    embedding_id = out_path.stem  # e.g. "weinreb_eggfm_diffmap"

    record: Dict[str, Any] = {
        "id": embedding_id,
        "filename": out_path.name,
        "path": str(out_path),
        "obsm_key": args.obsm_key,
        "method": "eggfm_diffmap",
        "params_path": args.params,
        "energy_ckpt": args.energy_ckpt,
        "cfg_key": args.cfg_key,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    _update_manifest(manifest_path, record)
    print(f"[MEDIT.DIFF] Updated manifest at {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
