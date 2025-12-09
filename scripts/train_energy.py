#!/usr/bin/env python3
# scripts/train_energy.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
import torch
import scanpy as sc  # type: ignore
from scipy import sparse  # type: ignore
import numpy as np

from medit.eggfm import train_energy_model


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train an EGGFM energy model (MEDIT)."
    )
    p.add_argument("--params", required=True, help="configs/params.yml")
    p.add_argument(
        "--ad",
        required=True,
        help="QC .h5ad (e.g. data/interim/weinreb_qc.h5ad)",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output dir for checkpoint (e.g. out/models/eggfm)",
    )
    p.add_argument(
        "--ckpt-name",
        default="eggfm_energy_weinreb.pt",
        help="Checkpoint filename (default: eggfm_energy_weinreb.pt)",
    )
    args = p.parse_args()

    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())
    model_cfg: Dict[str, Any] = params.get("eggfm_model", {})
    train_cfg: Dict[str, Any] = params.get("eggfm_train", {})

    ad_prep = sc.read_h5ad(args.ad)
    print(
        f"[MEDIT.EGGFM] Loaded QC AnnData: "
        f"{ad_prep.n_obs} cells × {ad_prep.n_vars} genes",
        flush=True,
    )

    # ensure dense for stats if needed
    X = ad_prep.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-2, None)

    model = train_energy_model(
        ad_prep=ad_prep,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )

    payload = {
        "state_dict": model.state_dict(),
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "n_genes": ad_prep.n_vars,
        "var_names": ad_prep.var_names.to_list(),
        "mean": mean,
        "std": std,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / args.ckpt_name
    torch.save(payload, ckpt_path)
    print(f"[MEDIT.EGGFM] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
