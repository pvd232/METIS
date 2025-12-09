#!/usr/bin/env python3
# scripts/qc.py
"""
Run QC + HVG preprocessing for single-cell RNA-seq AnnData (Weinreb / MEDIT).

This script is a thin wrapper around `medit.qc.pipeline.run_qc` and:
  - loads a raw AnnData file (e.g. Weinreb stateFate_inVitro.h5ad)
  - filters genes and cells based on basic thresholds
  - optionally filters cells by mitochondrial percentage (if configured)
  - computes highly variable genes (HVGs) and log-normalized expression
  - writes a QC'd AnnData to disk in data/interim/

Usage:

  python scripts/qc.py \
      --params configs/params.yml \
      --ad data/raw/stateFate_inVitro.h5ad \
      --out data/interim/weinreb_qc.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

from medit.qc import run_qc


def main() -> None:
    p = argparse.ArgumentParser(
        description="QC + HVG preprocessing for Weinreb (MEDIT)."
    )
    p.add_argument(
        "--params",
        required=True,
        help="Path to configs/params.yml",
    )
    p.add_argument(
        "--ad",
        required=True,
        help="Path to raw .h5ad (e.g. data/raw/stateFate_inVitro.h5ad)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Path to write QC’d .h5ad (e.g. data/interim/weinreb_qc.h5ad)",
    )
    args = p.parse_args()

    qc_ad = run_qc(
        params_path=Path(args.params),
        ad_path=Path(args.ad),
        out_path=Path(args.out),
    )
    print(
        f"[MEDIT.QC] done. QC AnnData shape: {qc_ad.n_obs} cells × {qc_ad.n_vars} genes"
    )


if __name__ == "__main__":
    main()
