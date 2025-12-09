#!/usr/bin/env python3
# scripts/make_h5ad.py
"""
Build the Weinreb stateFate_inVitro_normed_counts.h5ad from GSM raw files.

This is a thin wrapper around `medit.make_h5ad.build_weinreb_h5ad` and is meant
to be run once (or rarely) to construct the canonical `.h5ad` in `data/raw/`.

Usage:

  # Default paths: data/raw/*.gz -> data/raw/stateFate_inVitro_normed_counts.h5ad
  python scripts/make_h5ad.py

  # Custom data directory and output
  python scripts/make_h5ad.py \
      --data-dir data/raw \
      --out data/raw/stateFate_inVitro_normed_counts.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

from medit.make_h5ad import build_weinreb_h5ad


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build Weinreb stateFate_inVitro_normed_counts.h5ad from GSM files."
    )
    p.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing GSM4185642_stateFate_inVitro_* files (default: data/raw).",
    )
    p.add_argument(
        "--out",
        default=None,
        help=(
            "Optional output .h5ad path. If omitted, defaults to "
            "DATA_DIR/stateFate_inVitro_normed_counts.h5ad."
        ),
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out) if args.out is not None else None

    result = build_weinreb_h5ad(data_dir=data_dir, out_path=out_path)
    print(f"[MEDIT.MAKE_H5AD] Done. Output: {result}")


if __name__ == "__main__":
    main()
