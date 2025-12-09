#!/usr/bin/env python3
# scripts/inspect_h5ad.py
"""
Inspect a single AnnData (.h5ad) file.

This script is a thin wrapper around `medit.inspect_h5ad.summarize_h5ad` and:
  - loads an .h5ad from disk,
  - prints a concise summary (cells, genes, layers, obs/var columns),
  - is useful for quickly sanity-checking intermediate artifacts.

Usage:

  python scripts/inspect_h5ad.py \
      --ad data/interim/weinreb_qc.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

from medit.inspect_h5ad import summarize_h5ad


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect a .h5ad file (MEDIT).")
    p.add_argument(
        "--ad",
        required=True,
        help="Path to .h5ad file (e.g. data/interim/weinreb_qc.h5ad)",
    )
    args = p.parse_args()

    summary = summarize_h5ad(Path(args.ad))
    print("\n[MEDIT.INSPECT] Summary dict:")
    for k, v in summary.items():
        if isinstance(v, list) and len(v) > 10:
            print(f"  {k}: {v[:10]} ... (total {len(v)})")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
