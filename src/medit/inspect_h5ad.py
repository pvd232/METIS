# src/medit/inspect_h5ad.py
"""
Utilities for inspecting AnnData (.h5ad) files.

The main entrypoint is `summarize_h5ad`, which:
  - loads the AnnData into memory,
  - prints a human-readable summary (cells, genes, layers),
  - prints a preview of obs/var columns and dtypes,
  - returns a summary dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import scanpy as sc  # type: ignore


def summarize_h5ad(
    ad_path: str | Path,
    *,
    max_obs_cols: int = 20,
    max_var_cols: int = 20,
) -> Dict[str, Any]:
    p = Path(ad_path)
    print(f"[MEDIT.INSPECT] Loading AnnData from: {p}", flush=True)

    ad = sc.read_h5ad(str(p))
    print(
        f"[MEDIT.INSPECT] AnnData shape: {ad.n_obs} cells × {ad.n_vars} genes",
        flush=True,
    )

    layers = list(ad.layers.keys())
    print(f"[MEDIT.INSPECT] Layers: {layers if layers else 'None'}", flush=True)

    obs_cols = list(ad.obs.columns)
    var_cols = list(ad.var.columns)

    print(
        f"[MEDIT.INSPECT] obs columns ({len(obs_cols)} total): "
        f"{obs_cols[:max_obs_cols]}{' ...' if len(obs_cols) > max_obs_cols else ''}",
        flush=True,
    )
    print(
        f"[MEDIT.INSPECT] var columns ({len(var_cols)} total): "
        f"{var_cols[:max_var_cols]}{' ...' if len(var_cols) > max_var_cols else ''}",
        flush=True,
    )

    print("\n[MEDIT.INSPECT] obs dtypes (first 10):", flush=True)
    for col in obs_cols[:10]:
        print(f"  - {col}: {ad.obs[col].dtype}", flush=True)

    print("\n[MEDIT.INSPECT] var dtypes (first 10):", flush=True)
    for col in var_cols[:10]:
        print(f"  - {col}: {ad.var[col].dtype}", flush=True)

    summary: Dict[str, Any] = {
        "path": str(p),
        "n_obs": int(ad.n_obs),
        "n_vars": int(ad.n_vars),
        "layers": layers,
        "obs_cols": obs_cols,
        "var_cols": var_cols,
    }
    return summary
