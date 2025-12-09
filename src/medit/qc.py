# src/medit/qc.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import scanpy as sc  # type: ignore
from scipy import sparse
import yaml


def _maybe_add_mt_qc_metrics(ad: sc.AnnData) -> None:
    """
    Ensure we have a mitochondrial percentage column if possible.

    - If ad.obs already has 'pct_counts_mt' or 'mitopercent', do nothing.
    - Otherwise, try to infer MT genes by prefix 'MT-' and run
      sc.pp.calculate_qc_metrics to populate 'pct_counts_mt'.
    """
    if ("pct_counts_mt" in ad.obs) or ("mitopercent" in ad.obs):
        return

    mt_mask = ad.var_names.str.upper().str.startswith("MT-")
    if mt_mask.any():
        ad.var["mt"] = mt_mask
        sc.pp.calculate_qc_metrics(
            ad,
            qc_vars=["mt"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )


def prep(ad: sc.AnnData, params: Dict[str, Any]) -> sc.AnnData:
    """
    Core QC + HVG selection + normalization on an in-memory AnnData.

    Expects in params:
      qc:
        min_genes: int
        max_pct_mt: float   # optional; if missing, no mito filter
        min_cells: int      # optional; fallback is 0.1% of cells
      hvg_n_top_genes: int
    """
    n_cells = ad.n_obs
    qc_cfg = params.get("qc", {})

    # -------------------------
    # 1. Gene + cell filters
    # -------------------------
    min_cells_cfg = int(qc_cfg.get("min_cells", 0))
    if min_cells_cfg > 0:
        min_cells = min_cells_cfg
    else:
        min_cells = max(3, int(0.001 * n_cells))

    print(f"[MEDIT.QC] filter_genes min_cells={min_cells}", flush=True)
    sc.pp.filter_genes(ad, min_cells=min_cells)

    min_genes = int(qc_cfg["min_genes"])
    print(f"[MEDIT.QC] filter_cells min_genes={min_genes}", flush=True)
    sc.pp.filter_cells(ad, min_genes=min_genes)

    # Drop zero-count cells
    totals = np.ravel(ad.X.sum(axis=1))
    ad = ad[totals > 0, :].copy()

    # -------------------------
    # 2. Optional mito filter
    # -------------------------
    _maybe_add_mt_qc_metrics(ad)

    max_pct_mt = float(qc_cfg.get("max_pct_mt", 1.0))
    if max_pct_mt < 1.0:
        mt_cols = [c for c in ("pct_counts_mt", "mitopercent") if c in ad.obs]
        if mt_cols:
            mt_col = mt_cols[0]
            before = ad.n_obs
            ad = ad[ad.obs[mt_col] < max_pct_mt].copy()
            after = ad.n_obs
            print(
                f"[MEDIT.QC] mito filter {mt_col} < {max_pct_mt}: "
                f"{before} -> {after} cells",
                flush=True,
            )
        else:
            print(
                "[MEDIT.QC] max_pct_mt set but no mito column found in ad.obs; "
                "skipping mito filter.",
                flush=True,
            )

    print(
        f"[MEDIT.QC] after basic filters: n_obs={ad.n_obs}, n_vars={ad.n_vars}",
        flush=True,
    )

    # -------------------------
    # 3. Preserve raw counts
    # -------------------------
    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()
        print("[MEDIT.QC] Saved raw counts into ad.layers['counts']", flush=True)

    # -------------------------
    # 4. HVG selection
    # -------------------------
    hvg_n_top = int(params["hvg_n_top_genes"])
    print(f"[MEDIT.QC] Computing HVGs with n_top_genes={hvg_n_top}", flush=True)

    sc.pp.highly_variable_genes(
        ad,
        n_top_genes=hvg_n_top,
        flavor="seurat_v3",
        subset=False,
    )

    n_hvg = int(ad.var["highly_variable"].sum())
    print(
        f"[MEDIT.QC] HVGs flagged={n_hvg} (requested n_top_genes={hvg_n_top}); "
        f"n_vars BEFORE subset={ad.n_vars}",
        flush=True,
    )

    ad = ad[:, ad.var["highly_variable"]].copy()
    print(f"[MEDIT.QC] n_vars AFTER HVG subset={ad.n_vars}", flush=True)

    # -------------------------
    # 5. Normalize / log-transform
    # -------------------------
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    return ad


def run_qc(
    params_path: Path,
    ad_path: Path,
    out_path: Path,
) -> sc.AnnData:
    """
    High-level QC entrypoint for Weinreb-style datasets (no perturbation split).
    """
    params: Dict[str, Any] = yaml.safe_load(params_path.read_text())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ad_full = sc.read_h5ad(str(ad_path), backed="r")
    print(
        f"[MEDIT.QC] load full AnnData: n_obs={ad_full.n_obs}, n_vars={ad_full.n_vars}",
        flush=True,
    )

    ad = ad_full.to_memory()
    ad.obs_names_make_unique()
    # ensure sparse CSR for downstream ops
    if not sparse.issparse(ad.X):
        ad.X = sparse.csr_matrix(ad.X)

    qc_ad = prep(ad, params)

    print(f"[MEDIT.QC] writing QC AnnData to {out_path}", flush=True)
    qc_ad.write_h5ad(out_path)

    return qc_ad
