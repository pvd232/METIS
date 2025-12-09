# src/medit/make_h5ad.py
"""
Build the canonical Weinreb stateFate_inVitro_normed_counts.h5ad from GSM files.

This is the one-shot data builder that:
  - reads the normalized counts matrix (cells × genes),
  - attaches gene names, cell barcodes, and metadata,
  - attaches clone membership matrix (as obsm["X_clone_membership"]),
  - writes a compressed .h5ad in data/raw/.

Usage from Python:

  from pathlib import Path
  from medit.make_h5ad import build_weinreb_h5ad

  out_path = build_weinreb_h5ad(Path("data/raw"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import anndata as ad
import pandas as pd
from scipy.io import mmread  # type: ignore


def build_weinreb_h5ad(
    data_dir: Path,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Build Weinreb stateFate_inVitro_normed_counts.h5ad from raw GSM files.

    Parameters
    ----------
    data_dir:
        Directory containing the GSM4185642_* files.
    out_path:
        Optional explicit output path. If None, defaults to
        data_dir / "stateFate_inVitro_normed_counts.h5ad".

    Returns
    -------
    out_path : Path
        Path to the written .h5ad file.
    """
    data_dir = data_dir.resolve()

    # 1) Expression matrix (cells x genes)
    expr_path = data_dir / "GSM4185642_stateFate_inVitro_normed_counts.mtx.gz"
    X = mmread(str(expr_path)).tocsr()

    # 2) Gene names (columns)
    gene_names_path = data_dir / "GSM4185642_stateFate_inVitro_gene_names.txt.gz"
    gene_names = (
        pd.read_csv(gene_names_path, header=None, sep="\t")[0]
        .astype(str)
        .values
    )

    # 3) Cell barcodes (rows)
    cell_barcodes_path = data_dir / "GSM4185642_stateFate_inVitro_cell_barcodes.txt.gz"
    cell_barcodes = (
        pd.read_csv(cell_barcodes_path, header=None, sep="\t")[0]
        .astype(str)
        .values
    )

    # 4) Metadata (one row per cell)
    metadata_path = data_dir / "GSM4185642_stateFate_inVitro_metadata.txt.gz"
    metadata = pd.read_csv(metadata_path, sep="\t")

    # Basic shape checks
    if X.shape[0] != len(cell_barcodes):
        raise ValueError(
            f"Cell count mismatch: X.shape[0]={X.shape[0]} "
            f"but len(cell_barcodes)={len(cell_barcodes)}"
        )
    if X.shape[1] != len(gene_names):
        raise ValueError(
            f"Gene count mismatch: X.shape[1]={X.shape[1]} "
            f"but len(gene_names)={len(gene_names)}"
        )
    if metadata.shape[0] != X.shape[0]:
        raise ValueError(
            f"Metadata row mismatch: metadata.shape[0]={metadata.shape[0]} "
            f"but X.shape[0]={X.shape[0]}"
        )

    # 5) Build AnnData
    adata = ad.AnnData(X=X)
    adata.obs_names = cell_barcodes
    adata.var_names = gene_names

    # Attach metadata aligned by obs index
    metadata.index = adata.obs_names
    adata.obs = metadata

    # 6) Optional: clone membership matrix
    clone_path = data_dir / "GSM4185642_stateFate_inVitro_clone_matrix.mtx.gz"
    if clone_path.exists():
        clone_mtx = mmread(str(clone_path)).tocsr()
        if clone_mtx.shape[0] != adata.n_obs:
            raise ValueError(
                f"Clone matrix row mismatch: clone_mtx.shape[0]={clone_mtx.shape[0]} "
                f"but n_obs={adata.n_obs}"
            )
        adata.obsm["X_clone_membership"] = clone_mtx

    # 7) Save as .h5ad
    if out_path is None:
        out_path = data_dir / "stateFate_inVitro_normed_counts.h5ad"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    adata.write_h5ad(str(out_path), compression="gzip")
    print(f"[MEDIT.MAKE_H5AD] Wrote {out_path}")
    return out_path
