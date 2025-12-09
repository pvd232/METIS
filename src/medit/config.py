# src/medit/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from scipy import sparse


@dataclass
class WeinrebDatasetConfig:
    """
    Canonical configuration + lightweight schema hints for the
    Weinreb et al. `stateFate_inVitro` AnnData object.

    This reflects:

      - ad.obs columns
      - ad.var index (gene symbols)
      - ad.obsm['X_clone_membership']

    Shape of the *raw* AnnData matrix is (n_cells, n_genes) ≈ (130887, 25289).

    Obs keys (columns in ad.obs)
    ----------------------------
    - `library`:
        Library ID / batch / sub-replicate label
        e.g. "LK_d2", "LSK_d4_1_2", "d6_2_2"
    - `cell_barcode`:
        InDrops cell barcode (unique-ish ID per captured cell)
        e.g. "AAACAAAC-AAACTGGT"
    - `time_point`:
        Time point in days after culture (2.0, 4.0, 6.0)
    - `starting_population`:
        FACS starting population (initial condition)
        "Lin-Kit+Sca1+" (LSK, stem-like) or "Lin-Kit+Sca1-" (LK, progenitor)
    - `cell_type_annotation`:
        Manual / semi-supervised cell type annotation
        e.g. "Undifferentiated", "Neutrophil", "Monocyte", ...
    - `well`:
        Plate / well index (technical batch; 0, 1, 2)
    - `spring_x`, `spring_y`:
        SPRING 2D layout coordinates for visualization (not used for modeling)

    Var / obsm schema
    -----------------
    - ad.var_names:
        Gene symbols, e.g. "0610006L08Rik", "Gata1", "Mpo", ...
    - ad.obsm["X_clone_membership"]:
        Clone membership matrix (cells × clones), typically sparse 0/1.
        Each row encodes clone assignments for that cell.

    Paths
    -----
    This config also centralizes the canonical file locations for MEDIT:

    - raw_path:    raw Weinreb h5ad (normed counts)
    - qc_path:     QC + HVG-restricted AnnData (output of scripts/qc.py)
    - embed_path:  downstream diffusion / manifold embedding h5ad
    """

    # Dataset name / ID
    name: str = "weinreb_stateFate_inVitro"

    # Canonical file layout for this repo
    raw_path: Path = Path("data/raw/stateFate_inVitro_normed_counts.h5ad")
    qc_path: Path = Path("data/interim/weinreb_qc.h5ad")
    embed_path: Path = Path("data/interim/weinreb_eggfm_diffmap.h5ad")

    # Keys in .obs
    library_key: str = "library"
    cell_barcode_key: str = "cell_barcode"
    time_key: str = "time_point"
    starting_pop_key: str = "starting_population"
    cell_type_key: str = "cell_type_annotation"
    well_key: str = "well"
    spring_x_key: str = "spring_x"
    spring_y_key: str = "spring_y"

    # Keys in .obsm
    clone_membership_key: str = "X_clone_membership"

    # Optional: if we ever want to restrict to a subset of time points / types
    allowed_time_points: Optional[list[float]] = None
    allowed_cell_types: Optional[list[str]] = None


# Convenience accessor for scripts / notebooks
WEINREB = WeinrebDatasetConfig()
