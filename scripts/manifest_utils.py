#!/usr/bin/env python3
# scripts/manifest_utils.py
"""
Helpers for writing simple JSON manifests for embedding .h5ad files.

Typical layout for an embedding:

  data/interim/weinreb_embedding.h5ad
  data/interim/weinreb_embedding_manifest/
    manifest.json
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import scanpy as sc  # type: ignore


def _short_git_sha() -> str:
    """Return the current git short SHA, or 'unknown' if it fails."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _summarize_ad(ad: sc.AnnData, path: Path | None = None) -> dict[str, Any]:
    """
    Lightweight summary of an AnnData, focused on obsm shapes.
    """
    obsm: dict[str, Any] = {}
    for key, val in ad.obsm.items():
        shape = getattr(val, "shape", None)
        if shape is not None:
            obsm[key] = {"shape": list(shape)}

    info: dict[str, Any] = {
        "n_obs": int(ad.n_obs),
        "n_vars": int(ad.n_vars),
        "obsm": obsm,
    }
    if path is not None:
        info["path"] = str(path)
    return info


def _summarize_h5ad(path: Path) -> dict[str, Any]:
    """Load a .h5ad and return a summary dict."""
    ad = sc.read_h5ad(path)
    return _summarize_ad(ad, path=path)


def write_embedding_manifest(
    *,
    qc_path: Path,
    out_path: Path,
    pca_path: Path | None,
    eggfm_path: Path | None,
    params: Mapping[str, Any] | None = None,
    params_path: Path | None = None,
    cfg_key: str | None = None,
    k: int,
    n_neighbors: int,
    n_cells_sample: int,
    seed: int,
) -> Path:
    """
    Build and write a JSON manifest summarizing embedding artifacts.

    Parameters
    ----------
    qc_path
        Path to the QC .h5ad used as input.
    out_path
        Path to the *main* embedding .h5ad for this run.
        The manifest will be written under:
          <out_path.parent>/<out_path.stem>_manifest/manifest.json
    pca_path
        Optional path to a PCA→Diffmap .h5ad for this run.
    eggfm_path
        Optional path to an EGGFM→Diffmap .h5ad for this run.
    params
        Parsed params.yml as a dict (if available).
    params_path
        Path to the params.yml file used (if you have it).
    cfg_key
        YAML block key for the diffusion config (e.g. "eggfm_diffmap").
    k, n_neighbors, n_cells_sample, seed
        Core embedding hyperparameters for this run.

    Returns
    -------
    Path
        Path to the written manifest JSON.
    """
    # ---- summarize artifacts ----
    qc_info = _summarize_h5ad(qc_path)
    pca_info = _summarize_h5ad(pca_path) if pca_path is not None else None
    eggfm_info = _summarize_h5ad(eggfm_path) if eggfm_path is not None else None

    # ---- base manifest ----
    manifest: dict[str, Any] = {
        "git": _short_git_sha(),
        "qc": qc_info,
        "embedding_h5ad": str(out_path),
        "config": {
            "params_path": str(params_path) if params_path is not None else None,
            "cfg_key": cfg_key,
            "k": int(k),
            "n_neighbors": int(n_neighbors),
            "n_cells_sample": int(n_cells_sample),
            "seed": int(seed),
        },
        "embeddings": {},
    }

    # ---- diffusion block from params.yml, if present ----
    if params is not None and cfg_key and cfg_key in params:
        manifest["config"]["diffusion_block"] = params[cfg_key]

    # ---- embed summaries ----
    if pca_info is not None:
        manifest["embeddings"]["pca_diffmap"] = pca_info
    if eggfm_info is not None:
        manifest["embeddings"]["eggfm_diffmap"] = eggfm_info

    # ---- output path: per-embedding folder ----
    manifest_dir = out_path.parent / f"{out_path.stem}_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest_path
