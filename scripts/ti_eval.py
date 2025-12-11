#!/usr/bin/env python3
"""
scripts/ti_eval.py

Config-driven TI evaluation.

All runtime config is read from a params YAML, e.g.:

ti_eval:
  ad_path: data/embedding/weinreb_eggfm_diffmap.h5ad
  embedding_key: X_eggfm_diffmap
  time_key: "Time point"
  cluster_key: "Cell type annotation"
  fate_key: null
  baseline_embedding_key: null
  root_mask_key: null
  n_neighbors: 30
  n_dcs: 10
  max_cells: 60000        # optional; will be clipped to <= 60000
  out_dir: out/metrics/ti
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore
import yaml

from medit.ti import evaluate_embedding_for_ti


# -----------------------------
# Small helpers
# -----------------------------


def _to_plain(o: Any) -> Any:
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    try:
        if hasattr(o, "tolist"):
            return o.tolist()
    except Exception:
        pass
    return o


def _allocate_run_dir(base_out_dir: Path, base_run_id: str) -> Path:
    run_dir = base_out_dir / base_run_id
    if not run_dir.exists():
        return run_dir

    siblings = [
        p.name
        for p in base_out_dir.iterdir()
        if p.is_dir() and p.name.startswith(base_run_id)
    ]

    max_idx = 1
    for name in siblings:
        if name == base_run_id:
            max_idx = max(max_idx, 1)
        elif name.startswith(base_run_id + "__r"):
            tail = name.split("__r", 1)[-1]
            try:
                idx = int(tail)
                max_idx = max(max_idx, idx)
            except ValueError:
                continue

    next_idx = max_idx + 1
    new_name = f"{base_run_id}__r{next_idx}"
    return base_out_dir / new_name


def _infer_time_key(ad: sc.AnnData, explicit: Optional[str]) -> Optional[str]:
    if explicit is not None:
        return explicit
    candidates = [
        "Time point",
        "time",
        "timepoint",
        "pseudotime",
        "t",
        "day",
    ]
    obs_cols = list(ad.obs.columns)
    for c in candidates:
        if c in obs_cols:
            return c
    return None


def _infer_cluster_key(ad: sc.AnnData, explicit: Optional[str]) -> Optional[str]:
    if explicit is not None:
        return explicit
    candidates = [
        "Cell type annotation",
        "cell_type",
        "celltype",
        "cluster",
        "clusters",
        "louvain",
        "leiden",
    ]
    obs_cols = list(ad.obs.columns)
    for c in candidates:
        if c in obs_cols:
            return c
    return None


def _extract_params_subset(params: Dict[str, Any]) -> Dict[str, Any]:
    keys_of_interest = [
        "eggfm_model",
        "eggfm_train",
        "eggfm_diffmap",
        "embedding",
        "spec",
        "ti_eval",
    ]
    subset: Dict[str, Any] = {}
    for k in keys_of_interest:
        if k in params:
            subset[k] = params[k]
    return subset


def _subsample_ad(
    ad: sc.AnnData,
    max_cells: Optional[int],
    random_state: int = 0,
) -> sc.AnnData:
    """
    Subsample up to max_cells cells from ad, if requested.

    max_cells is assumed already clipped to <= 60000 by caller.
    """
    print(f"[TI] _subsample_ad: incoming n_obs = {ad.n_obs}, max_cells = {max_cells}")
    if max_cells is None or max_cells <= 0 or ad.n_obs <= max_cells:
        print(f"[TI] _subsample_ad: using all {ad.n_obs} cells")
        return ad.copy()

    rng = np.random.default_rng(random_state)
    idx = rng.choice(ad.n_obs, size=max_cells, replace=False)
    ad_sub = ad[idx].copy()
    print(
        f"[TI] _subsample_ad: subsampled {ad.n_obs} -> {ad_sub.n_obs} cells "
        f"(max_cells={max_cells})"
    )
    return ad_sub


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Config-driven TI evaluation: everything comes from params.yml."
    )
    p.add_argument(
        "--params",
        type=str,
        default="configs/params.yml",
        help="Params YAML with a ti_eval block (default: configs/params.yml).",
    )
    p.add_argument(
        "--cfg-key",
        type=str,
        default="ti_eval",
        help="YAML block key for TI config (default: ti_eval).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    params_path = Path(args.params)
    params: Dict[str, Any] = yaml.safe_load(params_path.read_text())
    ti_cfg: Dict[str, Any] = params.get(args.cfg_key, {})

    if not ti_cfg:
        raise RuntimeError(
            f"No '{args.cfg_key}' block found in {params_path}. "
            "Add a ti_eval: section with ad_path, embedding_key, etc."
        )

    # --- Pull config from ti_eval block ---
    ad_path = Path(ti_cfg["ad_path"])
    embedding_key = ti_cfg["embedding_key"]

    time_key_cfg = ti_cfg.get("time_key", None)
    cluster_key_cfg = ti_cfg.get("cluster_key", None)
    fate_key = ti_cfg.get("fate_key", None)
    baseline_embedding_key = ti_cfg.get("baseline_embedding_key", None)
    root_mask_key = ti_cfg.get("root_mask_key", None)

    n_neighbors = int(ti_cfg.get("n_neighbors", 30))
    n_dcs = int(ti_cfg.get("n_dcs", 10))

    # max_cells from config, clipped to <= 60000
    max_cells_raw = ti_cfg.get("max_cells", None)
    if max_cells_raw is None:
        max_cells: Optional[int] = None
    else:
        max_cells = int(max_cells_raw)
        # if max_cells > 120000:
        #     print(
        #         f"[TI] max_cells from config = {max_cells} > 60000; "
        #         f"clipping to 60000"
        #     )
        #     max_cells = 60000

    out_dir = Path(ti_cfg.get("out_dir", "out/metrics/ti"))

    # --- Load AnnData ---
    print(f"[TI] Reading AnnData from: {ad_path}")
    ad = sc.read_h5ad(ad_path)
    print(f"[TI] Loaded AnnData: n_obs={ad.n_obs}, n_vars={ad.n_vars}")

    # --- Optional subsample ---
    ad_eval = _subsample_ad(ad, max_cells=max_cells, random_state=0)
    print(f"[TI] After subsample: n_obs={ad_eval.n_obs}")

    # --- Infer time/cluster if not forced in config ---
    time_key = _infer_time_key(ad_eval, time_key_cfg)
    cluster_key = _infer_cluster_key(ad_eval, cluster_key_cfg)

    print(f"[TI] Using embedding_key = {embedding_key}")
    print(f"[TI] time_key    = {time_key}")
    print(f"[TI] cluster_key = {cluster_key}")
    print(f"[TI] n_neighbors = {n_neighbors}, n_dcs = {n_dcs}")

    # --- Run evaluation ---
    metrics = evaluate_embedding_for_ti(
        ad=ad_eval,
        embedding_key=embedding_key,
        time_key=time_key,
        cluster_key=cluster_key,
        fate_key=fate_key,
        baseline_embedding_key=baseline_embedding_key,
        root_mask_key=root_mask_key,
        n_neighbors=n_neighbors,
        n_dcs=n_dcs,
        random_state=0,
    )

    dataset_stem = ad_path.stem
    emb_key_clean = embedding_key.replace(" ", "_")
    base_run_id = f"{dataset_stem}__{emb_key_clean}"

    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _allocate_run_dir(out_dir, base_run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    run_id = run_dir.name

    params_subset = _extract_params_subset(params)

    meta: Dict[str, Any] = {
        "run_id": run_id,
        "base_run_id": base_run_id,
        "ad_path": str(ad_path),
        "embedding_key": embedding_key,
        "time_key": time_key,
        "cluster_key": cluster_key,
        "fate_key": fate_key,
        "baseline_embedding_key": baseline_embedding_key,
        "root_mask_key": root_mask_key,
        "n_neighbors": float(n_neighbors),
        "n_dcs": float(n_dcs),
        "max_cells": int(max_cells) if max_cells is not None else None,
        "eval_n_cells": int(ad_eval.n_obs),
        "params_path": str(params_path),
    }

    json_path = run_dir / "metrics.json"
    csv_path = run_dir / "metrics.csv"
    manifest_path = run_dir / "manifest.json"

    payload = {
        "meta": meta,
        "metrics": metrics,
        "params": params_subset,
    }

    manifest_payload = {
        "run_id": run_id,
        "base_run_id": base_run_id,
        "ad_path": str(ad_path),
        "embedding_key": embedding_key,
        "time_key": time_key,
        "cluster_key": cluster_key,
        "params_path": str(params_path),
        "params": params_subset,
    }

    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, default=_to_plain)

    with manifest_path.open("w") as f:
        json.dump(manifest_payload, f, indent=2, default=_to_plain)

    row: Dict[str, Any] = {**meta}
    for k, v in metrics.items():
        if isinstance(v, (np.ndarray, list)):
            continue
        row[k] = _to_plain(v)
    df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)

    print(json.dumps(payload, indent=2, default=_to_plain))
    print(f"\n[TI] Wrote JSON metrics to {json_path}")
    print(f"[TI] Wrote CSV metrics to  {csv_path}")
    print(f"[TI] Wrote params manifest to {manifest_path}")


if __name__ == "__main__":
    main()
