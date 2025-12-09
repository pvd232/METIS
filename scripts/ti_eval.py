#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import scanpy as sc

from medit.ti import evaluate_embedding_for_ti


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an embedding for TI.")
    p.add_argument(
        "--ad",
        type=str,
        required=True,
        help="Path to .h5ad file with embeddings in .obsm.",
    )
    p.add_argument(
        "--embedding-key",
        type=str,
        required=True,
        help="Key in .obsm for the embedding to evaluate.",
    )
    p.add_argument(
        "--time-key",
        type=str,
        default=None,
        help="Optional .obs column with ground-truth time / stage.",
    )
    p.add_argument(
        "--cluster-key",
        type=str,
        default=None,
        help="Optional .obs column with cluster labels for this embedding.",
    )
    p.add_argument(
        "--fate-key",
        type=str,
        default=None,
        help="Optional .obs column with fate / lineage labels.",
    )
    p.add_argument(
        "--baseline-embedding-key",
        type=str,
        default=None,
        help="Optional .obsm key for baseline embedding (for graph overlap).",
    )
    p.add_argument(
        "--root-mask-key",
        type=str,
        default=None,
        help="Optional .obs boolean column marking root cells for DPT.",
    )
    p.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="Number of neighbors for kNN graph.",
    )
    p.add_argument(
        "--n-dcs",
        type=int,
        default=10,
        help="Number of diffusion components for DPT.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="out/metrics/ti",
        help="Directory to write TI metrics (JSON + summary CSV).",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional extra tag to disambiguate runs (e.g. 'seed7').",
    )
    return p.parse_args()


def _to_plain(o: Any) -> Any:
    # For JSON serialization
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    try:
        if hasattr(o, "tolist"):
            return o.tolist()
    except Exception:
        pass
    return o


def _write_json(out_path: Path, meta: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    payload = {"meta": meta, "metrics": metrics}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=_to_plain)


def _append_csv(summary_path: Path, row: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Normalize numpy types
    row_plain = {k: _to_plain(v) for k, v in row.items()}
    df_row = pd.DataFrame([row_plain])

    if summary_path.exists():
        df_old = pd.read_csv(summary_path)
        # align columns (union)
        df_all = pd.concat([df_old, df_row], ignore_index=True, sort=False)
        df_all.to_csv(summary_path, index=False)
    else:
        df_row.to_csv(summary_path, index=False)


def main() -> None:
    args = parse_args()

    ad_path = Path(args.ad)
    ad = sc.read_h5ad(ad_path)

    metrics = evaluate_embedding_for_ti(
        ad=ad,
        embedding_key=args.embedding_key,
        time_key=args.time_key,
        cluster_key=args.cluster_key,
        fate_key=args.fate_key,
        baseline_embedding_key=args.baseline_embedding_key,
        root_mask_key=args.root_mask_key,
        n_neighbors=args.n_neighbors,
        n_dcs=args.n_dcs,
    )

    # Basic meta for provenance
    dataset_stem = ad_path.stem  # e.g. "weinreb_embedding"
    run_id_parts = [dataset_stem, args.embedding_key]
    if args.tag:
        run_id_parts.append(args.tag)
    run_id = "__".join(run_id_parts)

    meta = {
        "run_id": run_id,
        "ad_path": str(ad_path),
        "embedding_key": args.embedding_key,
        "time_key": args.time_key,
        "cluster_key": args.cluster_key,
        "fate_key": args.fate_key,
        "baseline_embedding_key": args.baseline_embedding_key,
        "root_mask_key": args.root_mask_key,
        "n_neighbors": args.n_neighbors,
        "n_dcs": args.n_dcs,
    }

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"{run_id}.json"
    summary_csv = out_dir / "ti_metrics_summary.csv"

    _write_json(json_path, meta, metrics)

    # Flatten for CSV: meta + scalar metrics only
    row = {**meta}
    for k, v in metrics.items():
        # skip large arrays (e.g. pt_values) in CSV
        if isinstance(v, (np.ndarray, list)) and len(getattr(v, "shape", [])) > 0:
            continue
        row[k] = v

    _append_csv(summary_csv, row)

    # Also echo to stdout for quick inspection
    print(json.dumps({"meta": meta, "metrics": metrics}, indent=2, default=_to_plain))


if __name__ == "__main__":
    main()
