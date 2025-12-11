#!/usr/bin/env python3
"""
scripts/summarize_ti_metrics.py

Scan out/metrics/ti/**/metrics.json and aggregate:
  - metric_mode (scm, riem_tangent, riem_curvature, riem_normal, etc.)
  - SCM hyperparams (metric_gamma, metric_lambda, energy_clip_abs)
  - Riemannian hyperparams (tangent_dim, tangent_eps, curvature_scale, normal_weight, ...)
  - TI metrics (pt_variance, pt_spearman_vs_time, pt_kendall_vs_time, eval_n_cells)

into a single CSV:

  out/metrics/ti_ablation_summary.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


TI_ROOT = Path("out/metrics/ti")
OUT_CSV = Path("out/metrics/ti_ablation_summary6.csv")


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def find_metrics_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/metrics.json"))


def main() -> None:
    # Helpful debug info
    print(f"[SUMMARY] CWD: {Path.cwd().resolve()}")
    print(f"[SUMMARY] Looking for TI metrics under: {TI_ROOT.resolve()}")

    if not TI_ROOT.exists():
        print(f"[SUMMARY] No TI directory found at {TI_ROOT.resolve()}")
        return

    files = find_metrics_files(TI_ROOT)
    if not files:
        print(f"[SUMMARY] No metrics.json files found under {TI_ROOT.resolve()}")
        return

    print(f"[SUMMARY] Found {len(files)} metrics.json files")

    rows: List[Dict[str, Any]] = []

    for path in files:
        try:
            payload = json.loads(path.read_text())
        except Exception as e:
            print(f"[SUMMARY] WARNING: failed to read {path}: {e}")
            continue

        meta = payload.get("meta", {})
        metrics = payload.get("metrics", {})
        params = payload.get("params", {})

        eggfm_diffmap = params.get("eggfm_diffmap", {})
        eggfm_train = params.get("eggfm_train", {})
        # embedding = params.get("embedding", {})  # not used yet, but kept if needed

        row: Dict[str, Any] = {}

        # --------- identify run ----------
        row["metrics_path"] = str(path)
        row["run_id"] = meta.get("run_id")
        row["base_run_id"] = meta.get("base_run_id")
        row["ad_path"] = meta.get("ad_path")
        row["embedding_key"] = meta.get("embedding_key")

        # --------- TI meta ----------
        row["eval_n_cells"] = meta.get("eval_n_cells")
        row["n_neighbors"] = meta.get("n_neighbors")
        row["n_dcs"] = meta.get("n_dcs")
        row["max_cells"] = meta.get("max_cells")

        # --------- EGGFM / diffusion config ----------
        row["metric_mode"] = eggfm_diffmap.get("metric_mode")
        row["geometry_source"] = eggfm_diffmap.get("geometry_source")
        row["energy_source"] = eggfm_diffmap.get("energy_source")
        row["t"] = eggfm_diffmap.get("t")
        row["distance_power"] = eggfm_diffmap.get("distance_power")

        # SCM hyperparams
        row["metric_gamma"] = eggfm_diffmap.get("metric_gamma")
        row["metric_lambda"] = eggfm_diffmap.get("metric_lambda")
        row["energy_clip_abs"] = eggfm_diffmap.get("energy_clip_abs")
        row["energy_batch_size"] = eggfm_diffmap.get("energy_batch_size")

        # Riemannian hyperparams (if present)
        row["tangent_dim"] = eggfm_diffmap.get("tangent_dim")
        row["tangent_eps"] = eggfm_diffmap.get("tangent_eps")
        row["tangent_k"] = eggfm_diffmap.get("tangent_k")
        row["curvature_k"] = eggfm_diffmap.get("curvature_k")
        row["curvature_scale"] = eggfm_diffmap.get("curvature_scale")
        row["normal_k"] = eggfm_diffmap.get("normal_k")
        row["normal_weight"] = eggfm_diffmap.get("normal_weight")

        # Training config (optional but handy)
        row["train_sigma"] = eggfm_train.get("sigma")
        row["train_lr"] = eggfm_train.get("lr")
        row["train_batch_size"] = eggfm_train.get("batch_size")
        row["train_riemann_reg_type"] = eggfm_train.get("riemann_reg_type")
        row["train_riemann_reg_weight"] = eggfm_train.get("riemann_reg_weight")

        # --------- TI metrics ----------
        row["pt_variance"] = metrics.get("pt_variance")
        row["pt_min"] = metrics.get("pt_min")
        row["pt_max"] = metrics.get("pt_max")
        row["pt_spearman_vs_time"] = metrics.get("pt_spearman_vs_time")
        row["pt_kendall_vs_time"] = metrics.get("pt_kendall_vs_time")

        rows.append(row)

    if not rows:
        print("[SUMMARY] Parsed 0 rows from metrics.json files — nothing to write.")
        return

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"[SUMMARY] Wrote {len(df)} rows to {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
