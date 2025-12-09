# src/medit/ti/__init__.py

"""
Trajectory inference (TI) evaluation helpers.

This module provides utilities to:
  - run diffusion pseudotime on a given embedding,
  - compare pseudotime to ground-truth time / stage labels,
  - assess how well clusters capture fate labels,
  - quantify how much a candidate embedding's kNN graph
    deviates from a baseline embedding (graph-preservation).
"""

from .eval import (
    compute_pseudotime_metrics,
    compute_branch_metrics,
    knn_graph_overlap,
    evaluate_embedding_for_ti,
)
