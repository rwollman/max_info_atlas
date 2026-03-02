"""
Clustering comparison utilities.

Provides tools for computing pairwise Adjusted Rand Index (ARI) between
clustering configurations, with support for chunked/parallel execution.
"""

from .ari import (
    build_comparison_pairs,
    pairs_to_job_list,
    compute_ari_for_chunk,
    combine_ari_chunks,
    compute_pairwise_ari,
    discover_sections,
    load_labels_for_config,
    n_pairs,
    n_chunks,
)
from .config import AriConfig

__all__ = [
    "AriConfig",
    "build_comparison_pairs",
    "pairs_to_job_list",
    "compute_ari_for_chunk",
    "combine_ari_chunks",
    "compute_pairwise_ari",
    "discover_sections",
    "load_labels_for_config",
    "n_pairs",
    "n_chunks",
]
