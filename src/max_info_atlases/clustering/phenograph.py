"""
PhenoGraph clustering implementation.

PhenoGraph is a variant of Leiden clustering that uses a Jaccard-weighted graph
based on shared neighbors instead of the raw kNN graph. Uses an efficient
cell-centric algorithm with set operations and batching.
"""

import time
import heapq
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import numpy as np

from .base import ClusteringMethod, get_resolution_values, format_resolution_dirname


def _edge_list_to_nn_sets(edge_list: np.ndarray) -> list:
    """Convert edge list to list of neighbor sets per cell."""
    n_nodes = int(edge_list.max()) + 1
    nn_sets = [set() for _ in range(n_nodes)]
    for i, j in edge_list:
        ii, jj = int(i), int(j)
        if ii != jj and jj >= 0:
            nn_sets[ii].add(jj)
    return nn_sets


def compute_jaccard_graph(
    edge_list: np.ndarray,
    k_jaccard: int = 15,
    batch_size: int = 5000,
    cell_timeout: int = 600,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Jaccard-weighted graph from kNN edge list.

    Uses a memory-efficient, cell-centric algorithm: for each cell i, finds
    potential neighbors (cells that share at least one neighbor with i),
    computes Jaccard similarity via set operations, and keeps top-k_jaccard
    per cell.

    Args:
        edge_list: kNN edge list of shape (n_edges, 2) with [source, target]
        k_jaccard: Number of top Jaccard neighbors to keep per cell
        batch_size: Cells per batch for memory management
        cell_timeout: Seconds per cell before skipping (0 = no timeout)

    Returns:
        Tuple of (weighted_edge_list, weights)
    """
    n_nodes = int(edge_list.max()) + 1

    print(f"  Computing Jaccard graph: {n_nodes:,} nodes, k_jaccard={k_jaccard}")

    # Build neighbor sets from edge list
    print(f"  Building neighbor sets...")
    nn_indices_filtered = _edge_list_to_nn_sets(edge_list)
    avg_k = np.mean([len(s) for s in nn_indices_filtered])
    print(f"  Avg neighbors per cell: {avg_k:.1f}")

    top_k_neighbors = defaultdict(list)
    start_time = time.time()
    n_samples = len(nn_indices_filtered)
    total_batches = (n_samples + batch_size - 1) // batch_size
    problem_cells = []

    for batch_idx, batch_start in enumerate(range(0, n_samples, batch_size)):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_time = time.time()
        print(f"  Batch {batch_idx + 1}/{total_batches}: cells {batch_start}-{batch_end}")

        batch_top_k = defaultdict(list)

        for i in range(batch_start, batch_end):
            cell_start = time.time()
            neighbors_i = nn_indices_filtered[i]

            try:
                potential = set()
                for nb in neighbors_i:
                    potential.update(
                        j for j in nn_indices_filtered[nb] if j != i
                    )

                if len(potential) > 1_000_000:
                    problem_cells.append((i, len(potential)))
                    continue

                if cell_timeout and (time.time() - cell_start) > cell_timeout:
                    problem_cells.append((i, -1))
                    continue

                for j in potential:
                    neighbors_j = nn_indices_filtered[j]
                    inter = len(neighbors_i & neighbors_j)
                    union = len(neighbors_i | neighbors_j)
                    sim = inter / union if union > 0 else 0.0

                    if sim > 0:
                        heap = batch_top_k[i]
                        if len(heap) < k_jaccard:
                            heapq.heappush(heap, (-sim, j))
                        elif -sim > heap[0][0]:
                            heapq.heappushpop(heap, (-sim, j))

            except Exception:
                problem_cells.append((i, -2))

            if (i - batch_start) % 500 == 0 and (i - batch_start) > 0:
                print(f"    Processed {i - batch_start}/{batch_end - batch_start}")

        for src, heap in batch_top_k.items():
            current = top_k_neighbors[src]
            for item in heap:
                if len(current) < k_jaccard:
                    heapq.heappush(current, item)
                elif item[0] < current[0][0]:
                    heapq.heappushpop(current, item)

        elapsed = time.time() - batch_time
        print(f"  Batch {batch_idx + 1} completed in {elapsed:.1f}s")

    if problem_cells:
        print(f"  WARNING: {len(problem_cells)} problem cells skipped")

    # Convert to edge list and weights
    edges_list = []
    weights_list = []
    for src, heap in top_k_neighbors.items():
        for neg_sim, dst in sorted(heap):
            edges_list.append([src, dst])
            weights_list.append(-neg_sim)

    elapsed = time.time() - start_time
    print(f"  Jaccard graph done in {elapsed:.1f}s, {len(edges_list):,} edges")

    if not edges_list:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float32)

    edges = np.array(edges_list, dtype=np.int32)
    weights = np.array(weights_list, dtype=np.float32)
    return edges, weights


class PhenoGraphClustering(ClusteringMethod):
    """
    PhenoGraph clustering with Jaccard-weighted graph.

    PhenoGraph builds a kNN graph, then transforms it using Jaccard similarity
    of shared neighbors (top-k per cell) before running Leiden community detection.
    """

    def __init__(
        self,
        resolution: Optional[float] = None,
        resolution_idx: Optional[int] = None,
        n_resolutions: int = 50,
        objective: str = 'modularity',
        k_jaccard: int = 15,
    ):
        """
        Initialize PhenoGraph clustering.

        Args:
            resolution: Resolution parameter (higher = more clusters)
            resolution_idx: Index into standard resolution array (0-49)
            n_resolutions: Number of resolution values in standard array
            objective: Objective function ('modularity' or 'CPM')
            k_jaccard: Top-k Jaccard neighbors per cell
        """
        self.n_resolutions = n_resolutions
        self.objective = objective
        self.k_jaccard = k_jaccard

        # Determine resolution
        if resolution is not None:
            self.resolution = resolution
            self.resolution_idx = None
        elif resolution_idx is not None:
            resolutions = get_resolution_values(n_resolutions)
            self.resolution = resolutions[resolution_idx]
            self.resolution_idx = resolution_idx
        else:
            resolutions = get_resolution_values(n_resolutions)
            self.resolution_idx = n_resolutions // 2
            self.resolution = resolutions[self.resolution_idx]

    def fit(self, edge_list: np.ndarray) -> np.ndarray:
        """
        Fit PhenoGraph clustering on a kNN edge list.

        Transforms the kNN graph using Jaccard similarity (top-k per cell),
        then runs Leiden clustering on the weighted graph.

        Args:
            edge_list: kNN edge list of shape (n_edges, 2) with [source, target]

        Returns:
            Array of cluster assignments
        """
        import igraph

        jaccard_edges, jaccard_weights = compute_jaccard_graph(
            edge_list,
            k_jaccard=self.k_jaccard,
        )

        n_nodes = int(edge_list.max()) + 1
        return self.fit_weighted_graph(jaccard_edges, jaccard_weights, n_nodes)

    def fit_weighted_graph(
        self,
        weighted_edges: np.ndarray,
        weights: np.ndarray,
        n_nodes: int,
    ) -> np.ndarray:
        """
        Fit PhenoGraph clustering on a precomputed weighted graph.
        """
        import igraph

        graph = igraph.Graph(n=n_nodes, edges=weighted_edges.tolist())
        graph.es['weight'] = weights.tolist()

        partition = graph.community_leiden(
            resolution_parameter=self.resolution,
            objective_function=self.objective,
            weights='weight',
        )

        return np.array(partition.membership, dtype=np.int32)
    
    def get_params(self) -> Dict[str, Any]:
        """Get clustering parameters."""
        return {
            'method': 'phenograph',
            'resolution': self.resolution,
            'resolution_idx': self.resolution_idx,
            'objective': self.objective,
            'k_jaccard': self.k_jaccard,
        }


def run_phenograph_on_file(
    input_npy: Union[str, Path],
    output_folder: Union[str, Path],
    resolution_idx: int,
    sections_path: Optional[Union[str, Path]] = None,
    k_jaccard: int = 15,
) -> None:
    """
    Run PhenoGraph clustering on a precomputed weighted graph file and save results by section.

    Args:
        input_npy: Path to precomputed PhenoGraph weighted graph .npz file
        output_folder: Base output folder
        resolution_idx: Resolution index (0-49)
        sections_path: Path to Sections.npy file (for splitting output)
        k_jaccard: Top-k Jaccard neighbors per cell
    """
    input_npy = Path(input_npy)
    output_folder = Path(output_folder)

    with np.load(input_npy) as weighted:
        if 'edges' not in weighted or 'weights' not in weighted:
            raise ValueError(
                f"Expected PhenoGraph weighted graph .npz with 'edges' and 'weights': {input_npy}"
            )
        weighted_edges = weighted['edges']
        weights = weighted['weights']
        if 'n_nodes' in weighted:
            n_nodes = int(weighted['n_nodes'])
        elif weighted_edges.size > 0:
            n_nodes = int(weighted_edges.max()) + 1
        else:
            n_nodes = 0

    clustering = PhenoGraphClustering(
        resolution_idx=resolution_idx,
        k_jaccard=k_jaccard,
    )
    cluster_assignments = clustering.fit_weighted_graph(weighted_edges, weights, n_nodes)
    
    # Save by section if sections provided
    res_dirname = format_resolution_dirname(clustering.resolution)
    output_dir = Path(output_folder) / res_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if sections_path is not None:
        sections = np.load(sections_path, allow_pickle=True)
        unique_sections = np.unique(sections)
        
        for section in unique_sections:
            indices = np.where(sections == section)[0]
            section_clusters = cluster_assignments[indices]
            
            output_file = output_dir / f"{section}.npy"
            np.save(output_file, section_clusters)
    else:
        # Save all as single file
        np.save(output_dir / "clusters.npy", cluster_assignments)
