"""
PhenoGraph clustering implementation.

PhenoGraph is a variant of Leiden clustering that uses a Jaccard-weighted graph
based on shared neighbors instead of the raw kNN graph. Uses an efficient
cell-centric algorithm with set operations, batching, and checkpointing.
"""

import os
import time
import pickle
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
    temp_dir: Optional[Union[str, Path]] = None,
    cell_timeout: int = 600,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Jaccard-weighted graph from kNN edge list.

    Uses a memory-efficient, cell-centric algorithm: for each cell i, finds
    potential neighbors (cells that share at least one neighbor with i),
    computes Jaccard similarity via set operations, and keeps top-k_jaccard
    per cell. Supports checkpointing for resumability.

    Args:
        edge_list: kNN edge list of shape (n_edges, 2) with [source, target]
        k_jaccard: Number of top Jaccard neighbors to keep per cell
        batch_size: Cells per batch for memory management
        temp_dir: Directory for checkpoint files (enables resumability)
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

    # Temp dir for checkpointing
    acquired_lock = False
    if temp_dir:
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        
        complete_marker = temp_path / "complete.marker"
        lock_file = temp_path / "computing.lock"
        
        # Check if computation is already complete
        if complete_marker.exists():
            print(f"  Found complete Jaccard computation: {temp_path}")
            batches = sorted(temp_path.glob("jaccard_batch_*.pkl"))
            reusing = True
        else:
            # Try to acquire lock (atomic file creation)
            try:
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.write(fd, f"pid={os.getpid()}\n".encode())
                os.close(fd)
                print(f"  Acquired lock - will compute Jaccard")
                acquired_lock = True
                reusing = False
            except FileExistsError:
                # Another job is computing - wait for it
                print(f"  Another job is computing Jaccard - waiting...")
                for wait_iter in range(300):  # Wait up to 5 hours (60s * 300)
                    time.sleep(60)
                    if complete_marker.exists():
                        print(f"  Computation complete - will load checkpoints")
                        batches = sorted(temp_path.glob("jaccard_batch_*.pkl"))
                        reusing = True
                        break
                else:
                    print(f"  WARNING: Timeout waiting for Jaccard computation")
                    print(f"  Will compute anyway (possibly redundant)")
                    reusing = False
    else:
        temp_path = None
        reusing = False
        complete_marker = None
        lock_file = None
        acquired_lock = False

    top_k_neighbors = defaultdict(list)
    start_time = time.time()

    if reusing and temp_path:
        # Load existing batches
        batch_files = sorted(temp_path.glob("jaccard_batch_*.pkl"))
        for bf in batch_files:
            with open(bf, "rb") as f:
                batch_dict = pickle.load(f)
            for src, neighbors in batch_dict.items():
                current = top_k_neighbors[src]
                for item in neighbors:
                    if len(current) < k_jaccard:
                        heapq.heappush(current, item)
                    elif item[0] < current[0][0]:
                        heapq.heappushpop(current, item)
    else:
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

            if temp_path:
                pkl = temp_path / f"jaccard_batch_{batch_idx + 1}.pkl"
                with open(pkl, "wb") as f:
                    pickle.dump(dict(batch_top_k), f)

        if problem_cells and temp_path:
            prob_file = temp_path / "problem_cells.txt"
            with open(prob_file, "w") as f:
                for c, info in problem_cells:
                    f.write(f"{c},{info}\n")
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

    # Mark computation as complete and release lock (only if we acquired it)
    if temp_path and acquired_lock and complete_marker:
        complete_marker.touch()
        if lock_file and lock_file.exists():
            lock_file.unlink()
        print(f"  Checkpoints saved and marked complete")

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
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize PhenoGraph clustering.

        Args:
            resolution: Resolution parameter (higher = more clusters)
            resolution_idx: Index into standard resolution array (0-49)
            n_resolutions: Number of resolution values in standard array
            objective: Objective function ('modularity' or 'CPM')
            k_jaccard: Top-k Jaccard neighbors per cell
            temp_dir: Directory for Jaccard checkpointing (enables resumability)
        """
        self.n_resolutions = n_resolutions
        self.objective = objective
        self.k_jaccard = k_jaccard
        self.temp_dir = Path(temp_dir) if temp_dir else None

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
            temp_dir=str(self.temp_dir) if self.temp_dir else None,
        )

        n_nodes = int(edge_list.max()) + 1

        graph = igraph.Graph(n=n_nodes, edges=jaccard_edges.tolist())
        graph.es['weight'] = jaccard_weights.tolist()

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
    Run PhenoGraph clustering on a kNN edge list file and save results by section.

    Args:
        input_npy: Path to kNN edge list .npy file
        output_folder: Base output folder
        resolution_idx: Resolution index (0-49)
        sections_path: Path to Sections.npy file (for splitting output)
        k_jaccard: Top-k Jaccard neighbors per cell
    """
    input_npy = Path(input_npy)
    output_folder = Path(output_folder)

    # Temp dir for Jaccard checkpointing (shared across resolutions for same input)
    temp_dir = output_folder.parent / ".temp_jaccard" / input_npy.stem

    edge_list = np.load(input_npy)
    clustering = PhenoGraphClustering(
        resolution_idx=resolution_idx,
        k_jaccard=k_jaccard,
        temp_dir=temp_dir,
    )
    cluster_assignments = clustering.fit(edge_list)
    
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
