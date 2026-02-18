"""
Leiden clustering implementation.

Works for both cell type features (via FEL graph) and environment features.
"""

import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .base import ClusteringMethod, get_resolution_values, format_resolution_dirname


class LeidenClustering(ClusteringMethod):
    """
    Leiden community detection clustering.
    
    Uses the igraph implementation of Leiden algorithm with modularity objective.
    """
    
    def __init__(
        self,
        resolution: Optional[float] = None,
        resolution_idx: Optional[int] = None,
        n_resolutions: int = 50,
        objective: str = 'modularity',
    ):
        """
        Initialize Leiden clustering.
        
        Args:
            resolution: Resolution parameter (higher = more clusters)
            resolution_idx: Index into standard resolution array (0-49)
            n_resolutions: Number of resolution values in standard array
            objective: Objective function ('modularity' or 'CPM')
        """
        self.n_resolutions = n_resolutions
        self.objective = objective
        
        # Determine resolution
        if resolution is not None:
            self.resolution = resolution
            self.resolution_idx = None
        elif resolution_idx is not None:
            resolutions = get_resolution_values(n_resolutions)
            self.resolution = resolutions[resolution_idx]
            self.resolution_idx = resolution_idx
        else:
            # Default to middle resolution
            resolutions = get_resolution_values(n_resolutions)
            self.resolution_idx = n_resolutions // 2
            self.resolution = resolutions[self.resolution_idx]
    
    def fit(self, edge_list: np.ndarray) -> np.ndarray:
        """
        Fit Leiden clustering on an edge list (graph).
        
        Args:
            edge_list: Edge list of shape (n_edges, 2) with [source, target] indices
            
        Returns:
            Array of cluster assignments
        """
        import igraph
        
        # Determine number of nodes
        n_nodes = int(edge_list.max()) + 1
        
        # Create graph
        graph = igraph.Graph(n=n_nodes, edges=edge_list.tolist())
        
        # Run Leiden clustering
        partition = graph.community_leiden(
            resolution_parameter=self.resolution,
            objective_function=self.objective,
        )
        
        return np.array(partition.membership, dtype=np.int32)
    
    def get_params(self) -> Dict[str, Any]:
        """Get clustering parameters."""
        return {
            'method': 'leiden',
            'resolution': self.resolution,
            'resolution_idx': self.resolution_idx,
            'objective': self.objective,
        }


def run_leiden_on_file(
    input_npy: Union[str, Path],
    output_folder: Union[str, Path],
    resolution_idx: int,
    sections_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Run Leiden clustering on an FEL.npy file and save results by section.
    
    This is a convenience function that mirrors the original script interface.
    
    Args:
        input_npy: Path to FEL.npy edge list file
        output_folder: Base output folder
        resolution_idx: Resolution index (0-49)
        sections_path: Path to Sections.npy file (for splitting output)
    """
    # Load edge list
    edge_list = np.load(input_npy)
    
    # Create clustering instance
    clustering = LeidenClustering(resolution_idx=resolution_idx)
    
    # Fit
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
