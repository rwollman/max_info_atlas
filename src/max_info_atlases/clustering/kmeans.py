"""
K-means clustering implementation.

Primarily used for environment features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .base import ClusteringMethod, get_k_values


class KMeansClustering(ClusteringMethod):
    """
    K-means clustering with optional PCA preprocessing.
    """
    
    def __init__(
        self,
        n_clusters: int,
        n_pcs: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize K-means clustering.
        
        Args:
            n_clusters: Number of clusters
            n_pcs: Number of PCA components (None = no PCA)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.n_pcs = n_pcs
        self.random_state = random_state
        self._pca = None
        self._kmeans = None
    
    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fit K-means clustering on features.
        
        Args:
            features: Feature matrix, shape (n_samples, n_features)
            
        Returns:
            Array of cluster assignments
        """
        # Apply PCA if requested
        if self.n_pcs is not None:
            self._pca = PCA(n_components=self.n_pcs, random_state=self.random_state)
            features = self._pca.fit_transform(features)
        
        # Fit K-means
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init='auto',
            random_state=self.random_state,
        )
        
        return self._kmeans.fit_predict(features)
    
    def get_params(self) -> Dict[str, Any]:
        """Get clustering parameters."""
        return {
            'method': 'kmeans',
            'n_clusters': self.n_clusters,
            'n_pcs': self.n_pcs,
            'random_state': self.random_state,
        }


def run_kmeans_on_csv(
    input_csv: Union[str, Path],
    output_folder: Union[str, Path],
    data_type: str = 'raw',
    k_values: Optional[List[int]] = None,
) -> None:
    """
    Run K-means clustering on a CSV file for multiple k values.
    
    This is a convenience function that mirrors the original script interface.
    
    Args:
        input_csv: Path to input CSV file (environment features)
        output_folder: Base output folder
        data_type: Feature type ('raw', 'pca15', 'pca50')
        k_values: List of k values to try (None = use default log-spaced values)
    """
    # Load data
    df = pd.read_csv(input_csv, index_col=None)
    sections = df.iloc[:, 0]
    features = df.iloc[:, 1:].select_dtypes(include=[np.number]).values
    
    # Determine PCA components
    n_pcs = None
    if data_type == 'pca15':
        n_pcs = 15
    elif data_type == 'pca50':
        n_pcs = 50
    
    # Default k values
    if k_values is None:
        k_values = get_k_values().tolist()
    
    # Run for each k
    for k in k_values:
        clustering = KMeansClustering(n_clusters=k, n_pcs=n_pcs)
        cluster_assignments = clustering.fit(features)
        
        # Save by section
        output_dir = Path(output_folder) / f"k_{k}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        unique_sections = pd.unique(sections)
        for section in unique_sections:
            mask = sections == section
            section_clusters = cluster_assignments[mask]
            np.save(output_dir / f"{section}.npy", section_clusters)
