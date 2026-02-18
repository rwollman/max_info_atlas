"""
LDA (Latent Dirichlet Allocation) clustering implementation.

Primarily used for environment features, treating cells as "documents"
and cell type counts as "words".
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation

from .base import ClusteringMethod


class LDAClustering(ClusteringMethod):
    """
    LDA-based clustering using topic assignments.
    
    Each cell is assigned to its most probable topic.
    """
    
    def __init__(
        self,
        n_topics: int,
        batch_size: int = 2048,
        random_state: int = 42,
    ):
        """
        Initialize LDA clustering.
        
        Args:
            n_topics: Number of topics (clusters)
            batch_size: Batch size for online learning
            random_state: Random seed for reproducibility
        """
        self.n_topics = n_topics
        self.batch_size = batch_size
        self.random_state = random_state
        self._lda = None
        self._topic_probs = None
    
    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        Fit LDA and return cluster assignments.
        
        Each sample is assigned to its most probable topic.
        
        Args:
            features: Feature matrix (counts), shape (n_samples, n_features)
            
        Returns:
            Array of cluster (topic) assignments
        """
        # Ensure non-negative (LDA requires counts)
        features = np.maximum(features, 0)
        
        # Fit LDA
        self._lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            batch_size=self.batch_size,
        )
        
        # Get topic probabilities
        self._topic_probs = self._lda.fit_transform(features)
        
        # Assign each sample to most probable topic
        return np.argmax(self._topic_probs, axis=1).astype(np.int32)
    
    def get_topic_probabilities(self) -> Optional[np.ndarray]:
        """
        Get the topic probability matrix from the last fit.
        
        Returns:
            Topic probabilities, shape (n_samples, n_topics), or None if not fit
        """
        return self._topic_probs
    
    def get_params(self) -> Dict[str, Any]:
        """Get clustering parameters."""
        return {
            'method': 'lda',
            'n_topics': self.n_topics,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
        }


def run_lda_on_csv(
    input_csv: Union[str, Path],
    output_folder: Union[str, Path],
    n_topics: int,
) -> None:
    """
    Run LDA clustering on a CSV file.
    
    This is a convenience function that mirrors the original script interface.
    
    Args:
        input_csv: Path to input CSV file (environment features)
        output_folder: Base output folder
        n_topics: Number of topics
    """
    # Load data
    df = pd.read_csv(input_csv, index_col=None)
    sections = df.iloc[:, 0]
    features = df.iloc[:, 1:].select_dtypes(include=[np.number]).values
    
    # Run LDA
    clustering = LDAClustering(n_topics=n_topics)
    cluster_assignments = clustering.fit(features)
    
    # Save by section
    output_dir = Path(output_folder) / f"k_{n_topics}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unique_sections = pd.unique(sections)
    for section in unique_sections:
        mask = sections == section
        section_clusters = cluster_assignments[mask]
        np.save(output_dir / f"{section}.npy", section_clusters)
