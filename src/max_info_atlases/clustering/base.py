"""
Abstract base class for clustering methods.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


class ClusteringMethod(ABC):
    """
    Abstract base class for all clustering methods.
    
    Subclasses must implement the fit() method.
    """
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the clustering model and return cluster assignments.
        
        Args:
            data: Input data (features or edge list depending on method)
            
        Returns:
            Array of cluster assignments, shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of this clustering instance.
        
        Returns:
            Dictionary of parameter names to values
        """
        pass
    
    def save_by_section(
        self,
        cluster_assignments: np.ndarray,
        sections: np.ndarray,
        output_dir: Union[str, Path],
    ) -> List[Path]:
        """
        Split cluster assignments by section and save to separate files.
        
        Args:
            cluster_assignments: Cluster assignments for all samples
            sections: Section labels for all samples
            output_dir: Directory to save output files
            
        Returns:
            List of paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        unique_sections = np.unique(sections)
        
        for section in unique_sections:
            mask = sections == section
            section_clusters = cluster_assignments[mask]
            
            output_file = output_dir / f"{section}.npy"
            np.save(output_file, section_clusters)
            saved_files.append(output_file)
        
        return saved_files


def get_resolution_values(
    n_resolutions: int = 50,
    log_min: float = -1.0,
    log_max: float = 2.5,
) -> np.ndarray:
    """
    Generate log-spaced resolution values for Leiden clustering.
    
    Args:
        n_resolutions: Number of resolution values
        log_min: Log10 of minimum resolution
        log_max: Log10 of maximum resolution
        
    Returns:
        Array of resolution values
    """
    return np.logspace(log_min, log_max, n_resolutions)


def format_resolution_dirname(resolution: float) -> str:
    """
    Format resolution value as a directory name.
    
    Converts resolution float to string with 3 decimal places,
    replacing '.' with 'p' for filesystem compatibility.
    
    Args:
        resolution: Resolution value
        
    Returns:
        Formatted string like 'res_0p100' for resolution=0.1
    
    Examples:
        >>> format_resolution_dirname(0.1)
        'res_0p100'
        >>> format_resolution_dirname(1.0)
        'res_1p000'
        >>> format_resolution_dirname(316.227766)
        'res_316p228'
    """
    return f"res_{resolution:.3f}".replace(".", "p")


def get_k_values(
    n_values: int = 50,
    k_min: int = 10,
    k_max: int = 1000,
) -> np.ndarray:
    """
    Generate log-spaced k values for K-means clustering.
    
    Args:
        n_values: Number of k values
        k_min: Minimum k value
        k_max: Maximum k value
        
    Returns:
        Array of unique integer k values
    """
    k_values = np.logspace(np.log10(k_min), np.log10(k_max), n_values)
    return np.unique(np.round(k_values).astype(int))
