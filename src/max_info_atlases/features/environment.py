"""
Environment feature creation from spatial data.

Creates neighborhood composition features by counting/weighting cell types
in k-nearest neighbor neighborhoods.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from sklearn.neighbors import NearestNeighbors


def create_environment_features(
    xy_coords: np.ndarray,
    type_vec: np.ndarray,
    k: int,
    weighted: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create environment features for a single section.
    
    For each cell, count/weight the cell types of its k nearest neighbors.
    
    Args:
        xy_coords: Spatial coordinates, shape (n_cells, 2)
        type_vec: Cell type labels, shape (n_cells,)
        k: Number of nearest neighbors
        weighted: If True, use Gaussian-weighted counts based on distance
        
    Returns:
        Tuple of (raw_counts, weighted_counts) where each is shape (n_cells, n_types)
    """
    n_cells = len(type_vec)
    n_types = int(type_vec.max()) + 1
    
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(xy_coords)
    distances, indices = nbrs.kneighbors(xy_coords)
    
    # Get neighbor types
    neighbor_types = type_vec[indices]  # shape (n_cells, k)
    
    # Raw counts
    raw_counts = np.zeros((n_cells, n_types), dtype=int)
    point_indices = np.arange(n_cells)[:, None].repeat(k, axis=1)
    np.add.at(raw_counts, (point_indices, neighbor_types), 1)
    
    # Weighted counts
    weighted_counts = np.zeros((n_cells, n_types), dtype=float)
    
    if weighted:
        # Use 6th nearest neighbor distance as local density estimate
        local_density = distances[:, min(5, k-1)][:, None] * 2
        local_density = np.maximum(local_density, 1e-6)  # Avoid division by zero
        
        # Gaussian weights
        weights = np.exp(-(distances**2) / (local_density**2))
        
        # Weighted sum for each type
        for i in range(n_cells):
            for j in range(k):
                weighted_counts[i, neighbor_types[i, j]] += weights[i, j]
    else:
        weighted_counts = raw_counts.astype(float)
    
    return raw_counts, weighted_counts


def create_environment_features_multisection(
    xy_dict: Dict[str, np.ndarray],
    type_dict: Dict[str, np.ndarray],
    k: int,
    weighted: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create environment features for multiple sections.
    
    Args:
        xy_dict: Dictionary mapping section names to XY coordinates
        type_dict: Dictionary mapping section names to type vectors
        k: Number of nearest neighbors
        weighted: If True, use Gaussian-weighted counts
        
    Returns:
        Tuple of (raw_counts_df, weighted_counts_df) DataFrames with section labels as index
    """
    # Determine max type across all sections
    max_type = max(tv.max() for tv in type_dict.values())
    n_types = int(max_type) + 1
    
    all_raw = []
    all_weighted = []
    section_labels = []
    
    sections = sorted(xy_dict.keys())
    
    for section in sections:
        if section not in type_dict:
            continue
            
        xy = xy_dict[section]
        types = type_dict[section]
        
        raw, wtd = create_environment_features(xy, types, k, weighted)
        
        # Pad to max type if needed
        if raw.shape[1] < n_types:
            raw = np.pad(raw, ((0, 0), (0, n_types - raw.shape[1])))
            wtd = np.pad(wtd, ((0, 0), (0, n_types - wtd.shape[1])))
        
        all_raw.append(raw)
        all_weighted.append(wtd)
        section_labels.extend([section] * len(types))
    
    # Concatenate and create DataFrames
    raw_matrix = np.concatenate(all_raw, axis=0)
    weighted_matrix = np.concatenate(all_weighted, axis=0)
    
    raw_df = pd.DataFrame(raw_matrix, index=section_labels)
    weighted_df = pd.DataFrame(weighted_matrix, index=section_labels)
    
    return raw_df, weighted_df


def save_environment_features(
    raw_df: pd.DataFrame,
    weighted_df: pd.DataFrame,
    output_dir: Union[str, Path],
    k: int,
) -> Tuple[Path, Path]:
    """
    Save environment features to CSV files.
    
    Args:
        raw_df: Raw count DataFrame
        weighted_df: Weighted count DataFrame
        output_dir: Output directory
        k: k value (for filename)
        
    Returns:
        Tuple of paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = output_dir / f"env_k_{k}.csv"
    weighted_path = output_dir / f"env_w_k_{k}.csv"
    
    raw_df.to_csv(raw_path)
    weighted_df.to_csv(weighted_path)
    
    return raw_path, weighted_path


def load_environment_features(
    csv_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load environment features from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (features, section_labels)
    """
    df = pd.read_csv(csv_path, index_col=0)
    
    # First column is section labels (from index)
    section_labels = df.index.values
    
    # Remaining columns are features
    features = df.values.astype(float)
    
    return features, section_labels
