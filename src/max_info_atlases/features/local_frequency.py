"""
Local frequency feature extraction from spatial data.

Creates neighborhood composition features by computing the abundance of 
each cell type in k-nearest neighbor spatial neighborhoods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from sklearn.neighbors import NearestNeighbors


def compute_local_frequency(
    xy_coords: np.ndarray,
    type_vec: np.ndarray,
    k: int,
    weighted: bool = False,
    max_type: Optional[int] = None,
) -> np.ndarray:
    """
    Compute local frequency features for a single section.
    
    For each cell, count or weight the cell types of its k nearest neighbors.
    
    Args:
        xy_coords: Spatial coordinates, shape (n_cells, 2)
        type_vec: Cell type labels (integers), shape (n_cells,)
        k: Number of nearest neighbors
        weighted: If True, use Gaussian-weighted counts based on distance
        max_type: Maximum type value (for consistent feature dimensions across sections)
        
    Returns:
        Frequency matrix of shape (n_cells, n_types+1)
        If weighted=True, returns weighted counts; otherwise raw counts
    """
    n_cells = len(type_vec)
    
    # Determine number of types
    if max_type is None:
        n_types = int(type_vec.max()) + 1
    else:
        n_types = max_type + 1
    
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(xy_coords)
    distances, indices = nbrs.kneighbors(xy_coords)
    
    # Get neighbor types
    neighbor_types = type_vec[indices]  # shape (n_cells, k)
    
    # Compute frequency matrix
    if weighted:
        # Gaussian weighting using 5th neighbor distance as scale
        # Use 4 (0-indexed) or k-1 if k < 6
        scale_idx = min(4, k - 1)
        local_density = distances[:, scale_idx][:, None] * 2
        local_density = np.maximum(local_density, 1e-6)  # Avoid division by zero
        
        # Gaussian weights
        weights = np.exp(-(distances**2) / (local_density**2))
        
        # Vectorized weighted accumulation
        freq_matrix = np.zeros((n_cells, n_types), dtype=np.float32)
        point_indices = np.arange(n_cells)[:, None].repeat(k, axis=1)
        np.add.at(freq_matrix, (point_indices.ravel(), neighbor_types.ravel()), weights.ravel())
    else:
        # Raw counts (vectorized)
        freq_matrix = np.zeros((n_cells, n_types), dtype=np.int32)
        point_indices = np.arange(n_cells)[:, None].repeat(k, axis=1)
        np.add.at(freq_matrix, (point_indices.ravel(), neighbor_types.ravel()), 1)
    
    return freq_matrix


def extract_local_frequency_features(
    adata,
    xy_dict: Dict[str, np.ndarray],
    type_column: str,
    k: int,
    weighted: bool = False,
    sections_array: Optional[np.ndarray] = None,
    section_column: Optional[str] = None,
) -> np.ndarray:
    """
    Extract local frequency features from AnnData with spatial coordinates.
    
    Args:
        adata: AnnData object with cell type annotations in .obs
        xy_dict: Dictionary mapping section names to XY coordinates
        type_column: Name of the column in adata.obs containing cell type labels
        k: Number of nearest neighbors
        weighted: If True, use Gaussian-weighted counts
        sections_array: Array of section labels (one per cell)
        section_column: Name of section column in adata.obs (alternative to sections_array)
        
    Returns:
        Feature matrix of shape (n_cells, n_types)
    """
    # Get cell type labels
    if type_column not in adata.obs.columns:
        raise ValueError(
            f"Column '{type_column}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    type_labels = adata.obs[type_column].values
    
    # Encode types as integers using pandas Categorical
    cat = pd.Categorical(type_labels)
    type_vec_all = cat.codes.astype(np.int32)
    max_type = len(cat.categories) - 1
    
    # Get section labels
    if sections_array is not None:
        sections = sections_array
    elif section_column is not None:
        if section_column not in adata.obs.columns:
            raise ValueError(
                f"Section column '{section_column}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        sections = adata.obs[section_column].values
    else:
        raise ValueError("Either sections_array or section_column must be provided")
    
    unique_sections = sorted(set(str(s) for s in np.unique(sections)))
    
    # Check that all sections have XY coordinates
    missing_sections = [s for s in unique_sections if s not in xy_dict]
    if missing_sections:
        raise ValueError(
            f"Missing XY coordinates for sections: {missing_sections}. "
            f"Available in xy_dict: {list(xy_dict.keys())}"
        )
    
    # Compute frequencies per section and concatenate
    freq_matrices = []
    for section in unique_sections:
        mask = sections == section
        section_types = type_vec_all[mask]
        section_xy = xy_dict[section]
        
        # Verify lengths match
        if len(section_xy) != len(section_types):
            raise ValueError(
                f"Section '{section}': XY coords length ({len(section_xy)}) != "
                f"type vector length ({len(section_types)})"
            )
        
        # Compute local frequency for this section
        freq_matrix = compute_local_frequency(
            section_xy, section_types, k, weighted=weighted, max_type=max_type
        )
        freq_matrices.append(freq_matrix)
    
    # Concatenate all sections
    full_freq_matrix = np.concatenate(freq_matrices, axis=0)
    
    return full_freq_matrix.astype(np.float32)


def load_xy_coordinates(xy_dir: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load XY coordinates from directory of {section}_XY.npy files.
    
    Args:
        xy_dir: Directory containing {section}_XY.npy files
        
    Returns:
        Dictionary mapping section names to XY coordinate arrays
    """
    xy_dir = Path(xy_dir)
    xy_dict = {}
    
    for xy_file in xy_dir.glob("*_XY.npy"):
        section = xy_file.stem.replace('_XY', '')
        xy_dict[section] = np.load(xy_file)
    
    if not xy_dict:
        raise ValueError(f"No *_XY.npy files found in {xy_dir}")
    
    return xy_dict


def extract_and_save_local_frequency(
    adata_path: Union[str, Path],
    xy_dir: Union[str, Path],
    output_dir: Union[str, Path],
    type_column: str,
    k: int,
    weighted: bool = False,
    sections_file: Optional[Union[str, Path]] = None,
    section_column: Optional[str] = None,
) -> Path:
    """
    Extract local frequency features and save to .npy file.
    
    Args:
        adata_path: Path to AnnData .h5ad file
        xy_dir: Directory with XY coordinate files
        output_dir: Output directory for features
        type_column: Column in adata.obs with cell type labels
        k: Number of nearest neighbors
        weighted: Whether to use Gaussian weighting
        sections_file: Path to Sections.npy file (optional)
        section_column: Column in adata.obs with section labels (optional)
        
    Returns:
        Path to saved feature file
    """
    import scanpy as sc
    
    # Load data in backed mode - we only need .obs (type/section labels), not .X
    print(f"Loading AnnData from {adata_path} (backed mode, .obs only)")
    adata = sc.read_h5ad(adata_path, backed='r')
    
    try:
        print(f"Loading XY coordinates from {xy_dir}")
        xy_dict = load_xy_coordinates(xy_dir)
        
        # Get sections
        if sections_file is not None:
            sections_array = np.load(sections_file, allow_pickle=True)
        else:
            sections_array = None
        
        # Extract features
        print(f"Computing local frequency features (k={k}, weighted={weighted}, type_column={type_column})")
        freq_features = extract_local_frequency_features(
            adata=adata,
            xy_dict=xy_dict,
            type_column=type_column,
            k=k,
            weighted=weighted,
            sections_array=sections_array,
            section_column=section_column,
        )
        
        # Save features
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build filename
        if weighted:
            filename = f"features_localfreq_w_k{k}.npy"
        else:
            filename = f"features_localfreq_k{k}.npy"
        
        output_path = output_dir / filename
        np.save(output_path, freq_features)
        
        print(f"Saved local frequency features to {output_path} (shape: {freq_features.shape})")
        
        return output_path
    finally:
        adata.file.close()
