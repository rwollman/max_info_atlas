"""
Centralized path parsing utilities.

This module provides a single source of truth for parsing metadata from folder paths
and building folder paths from metadata. It replaces the 200+ line extract_metadata_from_path()
function scattered across multiple scripts.

Path conventions (v3 - current):
- Leiden cell type: LeidenRawCorrelation/res_0p100/section.npy (resolution value with 'p' for decimal)
- Leiden env: LeidenRawCorrelation_envenv_w_k_20/res_1p000/section.npy
- Kmeans env: Kmeans_env_k_50/k_100/section.npy
- LDA env: LDA/k_50/k_100/section.npy
- PhenoGraph: PhenoGraphSCVIEuclidean/res_316p228/section.score
- Preexisting annotations: Preexisting_class/section.score (also parses legacy Allen_ prefix)

Legacy formats (still supported for parsing):
- v2: LeidenRawCorrelation/res_2/section.npy (resolution index)
- v1: LeidenRawCorrelation_res_2/section.npy (resolution index in folder name)
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class ClusteringMetadata:
    """Parsed metadata from a clustering result file path."""
    
    # Core identifiers
    method: str                          # 'leiden', 'kmeans', 'lda', 'phenograph', 'preexisting'
    section: str                         # e.g., 'C57BL6J-638850.40'
    
    # Feature type
    feature_type: str = 'celltype'       # 'celltype' or 'env'
    
    # Data transformation
    data_type: Optional[str] = None      # 'raw', 'pca15', 'pca50', 'scvi', 'log1p'
    distance: Optional[str] = None       # 'euclidean', 'cosine', 'correlation'
    
    # Environment-specific
    env_k: Optional[int] = None          # k for neighborhood environment
    weighted: bool = False               # weighted environment
    
    # Clustering parameters
    resolution_idx: Optional[int] = None # For Leiden/PhenoGraph (0-49 typically) - legacy
    resolution: Optional[float] = None   # For Leiden/PhenoGraph - actual resolution value
    clustering_k: Optional[int] = None   # For K-means/LDA
    
    # Preexisting annotation-specific
    annotation_level: Optional[str] = None  # For preexisting annotations: 'class', 'cluster', 'subclass', etc.
    
    # Raw algorithm string (for debugging/compatibility)
    algorithm_string: Optional[str] = None
    
    # File path
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method,
            'section': self.section,
            'feature_type': self.feature_type,
            'data_type': self.data_type,
            'distance': self.distance,
            'env_k': self.env_k,
            'weighted': self.weighted,
            'resolution_idx': self.resolution_idx,
            'resolution': self.resolution,
            'clustering_k': self.clustering_k,
            'annotation_level': self.annotation_level,
            'algorithm_string': self.algorithm_string,
            'file_path': self.file_path,
        }


def _parse_leiden_algorithm(algorithm: str) -> Dict[str, Any]:
    """
    Parse a Leiden algorithm string.
    
    Examples:
        - LeidenRawCorrelation -> data_type=raw, distance=correlation, env=False
        - LeidenPCA50Euclidean -> data_type=pca50, distance=euclidean, env=False
        - LeidenRawCorrelation_envenv_w_k_20 -> data_type=raw, distance=correlation, env=True, weighted=True, env_k=20
    """
    result = {
        'data_type': None,
        'distance': None,
        'feature_type': 'celltype',
        'weighted': False,
        'env_k': None,
    }
    
    # Check for environment features
    if '_envenv' in algorithm:
        result['feature_type'] = 'env'
        
        # Check for weighted
        if '_w_' in algorithm or '_w_k_' in algorithm:
            result['weighted'] = True
        
        # Extract env_k from algorithm name
        k_match = re.search(r'_k_(\d+)', algorithm)
        if k_match:
            result['env_k'] = int(k_match.group(1))
    
    # Parse the base part (before _envenv if present)
    base = algorithm.split('_envenv')[0] if '_envenv' in algorithm else algorithm
    
    # Remove 'Leiden' prefix
    if base.startswith('Leiden'):
        info = base[6:]  # Remove 'Leiden'
        
        # Parse data type and distance (they're concatenated)
        # Order: data_type + distance (e.g., RawCorrelation, PCA50Euclidean)
        
        # Known data types
        data_types = ['Raw', 'PCA15', 'PCA50', 'SCVI', 'Log1p']
        # Known distances
        distances = ['Correlation', 'Euclidean', 'Cosine']
        
        for dt in data_types:
            if info.startswith(dt):
                result['data_type'] = dt.lower()
                remaining = info[len(dt):]
                for dist in distances:
                    if remaining.startswith(dist):
                        result['distance'] = dist.lower()
                        break
                break
    
    return result


def _parse_kmeans_algorithm(algorithm: str) -> Dict[str, Any]:
    """
    Parse a K-means algorithm string.
    
    Examples:
        - Kmeans_env_k_50 -> env_k=50
        - Kmeans_pca15_env_k_50 -> data_type=pca15, env_k=50
    """
    result = {
        'data_type': 'raw',
        'feature_type': 'env',  # K-means is only for env features
        'env_k': None,
    }
    
    algorithm_lower = algorithm.lower()
    
    # Extract data type
    if 'pca15' in algorithm_lower:
        result['data_type'] = 'pca15'
    elif 'pca50' in algorithm_lower:
        result['data_type'] = 'pca50'
    
    # Extract env_k
    k_match = re.search(r'env_k_(\d+)', algorithm, re.IGNORECASE)
    if k_match:
        result['env_k'] = int(k_match.group(1))
    else:
        # Try generic k_ pattern
        k_match = re.search(r'_k_(\d+)', algorithm)
        if k_match:
            result['env_k'] = int(k_match.group(1))
    
    return result


def _parse_lda_algorithm(algorithm: str) -> Dict[str, Any]:
    """
    Parse an LDA algorithm string.
    
    Examples:
        - LDA -> basic LDA
        - LDA_env_k_50 -> env_k=50
    """
    result = {
        'data_type': 'raw',
        'feature_type': 'env',  # LDA is only for env features
        'env_k': None,
    }
    
    # Extract env_k if present
    k_match = re.search(r'_k_(\d+)', algorithm)
    if k_match:
        result['env_k'] = int(k_match.group(1))
    
    return result


def _parse_phenograph_algorithm(algorithm: str) -> Dict[str, Any]:
    """
    Parse a PhenoGraph algorithm string.
    
    Examples:
        - PhenoGraphSCVIEuclidean_res_9 -> data_type=scvi, distance=euclidean, resolution=9 (legacy)
        - PhenoGraphRawCorrelation_res_0 -> data_type=raw, distance=correlation, resolution=0 (legacy)
        - PhenoGraphLog1pCosine -> data_type=log1p, distance=cosine (resolution in path)
        - PhenoGraphPCA50Euclidean -> data_type=pca50, distance=euclidean (resolution in path)
    """
    result = {
        'data_type': None,
        'distance': None,
        'feature_type': 'celltype',
        'resolution_idx': None,
    }
    
    # Remove 'PhenoGraph' prefix
    if algorithm.startswith('PhenoGraph'):
        info = algorithm[10:]  # Remove 'PhenoGraph'
        
        # Extract resolution from _res_X suffix (if present in folder name)
        res_match = re.search(r'_res_(\d+)', info)
        if res_match:
            result['resolution_idx'] = int(res_match.group(1))
            # Remove the _res_X part to parse the rest
            info = info[:res_match.start()]
        
        # Parse data type and distance (they're concatenated like Leiden)
        # Order: data_type + distance (e.g., RawCorrelation, SCVIEuclidean)
        
        # Known data types (case-sensitive as they appear in folder names)
        data_types = ['Raw', 'SCVI', 'Log1p', 'PCA50', 'PCA15']
        # Known distances
        distances = ['Correlation', 'Euclidean', 'Cosine']
        
        for dt in data_types:
            if info.startswith(dt):
                result['data_type'] = dt.lower()
                remaining = info[len(dt):]
                for dist in distances:
                    if remaining.startswith(dist):
                        result['distance'] = dist.lower()
                        break
                break
    
    return result


def _parse_preexisting_algorithm(algorithm: str) -> Dict[str, Any]:
    """
    Parse a preexisting annotation algorithm string.
    
    Handles both 'Preexisting_' prefix (new) and 'Allen_' prefix (legacy).
    
    Examples:
        - Preexisting_class -> annotation_level=class
        - Preexisting_cluster -> annotation_level=cluster
        - Allen_class -> annotation_level=class  (legacy)
    """
    result = {
        'annotation_level': None,
        'feature_type': 'celltype',
    }
    
    if algorithm.startswith('Preexisting_'):
        result['annotation_level'] = algorithm[len('Preexisting_'):]
    elif algorithm.startswith('Allen_'):
        result['annotation_level'] = algorithm[len('Allen_'):]
    
    return result


def parse_path(file_path: str, base_dir: str) -> ClusteringMetadata:
    """
    Parse metadata from a clustering result file path.
    
    Args:
        file_path: Full path to the result file (.npy or .npz)
        base_dir: Base directory of results (for computing relative path)
        
    Returns:
        ClusteringMetadata with parsed information
        
    Raises:
        ValueError: If path cannot be parsed
        
    Examples:
        >>> parse_path('/results/LeidenRawCorrelation_envenv_w_k_20/res_1p000/C57BL6J-638850.40.npy', '/results')
        ClusteringMetadata(method='leiden', section='C57BL6J-638850.40', feature_type='env', resolution=1.0, ...)
    """
    # Get relative path and split into parts
    rel_path = os.path.relpath(file_path, base_dir)
    path_parts = rel_path.split(os.sep)
    
    # Extract section from filename
    filename = os.path.basename(file_path)
    section = filename.replace('.npz', '').replace('.npy', '').replace('.score', '')
    
    # Need at least algorithm/param/section.ext
    if len(path_parts) < 2:
        raise ValueError(f"Path too short to parse: {rel_path}")
    
    algorithm = path_parts[0]
    
    # Initialize metadata
    metadata = ClusteringMetadata(
        method='unknown',
        section=section,
        algorithm_string=algorithm,
        file_path=file_path,
    )
    
    # Parse based on algorithm type
    if algorithm.startswith('Leiden'):
        metadata.method = 'leiden'
        parsed = _parse_leiden_algorithm(algorithm)
        metadata.data_type = parsed['data_type']
        metadata.distance = parsed['distance']
        metadata.feature_type = parsed['feature_type']
        metadata.weighted = parsed['weighted']
        metadata.env_k = parsed['env_k']
        
        # Extract resolution from path
        # v3 format: LeidenRawCorrelation/res_0p100/section.npy (new - using actual value)
        # v2 format: LeidenRawCorrelation/res_2/section.npy (old - using index)
        # v1 format: LeidenRawCorrelation_res_2/section.npy (legacy)
        resolution_found = False
        
        # Check subdirectories first (v2/v3 format)
        for part in path_parts[1:]:
            if part.startswith('res_'):
                res_str = part.replace('res_', '')
                # Try new format first (res_0p100)
                if 'p' in res_str:
                    try:
                        metadata.resolution = float(res_str.replace('p', '.'))
                        resolution_found = True
                    except ValueError:
                        pass
                else:
                    # Old format (res_2)
                    try:
                        metadata.resolution_idx = int(res_str)
                        resolution_found = True
                    except ValueError:
                        pass
                break
        
        # If not found in subdirectories, check algorithm name (v1 format)
        if not resolution_found and '_res_' in algorithm:
            res_match = re.search(r'_res_(\d+)', algorithm)
            if res_match:
                metadata.resolution_idx = int(res_match.group(1))
    
    elif algorithm.startswith('Kmeans') or algorithm.startswith('kmeans'):
        metadata.method = 'kmeans'
        parsed = _parse_kmeans_algorithm(algorithm)
        metadata.data_type = parsed['data_type']
        metadata.feature_type = parsed['feature_type']
        metadata.env_k = parsed['env_k']
        
        # Extract clustering_k from path (e.g., k_100)
        for part in path_parts[1:]:
            if part.startswith('k_'):
                try:
                    metadata.clustering_k = int(part.replace('k_', ''))
                except ValueError:
                    pass
                break
    
    elif algorithm.startswith('LDA') or algorithm == 'LDA':
        metadata.method = 'lda'
        parsed = _parse_lda_algorithm(algorithm)
        metadata.data_type = parsed['data_type']
        metadata.feature_type = parsed['feature_type']
        
        # For LDA, path structure is LDA/k_50/k_100/section.npz
        # k_50 is env_k, k_100 is clustering_k (n_topics)
        k_values = []
        for part in path_parts[1:]:
            if part.startswith('k_'):
                try:
                    k_values.append(int(part.replace('k_', '')))
                except ValueError:
                    pass
        
        if len(k_values) >= 1:
            metadata.env_k = k_values[0]
        if len(k_values) >= 2:
            metadata.clustering_k = k_values[1]
    
    elif algorithm.startswith('PhenoGraph'):
        metadata.method = 'phenograph'
        parsed = _parse_phenograph_algorithm(algorithm)
        metadata.data_type = parsed['data_type']
        metadata.distance = parsed['distance']
        metadata.feature_type = parsed['feature_type']
        
        # Resolution can be in folder name or subdirectory
        if parsed['resolution_idx'] is not None:
            metadata.resolution_idx = parsed['resolution_idx']
        else:
            # Check subdirectories for res_X or res_Xp000 (similar to Leiden)
            for part in path_parts[1:]:
                if part.startswith('res_'):
                    res_str = part.replace('res_', '')
                    # Try new format first (res_0p100)
                    if 'p' in res_str:
                        try:
                            metadata.resolution = float(res_str.replace('p', '.'))
                        except ValueError:
                            pass
                    else:
                        # Old format (res_2)
                        try:
                            metadata.resolution_idx = int(res_str)
                        except ValueError:
                            pass
                    break
    
    elif algorithm.startswith('Preexisting_') or algorithm.startswith('Allen_'):
        metadata.method = 'preexisting'
        parsed = _parse_preexisting_algorithm(algorithm)
        metadata.annotation_level = parsed['annotation_level']
        metadata.feature_type = parsed['feature_type']
    
    else:
        # Unknown algorithm - try basic parsing
        metadata.method = algorithm.lower()
        
        # Try to extract k values from path
        for part in path_parts[1:]:
            if part.startswith('env_k_'):
                try:
                    metadata.env_k = int(part.replace('env_k_', ''))
                except ValueError:
                    pass
            elif part.startswith('k_'):
                try:
                    metadata.clustering_k = int(part.replace('k_', ''))
                except ValueError:
                    pass
            elif part.startswith('res_'):
                try:
                    metadata.resolution_idx = int(part.replace('res_', ''))
                except ValueError:
                    pass
    
    return metadata


def build_path(metadata: ClusteringMetadata, base_dir: str, extension: str = '.npy') -> str:
    """
    Build a file path from metadata (inverse of parse_path).
    
    Args:
        metadata: ClusteringMetadata with parameters
        base_dir: Base directory for results
        extension: File extension (.npy or .npz)
        
    Returns:
        Full path to the result file
        
    Examples:
        >>> m = ClusteringMetadata(method='leiden', section='C57BL6J-638850.40', 
        ...                        feature_type='env', data_type='raw', distance='correlation',
        ...                        env_k=20, weighted=True, resolution=1.0)
        >>> build_path(m, '/results')
        '/results/LeidenRawCorrelation_envenv_w_k_20/res_1p000/C57BL6J-638850.40.npy'
    """
    parts = []
    
    if metadata.method == 'leiden':
        # Build algorithm string
        algo = 'Leiden'
        if metadata.data_type:
            algo += metadata.data_type.capitalize()
        if metadata.distance:
            algo += metadata.distance.capitalize()
        
        if metadata.feature_type == 'env':
            algo += '_envenv'
            if metadata.weighted:
                algo += '_w'
            if metadata.env_k is not None:
                algo += f'_k_{metadata.env_k}'
        
        parts.append(algo)
        
        # Use resolution value if available (new format), otherwise fall back to index
        if metadata.resolution is not None:
            from .clustering.base import format_resolution_dirname
            parts.append(format_resolution_dirname(metadata.resolution))
        elif metadata.resolution_idx is not None:
            parts.append(f'res_{metadata.resolution_idx}')
    
    elif metadata.method == 'kmeans':
        # Build algorithm string
        algo = 'Kmeans'
        if metadata.data_type and metadata.data_type != 'raw':
            algo += f'_{metadata.data_type}'
        if metadata.env_k is not None:
            algo += f'_env_k_{metadata.env_k}'
        
        parts.append(algo)
        
        if metadata.clustering_k is not None:
            parts.append(f'k_{metadata.clustering_k}')
    
    elif metadata.method == 'lda':
        parts.append('LDA')
        
        if metadata.env_k is not None:
            parts.append(f'k_{metadata.env_k}')
        if metadata.clustering_k is not None:
            parts.append(f'k_{metadata.clustering_k}')
    
    elif metadata.method == 'phenograph':
        # Build algorithm string (similar to Leiden)
        algo = 'PhenoGraph'
        if metadata.data_type:
            dt = metadata.data_type
            if dt == 'scvi':
                algo += 'SCVI'
            elif dt == 'log1p':
                algo += 'Log1p'
            elif dt.startswith('pca'):
                algo += f"PCA{dt[3:]}"
            else:
                algo += dt.capitalize()
        if metadata.distance:
            algo += metadata.distance.capitalize()
        
        # Add resolution to folder name (using new format with resolution value)
        if metadata.resolution is not None:
            from .clustering.base import format_resolution_dirname
            res_dirname = format_resolution_dirname(metadata.resolution)
            algo += f'_{res_dirname}'
        elif metadata.resolution_idx is not None:
            algo += f'_res_{metadata.resolution_idx}'
        
        parts.append(algo)
    
    elif metadata.method == 'preexisting':
        # Preexisting annotations
        if metadata.annotation_level:
            parts.append(f'Preexisting_{metadata.annotation_level}')
        else:
            parts.append('Preexisting')
    
    else:
        # Generic
        parts.append(metadata.method)
    
    # Add filename
    parts.append(f'{metadata.section}{extension}')
    
    return os.path.join(base_dir, *parts)


def extract_metadata_from_path(file_path: str, base_dir: str) -> Dict[str, Any]:
    """
    Extract metadata from file path (compatibility wrapper).
    
    This function provides backward compatibility with the old interface.
    New code should use parse_path() directly.
    
    Args:
        file_path: Path to the result file
        base_dir: Base directory of results
        
    Returns:
        Dictionary with metadata (old format)
    """
    try:
        metadata = parse_path(file_path, base_dir)
    except ValueError:
        # Return minimal metadata on parse failure
        filename = os.path.basename(file_path)
        section = filename.replace('.npz', '').replace('.npy', '').replace('.score', '')
        return {
            'algorithm': 'unknown',
            'method': None,
            'data_type': None,
            'distance': None,
            'env_type': None,
            'weighted': None,
            'algorithm_k': None,
            'env_k': None,
            'clustering_k': None,
            'section_name': section,
            'file_path': file_path,
        }
    
    # Build clean algorithm identifier
    # For PhenoGraph and Preexisting, use the raw folder name but keep method standardized
    algorithm = metadata.algorithm_string or metadata.method
    
    # Build method string - for preexisting, include annotation level
    if metadata.method == 'preexisting' and metadata.annotation_level:
        method = f"Preexisting_{metadata.annotation_level}"
    elif metadata.method:
        method = metadata.method.capitalize()
    else:
        method = None
    
    # Convert to old format
    return {
        'algorithm': algorithm,
        'method': method,
        'data_type': metadata.data_type.upper() if metadata.data_type else None,
        'distance': metadata.distance.capitalize() if metadata.distance else None,
        'env_type': 'envenv' if metadata.feature_type == 'env' else None,
        'weighted': metadata.weighted,
        'algorithm_k': str(metadata.env_k) if metadata.env_k else None,
        'env_k': str(metadata.env_k) if metadata.env_k else None,
        'clustering_k': str(metadata.clustering_k) if metadata.clustering_k else (
            str(metadata.resolution_idx) if metadata.resolution_idx is not None else None
        ),
        'resolution': metadata.resolution,
        'section_name': metadata.section,
        'file_path': file_path,
    }
