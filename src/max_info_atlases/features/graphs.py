"""
k-NN graph construction for clustering.

Builds k-nearest neighbor graphs from feature matrices using various distance metrics.
"""

import numpy as np
from typing import Optional, Tuple, Union
from pathlib import Path
from sklearn.decomposition import PCA


def build_knn_graph(
    features: np.ndarray,
    k: int = 15,
    metric: str = 'cosine',
) -> np.ndarray:
    """
    Build k-nearest neighbor graph from features.
    
    Uses pynndescent for fast approximate nearest neighbor search.
    
    Args:
        features: Feature matrix, shape (n_samples, n_features)
        k: Number of nearest neighbors
        metric: Distance metric ('cosine', 'correlation', 'euclidean')
        
    Returns:
        Edge list of shape (n_edges, 2) with [source, target] indices
    """
    import pynndescent
    
    n_samples = features.shape[0]
    
    # Handle correlation metric by centering data (in-place to save memory)
    if metric == 'correlation':
        # Center in-place: subtract row means
        row_means = np.mean(features, axis=1, keepdims=True)
        features -= row_means  # in-place subtraction
        pynn_metric = 'cosine'
    else:
        pynn_metric = metric
    
    # Build index (n_jobs=1 for minimum memory usage with large datasets)
    index = pynndescent.NNDescent(
        features,
        metric=pynn_metric,
        n_neighbors=k,
        random_state=42,
        n_jobs=1,
    )
    
    # Get neighbor indices
    indices, distances = index.neighbor_graph
    
    # Create edge list
    edge_list = []
    for i in range(n_samples):
        for j_idx in range(k):
            j = indices[i, j_idx]
            if i != j:  # Exclude self-loops
                edge_list.append([i, j])
    
    return np.array(edge_list, dtype=np.int32)


def build_knn_graph_from_file(
    file_path: Union[str, Path],
    k: int = 15,
    metric: str = 'cosine',
) -> np.ndarray:
    """
    Build k-NN graph from numpy feature file.
    
    Args:
        file_path: Path to .npy file with features (already PCA-transformed if needed)
        k: Number of nearest neighbors
        metric: Distance metric
        
    Returns:
        Edge list of shape (n_edges, 2)
    """
    # Load features from .npy file as float32 to save memory
    features = np.load(file_path).astype(np.float32)
    
    return build_knn_graph(features, k=k, metric=metric)


def build_knn_graph_from_csv(
    csv_path: Union[str, Path],
    data_type: str = 'raw',
    n_pcs: Optional[int] = None,
    k: int = 15,
    metric: str = 'cosine',
) -> np.ndarray:
    """
    Build k-NN graph from CSV file (typically environment features).
    
    DEPRECATED: Use build_knn_graph_from_file() for .npy files from features pipeline.
    This function is kept for backwards compatibility with CSV-based workflows.
    
    Args:
        csv_path: Path to CSV file with features
        data_type: Feature type ('raw', 'pca15', 'pca50')
        n_pcs: Number of PCA components (overrides data_type if specified)
        k: Number of nearest neighbors
        metric: Distance metric
        
    Returns:
        Edge list of shape (n_edges, 2)
    """
    import pandas as pd
    
    # Load data
    df = pd.read_csv(csv_path, index_col=0)
    features = df.select_dtypes(include=[np.number]).values
    
    # Apply PCA if requested
    if data_type == 'pca15':
        n_pcs = 15
    elif data_type == 'pca50':
        n_pcs = 50
    
    if n_pcs is not None:
        pca = PCA(n_components=n_pcs, random_state=42)
        features = pca.fit_transform(features)
    
    return build_knn_graph(features, k=k, metric=metric)


def save_edge_list(
    edge_list: np.ndarray,
    output_path: Union[str, Path],
) -> Path:
    """
    Save edge list to numpy file.
    
    Args:
        edge_list: Edge list array
        output_path: Output file path
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, edge_list)
    return output_path


def load_edge_list(path: Union[str, Path]) -> np.ndarray:
    """
    Load edge list from numpy file.
    
    Args:
        path: Path to .npy file
        
    Returns:
        Edge list array
    """
    return np.load(path)


def edge_list_to_adjacency(
    edge_list: np.ndarray,
    n_nodes: Optional[int] = None,
) -> 'scipy.sparse.csr_matrix':
    """
    Convert edge list to sparse adjacency matrix.
    
    Args:
        edge_list: Edge list of shape (n_edges, 2)
        n_nodes: Number of nodes (inferred from edge list if not provided)
        
    Returns:
        Sparse adjacency matrix
    """
    import scipy.sparse as sp
    
    if n_nodes is None:
        n_nodes = int(edge_list.max()) + 1
    
    rows = edge_list[:, 0]
    cols = edge_list[:, 1]
    data = np.ones(len(edge_list), dtype=np.float32)
    
    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))


def build_folder_name(
    method: str,
    data_type: str,
    metric: str,
    feature_type: str = 'celltype',
    env_k: Optional[int] = None,
    weighted: bool = False,
) -> str:
    """
    Build folder name following the path conventions.
    
    Args:
        method: Clustering method ('Leiden', 'PhenoGraph', 'Kmeans', 'LDA')
        data_type: Data type ('raw', 'pca15', 'pca50', 'scvi', 'log1p', 'localfreq_k50', etc.)
        metric: Distance metric ('cosine', 'correlation', 'euclidean')
        feature_type: 'celltype' or 'env' (legacy)
        env_k: Environment k value (for env features, legacy)
        weighted: Whether environment is weighted (legacy)
        
    Returns:
        Folder name string
    
    Examples:
        >>> build_folder_name('PhenoGraph', 'scvi', 'euclidean')
        'PhenoGraphSCVIEuclidean'
        >>> build_folder_name('Leiden', 'raw', 'correlation')
        'LeidenRawCorrelation'
        >>> build_folder_name('Leiden', 'localfreq_k50', 'cosine')
        'LeidenLocalfreqK50Cosine'
        >>> build_folder_name('PhenoGraph', 'localfreq_w_k100', 'correlation')
        'PhenoGraphLocalfreqWK100Correlation'
        >>> build_folder_name('Leiden', 'localfreq_pca15_k50', 'euclidean')
        'LeidenLocalfreqPca15K50Euclidean'
    """
    # Handle local frequency features
    if data_type.startswith('localfreq'):
        # Parse localfreq names: localfreq_k50, localfreq_w_k50, localfreq_pca15_k50, etc.
        parts = data_type.split('_')
        
        # Build capitalized version
        data_str_parts = []
        for part in parts:
            if part == 'localfreq':
                data_str_parts.append('Localfreq')
            elif part == 'w':
                data_str_parts.append('W')
            elif part.startswith('pca') and part[3:].isdigit():
                # pca15 -> Pca15
                data_str_parts.append('Pca' + part[3:])
            elif part.startswith('k') and part[1:].isdigit():
                # k50 -> K50
                data_str_parts.append('K' + part[1:])
            else:
                data_str_parts.append(part.capitalize())
        
        data_str = ''.join(data_str_parts)
    
    # Handle traditional expression features
    elif data_type == 'scvi':
        data_str = 'SCVI'
    elif data_type == 'log1p':
        data_str = 'Log1p'
    elif data_type.startswith('pca'):
        data_str = f"PCA{data_type[3:]}"
    else:
        data_str = data_type.capitalize()
    
    metric_str = metric.capitalize()
    
    folder = f"{method}{data_str}{metric_str}"
    
    # Legacy env feature support
    if feature_type == 'env':
        folder += '_envenv'
        if weighted:
            folder += '_w'
        if env_k is not None:
            folder += f'_k_{env_k}'
    
    return folder
