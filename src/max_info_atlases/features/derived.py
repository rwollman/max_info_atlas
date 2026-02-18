"""
Derived feature operations.

Compute derived features from existing feature matrices (e.g., PCA of local frequencies).
Extensible framework for adding new operations in the future.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_pca(
    features: np.ndarray,
    n_components: int,
    scale: bool = True,
) -> np.ndarray:
    """
    Apply PCA to feature matrix.
    
    Args:
        features: Input feature matrix, shape (n_samples, n_features)
        n_components: Number of principal components to compute
        scale: Whether to standardize features before PCA
        
    Returns:
        PCA-transformed features, shape (n_samples, n_components)
    """
    # Standardize features if requested
    if scale:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = features
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    
    return features_pca.astype(np.float32)


def compute_derived_feature(
    source_features: np.ndarray,
    operation: str,
    params: Dict,
) -> np.ndarray:
    """
    Compute derived feature using specified operation.
    
    Args:
        source_features: Source feature matrix
        operation: Operation name ('pca', etc.)
        params: Operation-specific parameters
        
    Returns:
        Derived feature matrix
        
    Raises:
        ValueError: If operation is not supported
    """
    if operation == 'pca':
        n_components = params.get('n_components')
        if n_components is None:
            raise ValueError("PCA operation requires 'n_components' parameter")
        scale = params.get('scale', True)
        return apply_pca(source_features, n_components, scale=scale)
    
    # Add more operations here in the future
    # elif operation == 'umap':
    #     return apply_umap(source_features, **params)
    # elif operation == 'ica':
    #     return apply_ica(source_features, **params)
    
    else:
        raise ValueError(f"Unsupported derived feature operation: '{operation}'")


def build_derived_feature_name(
    source_name: str,
    operation: str,
    params: Dict,
) -> str:
    """
    Build standardized name for derived feature.
    
    Args:
        source_name: Name of source feature (e.g., 'localfreq_k50')
        operation: Operation name (e.g., 'pca')
        params: Operation parameters
        
    Returns:
        Derived feature name (e.g., 'localfreq_pca15_k50')
        
    Examples:
        >>> build_derived_feature_name('localfreq_k50', 'pca', {'n_components': 15})
        'localfreq_pca15_k50'
        >>> build_derived_feature_name('localfreq_w_k100', 'pca', {'n_components': 50})
        'localfreq_w_pca50_k100'
    """
    if operation == 'pca':
        n_components = params.get('n_components')
        if n_components is None:
            raise ValueError("PCA operation requires 'n_components' parameter")
        
        # Insert pca{n} after the base feature type
        # localfreq_k50 -> localfreq_pca15_k50
        # localfreq_w_k50 -> localfreq_w_pca15_k50
        
        # Split on underscore to find insertion point
        parts = source_name.split('_')
        
        # Find where to insert the pca component
        # We want: localfreq[_w]_pca{n}_k{k}
        if 'localfreq' in parts[0]:
            # Check if weighted
            if len(parts) > 1 and parts[1] == 'w':
                # localfreq_w_k50 -> ['localfreq', 'w', 'k50']
                # Insert after 'w': ['localfreq', 'w', 'pca15', 'k50']
                parts.insert(2, f'pca{n_components}')
            else:
                # localfreq_k50 -> ['localfreq', 'k50']
                # Insert after 'localfreq': ['localfreq', 'pca15', 'k50']
                parts.insert(1, f'pca{n_components}')
        else:
            # For other feature types, append operation
            parts.insert(-1, f'pca{n_components}')
        
        return '_'.join(parts)
    
    # For future operations, add similar logic
    else:
        # Default: prepend operation name
        return f"{source_name}_{operation}"


def extract_and_save_derived_feature(
    source_feature_path: Union[str, Path],
    output_dir: Union[str, Path],
    operation: str,
    params: Dict,
    output_name: Optional[str] = None,
) -> Path:
    """
    Load source feature, apply operation, and save result.
    
    Args:
        source_feature_path: Path to source feature .npy file
        output_dir: Output directory for derived features
        operation: Operation name ('pca', etc.)
        params: Operation-specific parameters
        output_name: Optional output feature name (auto-generated if None)
        
    Returns:
        Path to saved derived feature file
    """
    source_path = Path(source_feature_path)
    
    # Load source features
    print(f"Loading source features from {source_path}")
    source_features = np.load(source_path)
    
    # Compute derived feature
    print(f"Applying {operation} with params {params}")
    derived_features = compute_derived_feature(source_features, operation, params)
    
    # Generate output name if not provided
    if output_name is None:
        # Extract source feature name from filename
        # features_localfreq_k50.npy -> localfreq_k50
        source_name = source_path.stem.replace('features_', '')
        output_name = build_derived_feature_name(source_name, operation, params)
    
    # Save derived features
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"features_{output_name}.npy"
    np.save(output_path, derived_features)
    
    print(f"Saved derived features to {output_path} (shape: {derived_features.shape})")
    
    return output_path


def parse_derived_config(derived_specs: List[Dict]) -> List[Dict]:
    """
    Parse derived feature specifications from config.
    
    Args:
        derived_specs: List of derived feature specifications from YAML
        
    Returns:
        List of parsed specifications with expanded parameters
        
    Example input:
        [
            {'operation': 'pca', 'source': 'local_frequency', 'n_components': [15, 50]},
            {'operation': 'pca', 'source': 'localfreq_k50', 'n_components': [15]}
        ]
        
    Example output:
        [
            {'operation': 'pca', 'source': 'local_frequency', 'n_components': 15},
            {'operation': 'pca', 'source': 'local_frequency', 'n_components': 50},
            {'operation': 'pca', 'source': 'localfreq_k50', 'n_components': 15}
        ]
    """
    parsed = []
    
    for spec in derived_specs:
        operation = spec.get('operation')
        source = spec.get('source')
        
        if operation == 'pca':
            # Expand n_components if it's a list
            n_components_list = spec.get('n_components', [])
            if not isinstance(n_components_list, list):
                n_components_list = [n_components_list]
            
            for n_comp in n_components_list:
                parsed.append({
                    'operation': operation,
                    'source': source,
                    'n_components': n_comp,
                    'scale': spec.get('scale', True),
                })
        else:
            # For future operations, add similar logic
            parsed.append(spec)
    
    return parsed
