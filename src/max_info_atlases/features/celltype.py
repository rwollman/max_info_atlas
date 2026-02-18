"""
Cell type feature extraction from AnnData.

Supports extracting features from various representations:
- raw: Raw gene expression counts
- log1p: Log-normalized counts
- pca: PCA-reduced features (auto-computed if not present)
- scvi: scVI latent space (auto-computed if not present)

Note: scVI will be automatically trained if adata.obsm['X_scvi'] is not found.
This requires scvi-tools to be installed: pip install scvi-tools
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path


def extract_celltype_features(
    adata,
    data_type: str = 'raw',
    n_pcs: int = 50,
    scvi_n_latent: int = 30,
    scvi_max_epochs: int = 200,
) -> np.ndarray:
    """
    Extract cell type features from AnnData object.
    
    Args:
        adata: AnnData object with gene expression data
        data_type: Feature type to extract ('raw', 'log1p', 'pca', 'scvi')
        n_pcs: Number of principal components (for PCA)
        scvi_n_latent: Number of latent dimensions for scVI (default: 30)
        scvi_max_epochs: Maximum training epochs for scVI (default: 200)
        
    Returns:
        Feature matrix of shape (n_cells, n_features)
        
    Raises:
        ValueError: If data_type is not supported
    """
    data_type = data_type.lower()
    
    if data_type == 'raw':
        return _extract_raw(adata)
    elif data_type == 'log1p':
        return _extract_log1p(adata)
    elif data_type in ('pca', 'pca15', 'pca50'):
        if data_type == 'pca15':
            n_pcs = 15
        elif data_type == 'pca50':
            n_pcs = 50
        return _extract_pca(adata, n_pcs)
    elif data_type == 'scvi':
        return _extract_scvi(adata, n_latent=scvi_n_latent, max_epochs=scvi_max_epochs)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def _extract_raw(adata) -> np.ndarray:
    """Extract raw counts from AnnData."""
    # Check for preprocessed raw layer
    if 'raw' in adata.layers:
        X = adata.layers['raw']
    else:
        X = adata.X
    
    # Convert sparse to dense if needed, and use float32 to save memory
    if hasattr(X, 'toarray'):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def _extract_log1p(adata) -> np.ndarray:
    """Extract log1p-normalized counts from AnnData."""
    import scanpy as sc
    
    # Check for preprocessed log1p layer
    if 'log1p' in adata.layers:
        X = adata.layers['log1p']
        if hasattr(X, 'toarray'):
            return X.toarray().astype(np.float32)
        return np.asarray(X, dtype=np.float32)
    
    # Compute log1p on a copy
    adata_copy = adata.copy()
    sc.pp.log1p(adata_copy)
    
    X = adata_copy.X
    if hasattr(X, 'toarray'):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def _extract_pca(adata, n_pcs: int = 50) -> np.ndarray:
    """Extract PCA features from AnnData."""
    import scanpy as sc
    
    # Check for precomputed PCA
    if 'X_pca' in adata.obsm:
        pca = adata.obsm['X_pca']
        # Return requested number of components as float32
        return pca[:, :min(n_pcs, pca.shape[1])].astype(np.float32)
    
    # Compute PCA on a copy
    adata_copy = adata.copy()
    
    # Standard preprocessing
    sc.pp.log1p(adata_copy)
    sc.pp.scale(adata_copy)
    sc.tl.pca(adata_copy, n_comps=n_pcs)
    
    return adata_copy.obsm['X_pca'].astype(np.float32)


def _extract_scvi(adata, n_latent: int = 30, max_epochs: int = 200) -> np.ndarray:
    """
    Extract scVI latent space from AnnData, computing it if not present.
    
    Args:
        adata: AnnData object
        n_latent: Number of latent dimensions (default: 30)
        max_epochs: Maximum training epochs (default: 200)
        
    Returns:
        scVI latent representation
    """
    # Check for precomputed scVI
    if 'X_scvi' in adata.obsm:
        print("Using precomputed scVI latent space from adata.obsm['X_scvi']")
        return adata.obsm['X_scvi'].astype(np.float32)
    
    # Compute scVI automatically
    print("scVI latent space not found, computing it now...")
    print(f"  n_latent={n_latent}, max_epochs={max_epochs}")
    
    try:
        import scvi
    except ImportError:
        raise ImportError(
            "scvi-tools is not installed. Install with: pip install scvi-tools"
        )
    
    # Make a copy to avoid modifying the original
    import scanpy as sc
    adata_copy = adata.copy()
    
    # Setup scVI
    print("  Setting up scVI model...")
    scvi.model.SCVI.setup_anndata(adata_copy)
    
    # Train model
    print("  Training scVI model...")
    model = scvi.model.SCVI(adata_copy, n_latent=n_latent)
    model.train(max_epochs=max_epochs, early_stopping=True)
    
    # Get latent representation
    print("  Extracting latent representation...")
    latent = model.get_latent_representation().astype(np.float32)
    
    # Store in original adata for future use
    adata.obsm['X_scvi'] = latent
    
    print(f"  Done! scVI latent shape: {latent.shape}")
    
    return latent


def prepare_data_for_clustering(
    adata,
    data_type: str,
    n_pcs: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for clustering, returning features and section labels.
    
    Args:
        adata: AnnData object
        data_type: Feature type to extract
        n_pcs: Number of principal components
        
    Returns:
        Tuple of (features, section_labels)
    """
    features = extract_celltype_features(adata, data_type, n_pcs)
    
    # Get section labels if available
    if 'section' in adata.obs.columns:
        sections = adata.obs['section'].values
    elif 'Section' in adata.obs.columns:
        sections = adata.obs['Section'].values
    else:
        # Single section
        sections = np.zeros(len(adata), dtype=int)
    
    return features, sections


def _get_section_column(adata) -> np.ndarray:
    """Get section labels from AnnData .obs."""
    if 'section' in adata.obs.columns:
        return adata.obs['section'].values
    elif 'Section' in adata.obs.columns:
        return adata.obs['Section'].values
    else:
        raise ValueError(
            "No section column found in adata.obs. "
            "Expected 'section' or 'Section'."
        )


def extract_preexisting_annotations(
    adata,
    output_dir,
    obs_columns: dict,
    section_column: Optional[str] = None,
    sections_array: Optional[np.ndarray] = None,
) -> dict:
    """
    Extract preexisting annotations from AnnData .obs and save
    as per-section integer type vector .npy files.
    
    The percolation pipeline expects integer-encoded type vectors in
    {output_dir}/Preexisting_{level}/{section}.npy format. This function
    reads categorical/string annotations from .obs columns, encodes
    them as integers, and saves one .npy file per section.
    
    Args:
        adata: AnnData object with .obs containing annotation columns
        output_dir: Base output directory (files go into Preexisting_{level}/ subdirs)
        obs_columns: Mapping from level name to .obs column name.
            Example: {'class': 'class', 'subclass': 'subclass',
                      'cluster': 'cluster', 'supertype': 'supertype'}
        section_column: Name of section column in .obs (auto-detected if None)
        
    Returns:
        Dict with summary info per level:
            {level: {'n_categories': int, 'n_sections': int, 'n_cells': int}}
    
    Example:
        extract_preexisting_annotations(
            adata,
            output_dir='/results/clustering',
            obs_columns={'class': 'class', 'subclass': 'subclass'},
        )
        # Creates:
        #   /results/clustering/Preexisting_class/section1.npy
        #   /results/clustering/Preexisting_class/section2.npy
        #   /results/clustering/Preexisting_subclass/section1.npy
        #   /results/clustering/Preexisting_subclass/section2.npy
    """
    output_dir = Path(output_dir)
    
    # Get section labels
    if sections_array is not None:
        sections = sections_array
    elif section_column is not None:
        sections = adata.obs[section_column].values
    else:
        sections = _get_section_column(adata)
    
    unique_sections = np.unique(sections)
    summary = {}
    
    for level_name, obs_col in obs_columns.items():
        if obs_col not in adata.obs.columns:
            raise ValueError(
                f"Column '{obs_col}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        
        # Get annotation values
        annotations = adata.obs[obs_col].values
        
        # Encode as integers
        # Use pandas Categorical for consistent encoding
        import pandas as pd
        cat = pd.Categorical(annotations)
        type_vec = cat.codes.astype(np.int32)
        
        # Create output directory
        level_dir = output_dir / f"Preexisting_{level_name}"
        level_dir.mkdir(parents=True, exist_ok=True)
        
        # Save per section
        n_sections = 0
        for section in unique_sections:
            mask = sections == section
            section_types = type_vec[mask]
            
            output_file = level_dir / f"{section}.npy"
            np.save(output_file, section_types)
            n_sections += 1
        
        summary[level_name] = {
            'n_categories': len(cat.categories),
            'categories': list(cat.categories),
            'n_sections': n_sections,
            'n_cells': len(type_vec),
            'output_dir': str(level_dir),
        }
        
        print(f"  Preexisting_{level_name}: {len(cat.categories)} categories, "
              f"{n_sections} sections, {len(type_vec)} cells -> {level_dir}")
    
    return summary
