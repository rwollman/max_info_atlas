"""Create subsampled test datasets for end-to-end testing.

This script creates small test datasets with a subset of sections and cells
for faster end-to-end testing of the pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from anndata import AnnData
import anndata


def create_test_dataset(
    input_h5ad: str | Path,
    output_dir: str | Path,
    n_sections: int = 3,
    cells_per_section: int = 1000,
    section_column: str = "brain_section_label",
    output_name: str = "test_allen_3sections.h5ad",
    random_seed: int | None = 42,
) -> Path:
    """Create a subsampled test dataset.
    
    Args:
        input_h5ad: Path to original h5ad file
        output_dir: Directory to save test data
        n_sections: Number of sections to include
        cells_per_section: Maximum cells per section
        section_column: Column name in adata.obs containing section IDs
        output_name: Name for output h5ad file
        random_seed: Random seed for reproducibility (None for no seed)
        
    Returns:
        Path to created h5ad file
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Convert paths
    input_h5ad = Path(input_h5ad)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading metadata only from: {input_h5ad}")
    
    # Step 1: Use h5py to read only obs and var DataFrames (no X, no layers)
    with h5py.File(input_h5ad, 'r') as f:
        # Read obs DataFrame
        obs_df = anndata.experimental.read_elem(f['obs'])
        
        # Validate section column exists
        if section_column not in obs_df.columns:
            raise ValueError(
                f"Section column '{section_column}' not found in obs. "
                f"Available columns: {list(obs_df.columns)}"
            )
        
        # Get unique sections (only working with metadata)
        sections = obs_df[section_column].unique()
        print(f"Found {len(sections)} sections in dataset")
        
        # Select n sections
        test_sections = sections[:n_sections]
        print(f"Selected sections: {test_sections}")
        
        # Subsample cells per section (still only metadata)
        indices = []
        for section in test_sections:
            section_mask = obs_df[section_column] == section
            section_indices = np.where(section_mask)[0]
            
            # Sample up to cells_per_section
            n_sample = min(cells_per_section, len(section_indices))
            sampled = np.random.choice(section_indices, n_sample, replace=False)
            indices.extend(sampled)
            
            print(f"  {section}: sampled {n_sample} cells")
        
        # Convert to sorted array for efficient h5py access
        indices = np.sort(np.array(indices))
        
        print(f"\nExtracting {len(indices)} cells from file...")
        print(f"(Ignoring all layers, only loading X matrix)")
        
        # Step 2: Extract ONLY X matrix for selected indices (no layers!)
        X_data = f['X']
        
        # Handle different X storage formats
        if isinstance(X_data, h5py.Group):
            # Sparse CSR format
            print("  Reading sparse CSR matrix...")
            indptr = X_data['indptr'][:]
            
            # Only load the data and indices for our selected rows
            row_start = indptr[indices[0]]
            row_end = indptr[indices[-1] + 1]
            
            data = X_data['data'][row_start:row_end]
            col_indices = X_data['indices'][row_start:row_end]
            
            # Adjust indptr to be relative to our subset
            indptr_subset = indptr[indices] - row_start
            indptr_subset = np.append(indptr_subset, len(data))
            
            shape = (len(indices), X_data.attrs['shape'][1])
            X_subset = sparse.csr_matrix((data, col_indices, indptr_subset), shape=shape)
        else:
            # Dense matrix
            print("  Reading dense matrix...")
            X_subset = X_data[indices, :]
        
        # Get metadata for selected cells
        obs_subset = obs_df.iloc[indices].copy()
        
        # Read var DataFrame
        var_subset = anndata.experimental.read_elem(f['var'])
    
    # Step 3: Create new AnnData with just X, obs, var (explicitly no layers)
    adata_test = AnnData(X=X_subset, obs=obs_subset, var=var_subset)
    
    print(f"\nTest dataset summary:")
    print(f"  Total cells: {len(adata_test)}")
    print(f"  Cells per section:")
    for section, count in adata_test.obs[section_column].value_counts().items():
        print(f"    {section}: {count}")
    
    # Save h5ad
    output_h5ad = output_dir / output_name
    adata_test.write_h5ad(output_h5ad)
    print(f"\nSaved h5ad to: {output_h5ad}")
    
    # Save sections array
    sections_arr = adata_test.obs[section_column].values
    sections_file = output_dir / "Sections.npy"
    np.save(sections_file, sections_arr)
    print(f"Saved sections to: {sections_file}")
    
    # Extract and save XY coordinates per section
    xy_dir = output_dir / "xy_data"
    xy_dir.mkdir(exist_ok=True)
    
    # Check if spatial coordinates exist
    if 'x' in adata_test.obs.columns and 'y' in adata_test.obs.columns:
        print(f"\nExtracting XY coordinates from obs['x'] and obs['y']")
        xy_coords = adata_test.obs[['x', 'y']].values
        
        # Save XY coordinates per section
        for section in adata_test.obs[section_column].unique():
            section_mask = adata_test.obs[section_column] == section
            xy_section = xy_coords[section_mask]
            
            xy_file = xy_dir / f"{section}_XY.npy"
            np.save(xy_file, xy_section)
            print(f"  Saved XY for {section}: {xy_file} (shape: {xy_section.shape})")
    else:
        print(f"\nWARNING: No XY coordinates found!")
        print(f"  Available obs columns: {list(adata_test.obs.columns[:10])}...")
        print(f"  XY coordinates are required for percolation analysis")
    
    return output_h5ad


def main():
    """Command-line interface for creating test datasets."""
    parser = argparse.ArgumentParser(
        description="Create subsampled test dataset for end-to-end testing"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input h5ad file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save test data",
    )
    parser.add_argument(
        "--n-sections",
        type=int,
        default=3,
        help="Number of sections to include (default: 3)",
    )
    parser.add_argument(
        "--cells-per-section",
        type=int,
        default=1000,
        help="Maximum cells per section (default: 1000)",
    )
    parser.add_argument(
        "--section-column",
        type=str,
        default="brain_section_label",
        help="Column name in adata.obs containing section IDs (default: 'brain_section_label')",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="test_allen_3sections.h5ad",
        help="Name for output h5ad file (default: 'test_allen_3sections.h5ad')",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    create_test_dataset(
        input_h5ad=args.input,
        output_dir=args.output_dir,
        n_sections=args.n_sections,
        cells_per_section=args.cells_per_section,
        section_column=args.section_column,
        output_name=args.output_name,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
