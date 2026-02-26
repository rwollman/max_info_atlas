"""
Percolation worker for processing type vector files.

This worker runs percolation analysis on type vector files and saves results.
"""

import numpy as np
from pathlib import Path
from typing import Union

from .graph_percolation import GraphPercolation, EdgeListManager


def process_percolation_chunk(
    chunk_file: Union[str, Path],
    type_data_base_dir: Union[str, Path],
    xy_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    edge_list_dir: Union[str, Path],
    max_k: int = 100,
) -> int:
    """
    Process a chunk of type vector files and run percolation analysis.
    
    This worker:
    1. Reads type vector .npy files from chunk list
    2. Loads corresponding XY coordinates from xy_data_dir (section name based)
    3. Runs percolation analysis  
    4. Saves .npz (full results) and .score (just score) files
    
    Args:
        chunk_file: Path to chunk file containing list of type vector paths
        type_data_base_dir: Base directory for type vector files
        xy_data_dir: Directory containing XY coordinate files ({section_name}_XY.npy)
        output_dir: Output directory for percolation results
        edge_list_dir: Directory for edge list cache
        max_k: Maximum k for percolation (default: 100)
        
    Returns:
        Number of files processed
    """
    chunk_file = Path(chunk_file)
    type_data_base_dir = Path(type_data_base_dir)
    xy_data_dir = Path(xy_data_dir)
    output_dir = Path(output_dir)
    edge_list_dir = Path(edge_list_dir)
    
    # Create output and edge list directories
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_list_dir.mkdir(parents=True, exist_ok=True)
    
    # Read chunk file
    with open(chunk_file, 'r') as f:
        type_files = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(type_files)} files from chunk...")
    
    # Initialize edge list manager
    elm = EdgeListManager(base_dir=str(edge_list_dir))
    
    processed = 0
    
    for type_file_rel in type_files:
        try:
            # Full path to type vector file
            type_file = type_data_base_dir / type_file_rel
            
            if not type_file.exists():
                print(f"  Skipping missing file: {type_file_rel}")
                continue
            
            # Load type vector
            type_vec = np.load(type_file)
            
            # Load XY coordinates by section name
            # Extract section name from type vector filename (e.g., res_0/section_A.npy -> section_A)
            section_name = type_file.stem  # Get filename without extension
            xy_file = xy_data_dir / f"{section_name}_XY.npy"
            
            if not xy_file.exists():
                print(f"  Skipping (no XY file): {type_file_rel}")
                print(f"    Expected XY file: {xy_file}")
                continue
            
            XY = np.load(xy_file)
            
            # Verify XY and type_vec have same length
            if len(XY) != len(type_vec):
                print(f"  Skipping (XY/type mismatch): {type_file_rel}")
                print(f"    XY shape: {XY.shape}, type_vec length: {len(type_vec)}")
                continue
            
            # Run percolation
            gp = GraphPercolation(XY, type_vec, maxK=max_k)
            gp.percolation(edge_list_manager=elm, xy_name=type_file.stem)
            
            # Create output path matching input structure
            output_file_rel = Path(type_file_rel)
            output_npz = output_dir / output_file_rel
            output_npz.parent.mkdir(parents=True, exist_ok=True)
            
            # Save full results
            gp.save(str(output_npz))
            
            # Save both scores separately for efficient aggregation
            score = gp.raw_score()
            normalized_score = gp.normalized_score()
            score_file = output_npz.with_suffix('.score')
            with open(score_file, 'w') as f:
                f.write(f"{score}\t{normalized_score}\n")
            
            processed += 1
            
        except Exception as e:
            print(f"  ERROR processing {type_file_rel}: {e}")
            continue
    
    print(f"✓ Processed {processed}/{len(type_files)} files")
    return processed
