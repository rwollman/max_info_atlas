"""
Percolation analysis utilities.

Includes score calculation and result aggregation from chunked parallel jobs.
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

from .graph_percolation import GraphPercolation
from ..path_utils import parse_path, extract_metadata_from_path


def calculate_percolation_score(npz_file: Union[str, Path]) -> Tuple[Optional[Tuple[float, float]], Dict[str, float]]:
    """
    Calculate percolation scores from a saved .npz file.
    
    Args:
        npz_file: Path to the .npz file containing percolation results
        
    Returns:
        Tuple of ((raw_score, normalized_score), timing_dict) where scores are floats or None on error,
        and timing_dict contains 'load_time', 'score_time', 'total_time'
    """
    import time
    
    start_time = time.time()
    timing = {'load_time': 0.0, 'score_time': 0.0, 'total_time': 0.0}
    
    try:
        # Check if file exists and is non-empty
        npz_file = Path(npz_file)
        if not npz_file.exists():
            return None, timing
        
        if npz_file.stat().st_size == 0:
            return None, timing
        
        # Load data
        load_start = time.time()
        data = np.load(npz_file)
        timing['load_time'] = time.time() - load_start
        
        # Validate required keys
        required_keys = ['XY', 'type_vec', 'ent_real', 'ent_perm', 'pbond_vec']
        missing_keys = [key for key in required_keys if key not in data.keys()]
        if missing_keys:
            return None, timing
        
        # Create GraphPercolation instance and load data
        dummy_xy = np.array([[0, 0], [1, 1]])
        dummy_type_vec = np.array([0, 1])
        gp = GraphPercolation(XY=dummy_xy, type_vec=dummy_type_vec)
        gp.load(str(npz_file))
        
        # Calculate both scores
        score_start = time.time()
        score = gp.raw_score()
        normalized_score = gp.normalized_score()
        timing['score_time'] = time.time() - score_start
        
        # Validate scores
        if score is None or np.isnan(score) or np.isinf(score):
            return None, timing
        if normalized_score is None or np.isnan(normalized_score) or np.isinf(normalized_score):
            return None, timing
        
        timing['total_time'] = time.time() - start_time
        return (score, normalized_score), timing
        
    except Exception as e:
        timing['total_time'] = time.time() - start_time
        return None, timing


def process_chunk(
    chunk_file: Union[str, Path],
    base_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> pd.DataFrame:
    """
    Process a chunk of .npz files and return results as DataFrame.
    
    Args:
        chunk_file: Path to chunk file containing list of .npz paths
        base_dir: Base directory for percolation results
        output_dir: Directory for temporary output files
        
    Returns:
        DataFrame with percolation scores and metadata
    """
    # Read chunk file
    with open(chunk_file, 'r') as f:
        npz_files = [line.strip() for line in f if line.strip()]
    
    results = []
    
    for npz_file in npz_files:
        # Construct full path
        full_path = os.path.join(base_dir, npz_file)
        
        if not os.path.exists(full_path):
            continue
        
        try:
            # Extract metadata from path
            metadata = extract_metadata_from_path(full_path, str(base_dir))
            
            # Calculate scores
            scores, timing = calculate_percolation_score(full_path)
            
            # Add scores and timing
            if scores is not None:
                metadata['percolation_score'] = scores[0]
                metadata['normalized_percolation_score'] = scores[1]
            else:
                metadata['percolation_score'] = None
                metadata['normalized_percolation_score'] = None
            metadata['load_time_seconds'] = timing['load_time']
            metadata['score_time_seconds'] = timing['score_time']
            metadata['total_time_seconds'] = timing['total_time']
            
            results.append(metadata)
            
        except Exception as e:
            # Skip files that fail to process
            continue
    
    # Create DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Convert numeric columns
        for col in ['env_k', 'clustering_k', 'algorithm_k']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    return pd.DataFrame()


def combine_chunk_results(
    temp_dir: Union[str, Path],
    output_file: Union[str, Path],
    cleanup: bool = False,
) -> pd.DataFrame:
    """
    Combine temporary CSV files from chunked analysis into final results.
    
    Args:
        temp_dir: Directory containing chunk_result_*.csv files
        output_file: Path for combined output CSV
        cleanup: If True, remove temp files after combining
        
    Returns:
        Combined DataFrame
    """
    temp_dir = Path(temp_dir)
    output_file = Path(output_file)
    
    # Find all chunk result files
    pattern = str(temp_dir / "chunk_result_*.csv")
    temp_files = glob.glob(pattern)
    
    if not temp_files:
        print(f"No chunk result files found in {temp_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(temp_files)} chunk result files")
    
    # Load and combine
    dfs = []
    failed_files = []
    
    for temp_file in temp_files:
        try:
            df = pd.read_csv(temp_file, low_memory=False)
            
            # Validate required columns
            required = ['algorithm', 'section_name', 'percolation_score', 'file_path']
            missing = [col for col in required if col not in df.columns]
            
            if missing or len(df) == 0:
                failed_files.append(temp_file)
                continue
            
            dfs.append(df)
            
        except Exception as e:
            failed_files.append(temp_file)
    
    if failed_files:
        print(f"Warning: Failed to load {len(failed_files)} files")
    
    if not dfs:
        print("No valid data to combine")
        return pd.DataFrame()
    
    # Combine
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert numeric columns
    for col in ['env_k', 'clustering_k', 'algorithm_k']:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    # Sort
    sort_cols = ['algorithm', 'env_k', 'clustering_k', 'section_name']
    available_sort = [col for col in sort_cols if col in combined_df.columns]
    combined_df = combined_df.sort_values(available_sort)
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Total files: {len(combined_df)}")
    print(f"Valid scores: {combined_df['percolation_score'].notna().sum()}")
    print(f"Missing scores: {combined_df['percolation_score'].isna().sum()}")
    
    if combined_df['percolation_score'].notna().any():
        print(f"Score range: {combined_df['percolation_score'].min():.6f} to "
              f"{combined_df['percolation_score'].max():.6f}")
        print(f"Mean score: {combined_df['percolation_score'].mean():.6f}")
    
    print(f"\nSaved to: {output_file}")
    
    # Cleanup if requested
    if cleanup:
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception:
                pass
        print(f"Cleaned up {len(temp_files)} temp files")
    
    return combined_df


def combine_score_files(
    base_dir: Union[str, Path],
    output_file: Union[str, Path],
) -> List[str]:
    """
    Combine individual .score files into a single text file.
    
    Args:
        base_dir: Directory to search for .score files
        output_file: Output file path
        
    Returns:
        List of all scores
    """
    base_dir = Path(base_dir)
    output_file = Path(output_file)
    
    # Find all .score files
    score_files = list(base_dir.rglob("*.score"))
    print(f"Found {len(score_files)} score files")
    
    # Read all scores
    all_scores = []
    for score_file in score_files:
        try:
            with open(score_file, 'r') as f:
                content = f.read().strip()
                if content:
                    all_scores.append(content)
        except Exception:
            continue
    
    # Write combined file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for score in all_scores:
            f.write(score + "\n")
    
    print(f"Combined {len(all_scores)} scores into {output_file}")
    
    return all_scores


def aggregate_score_files_to_csv(
    base_dir: Union[str, Path],
    output_csv: Union[str, Path],
) -> pd.DataFrame:
    """
    Aggregate .score files into a CSV with metadata extracted from paths.
    
    This is the EFFICIENT way to aggregate percolation results at scale:
    - .score files are ~10 bytes each vs .npz files at KB-MB
    - No need to load large arrays (XY, type_vec, entropy curves)
    - Can aggregate millions of results without memory issues
    - Perfect for distributed workflows with separate aggregation step
    
    The .npz files are preserved for detailed analysis if needed.
    
    Args:
        base_dir: Base directory containing percolation results with .score files
        output_csv: Output CSV file path
        
    Returns:
        DataFrame with scores and metadata
    """
    base_dir = Path(base_dir)
    output_csv = Path(output_csv)
    
    # Find all .score files
    score_files = list(base_dir.rglob("*.score"))
    print(f"Found {len(score_files)} .score files")
    
    if not score_files:
        print("No .score files found")
        return pd.DataFrame()
    
    results = []
    failed_count = 0
    
    for score_file in score_files:
        try:
            # Read score - handle multiple formats:
            # v1 format (tab-separated): section_name\ttype_path\tsection_name\tscore
            # v2 format (single value): score
            # v3/v4 format (two values): raw_score\tnormalized_score
            # This provides backward compatibility with data generated by max_info v1 and v2
            with open(score_file, 'r') as f:
                score_str = f.read().strip()
                if not score_str:
                    failed_count += 1
                    continue
                
                score = None
                normalized_score = None
                
                # Check if it's tab-separated format
                if '\t' in score_str:
                    parts = score_str.split('\t')
                    if len(parts) >= 4:
                        # v1 format - score is last column
                        score = float(parts[-1])
                    elif len(parts) == 2:
                        # v3/v4 format - raw_score and normalized_score
                        score = float(parts[0])
                        normalized_score = float(parts[1])
                    else:
                        failed_count += 1
                        continue
                else:
                    # v2 format - single float value
                    score = float(score_str)
            
            # Extract metadata from path (pass full path, function will compute relative path)
            metadata = extract_metadata_from_path(str(score_file), str(base_dir))
            relative_path = str(score_file.relative_to(base_dir))
            
            # Add scores
            metadata['percolation_score'] = score
            if normalized_score is not None:
                metadata['normalized_percolation_score'] = normalized_score
            metadata['file_path'] = relative_path
            
            results.append(metadata)
            
        except Exception as e:
            # Skip files that fail to process
            failed_count += 1
            continue
    
    if not results:
        print(f"No valid scores extracted (failed: {failed_count})")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Convert numeric columns
    for col in ['env_k', 'clustering_k', 'algorithm_k']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort
    sort_cols = ['algorithm', 'env_k', 'clustering_k', 'section_name']
    available_sort = [col for col in sort_cols if col in df.columns]
    if available_sort:
        df = df.sort_values(available_sort)
    
    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    # Print summary
    print(f"\n=== AGGREGATION SUMMARY ===")
    print(f"Total files processed: {len(score_files)}")
    print(f"Successfully extracted: {len(df)}")
    print(f"Failed to extract: {failed_count}")
    
    # Report on percolation_score if present
    if 'percolation_score' in df.columns:
        print(f"Valid percolation scores: {df['percolation_score'].notna().sum()}")
        if df['percolation_score'].notna().any():
            print(f"Percolation score range: {df['percolation_score'].min():.6f} to "
                  f"{df['percolation_score'].max():.6f}")
            print(f"Mean percolation score: {df['percolation_score'].mean():.6f}")
    
    # Report on normalized_percolation_score if present
    if 'normalized_percolation_score' in df.columns:
        print(f"Valid normalized scores: {df['normalized_percolation_score'].notna().sum()}")
        if df['normalized_percolation_score'].notna().any():
            print(f"Normalized score range: {df['normalized_percolation_score'].min():.6f} to "
                  f"{df['normalized_percolation_score'].max():.6f}")
            print(f"Mean normalized score: {df['normalized_percolation_score'].mean():.6f}")
    
    print(f"\nSaved to: {output_csv}")
    
    return df


def reduce_scores_by_section(
    scores_csv: Union[str, Path],
    xy_dir: Union[str, Path],
    output_csv: Union[str, Path],
    section_column: str = 'section_name',
    use_normalized_score: bool = True,  # kept for backward compatibility; both scores are always computed
    clustering_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Reduce aggregated scores by computing weighted average across sections.
    
    This takes the per-section scores and computes a single score per
    algorithm/configuration weighted by the number of cells in each section.
    Both percolation_score and normalized_percolation_score are reduced whenever
    they are present in the input CSV.
    
    Args:
        scores_csv: Input CSV with per-section scores (from aggregate_score_files_to_csv)
        xy_dir: Directory containing {section}_XY.npy files to count cells
        output_csv: Output CSV file path
        section_column: Name of the column containing section names (default: 'section_name')
        use_normalized_score: Controls which score is used for sorting the output rows
                              (default: True → sort by normalized score weighted mean).
                              Both scores are always computed when available.
        clustering_dir: Directory containing step-3 clustering files.
                        If None, uses {scores_csv.parent}/clustering.
        
    Returns:
        DataFrame with reduced scores. Score columns are prefixed:
            raw_score_weighted_mean / raw_score_weighted_std
            raw_score_unweighted_mean / raw_score_unweighted_std
            normalized_score_weighted_mean / normalized_score_weighted_std
            normalized_score_unweighted_mean / normalized_score_unweighted_std
        
    Example:
        Input (per-section):
            algorithm | section | percolation_score | normalized_percolation_score
            Leiden_0  | sec1    | 3.5               | 0.72
            Leiden_0  | sec2    | 4.2               | 0.85
        
        Output (reduced):
            algorithm | raw_score_weighted_mean | normalized_score_weighted_mean | ...
            Leiden_0  | 3.85                    | 0.79                           | ...
    """
    scores_csv = Path(scores_csv)
    xy_dir = Path(xy_dir)
    output_csv = Path(output_csv)
    clustering_dir = Path(clustering_dir) if clustering_dir is not None else scores_csv.parent / "clustering"
    
    if not scores_csv.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv}")
    if not xy_dir.exists():
        raise FileNotFoundError(f"XY directory not found: {xy_dir}")
    
    # Load scores
    print(f"Loading scores from: {scores_csv}")
    df = pd.read_csv(scores_csv)
    
    if section_column not in df.columns:
        raise ValueError(f"Column '{section_column}' not found in {scores_csv}. "
                        f"Available columns: {list(df.columns)}")
    
    # Determine which score columns are available
    has_raw = 'percolation_score' in df.columns
    has_normalized = 'normalized_percolation_score' in df.columns
    
    if not has_raw and not has_normalized:
        raise ValueError(f"No valid score column found in {scores_csv}. "
                       f"Available columns: {list(df.columns)}")
    
    available = []
    if has_raw:
        available.append('percolation_score')
    if has_normalized:
        available.append('normalized_percolation_score')
    print(f"Score columns found: {available}")
    
    # Load cell counts from XY files
    print(f"Loading cell counts from: {xy_dir}")
    xy_files = list(xy_dir.glob("*_XY.npy"))
    
    if not xy_files:
        raise FileNotFoundError(f"No *_XY.npy files found in {xy_dir}")
    
    cell_counts = {}
    for xy_file in xy_files:
        section = xy_file.stem.replace('_XY', '')
        xy = np.load(xy_file)
        cell_counts[section] = len(xy)
    
    print(f"Found cell counts for {len(cell_counts)} sections")
    
    # Add cell counts to dataframe
    df['n_cells'] = df[section_column].map(cell_counts)
    
    # Check for missing cell counts
    missing = df['n_cells'].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows missing cell counts (sections not found in XY directory)")
        # Drop rows with missing cell counts
        df = df.dropna(subset=['n_cells'])

    # Load per-section type distributions from clustering outputs.
    # Later, each group's entropy is computed from pooled counts across sections.
    type_counts_by_relpath: Dict[str, Optional[Dict[Any, int]]] = {}
    if 'file_path' not in df.columns:
        print("WARNING: 'file_path' column missing; cannot load clustering type distributions")
    elif not clustering_dir.exists():
        print(f"WARNING: Clustering directory not found: {clustering_dir}")
    else:
        unique_paths = df['file_path'].dropna().unique()
        loaded_count = 0
        missing_count = 0
        failed_count = 0
        print(f"Loading type distributions from: {clustering_dir}")

        for rel_path in unique_paths:
            rel_path_str = str(rel_path)
            clustering_file = (clustering_dir / rel_path_str).with_suffix('.npy')

            if not clustering_file.exists():
                type_counts_by_relpath[rel_path_str] = None
                missing_count += 1
                continue

            try:
                type_vec = np.load(clustering_file)
                labels, counts = np.unique(type_vec, return_counts=True)
                count_map: Dict[Any, int] = {}
                for label, count in zip(labels, counts):
                    # Convert numpy scalar labels to python scalars for stable dict keys.
                    key = label.item() if hasattr(label, 'item') else label
                    count_map[key] = int(count)
                type_counts_by_relpath[rel_path_str] = count_map
                loaded_count += 1
            except Exception:
                type_counts_by_relpath[rel_path_str] = None
                failed_count += 1

        print(f"Loaded type distributions for {loaded_count}/{len(unique_paths)} sections")
        if missing_count > 0:
            print(f"WARNING: Missing {missing_count} clustering files")
        if failed_count > 0:
            print(f"WARNING: Failed to load {failed_count} clustering files")
    
    # Identify grouping columns (everything except section_column, score columns, n_cells, file_path)
    exclude_cols = {section_column, 'percolation_score', 'normalized_percolation_score', 'n_cells', 'file_path'}
    group_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Grouping by: {group_cols}")
    
    def _weighted_stats_for(group, col, prefix):
        """Compute weighted and unweighted stats for one score column."""
        weights = group['n_cells'].values
        scores = group[col].values
        # Drop NaN entries within the group
        valid = ~np.isnan(scores)
        if valid.sum() == 0:
            return {
                f'{prefix}_weighted_mean': np.nan,
                f'{prefix}_weighted_std': np.nan,
                f'{prefix}_unweighted_mean': np.nan,
                f'{prefix}_unweighted_std': np.nan,
            }
        w, s = weights[valid], scores[valid]
        weighted_mean = np.average(s, weights=w)
        weighted_std = np.sqrt(np.average((s - weighted_mean) ** 2, weights=w))
        return {
            f'{prefix}_weighted_mean': weighted_mean,
            f'{prefix}_weighted_std': weighted_std,
            f'{prefix}_unweighted_mean': s.mean(),
            f'{prefix}_unweighted_std': s.std(),
        }

    def _pooled_type_stats_for_group(group) -> Tuple[float, float]:
        """
        Compute pooled type-count and entropy for a group.

        This computes entropy over the concatenated type distribution across sections
        (implemented via pooled per-type counts), not an average of per-section entropies.
        """
        if not type_counts_by_relpath or 'file_path' not in group.columns:
            return np.nan, np.nan

        pooled_counts: Dict[Any, int] = {}
        for rel_path in group['file_path'].dropna().unique():
            count_map = type_counts_by_relpath.get(str(rel_path))
            if not count_map:
                continue
            for type_id, count in count_map.items():
                pooled_counts[type_id] = pooled_counts.get(type_id, 0) + count

        if not pooled_counts:
            return np.nan, np.nan

        counts = np.array(list(pooled_counts.values()), dtype=float)
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs)))
        return float(len(counts)), entropy
    
    # Compute weighted averages for all available score columns
    def weighted_stats(group):
        result = {}
        if has_raw:
            result.update(_weighted_stats_for(group, 'percolation_score', 'raw_score'))
        if has_normalized:
            result.update(_weighted_stats_for(group, 'normalized_percolation_score', 'normalized_score'))
        total_types, pooled_entropy = _pooled_type_stats_for_group(group)
        result['total_number_of_types'] = total_types
        result['type_distribution_entropy'] = pooled_entropy
        weights = group['n_cells'].values
        result['total_cells'] = weights.sum()
        result['n_sections'] = len(group)
        result['min_cells_per_section'] = weights.min()
        result['max_cells_per_section'] = weights.max()
        return pd.Series(result)
    
    # Group and aggregate
    reduced = df.groupby(group_cols, dropna=False).apply(weighted_stats).reset_index()
    
    # Sort by weighted mean of the primary score (descending)
    sort_col = ('normalized_score_weighted_mean' if (use_normalized_score and has_normalized)
                else 'raw_score_weighted_mean')
    if sort_col in reduced.columns:
        reduced = reduced.sort_values(sort_col, ascending=False)
    
    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_csv(output_csv, index=False)
    
    # Print summary
    print(f"\n=== REDUCTION SUMMARY ===")
    print(f"Input rows (per-section): {len(df)}")
    print(f"Output rows (reduced): {len(reduced)}")
    print(f"Grouped by: {', '.join(group_cols)}")
    print(f"Total cells across all sections: {df['n_cells'].sum():,}")
    
    if len(reduced) > 0 and sort_col in reduced.columns:
        std_col = sort_col.replace('_mean', '_std')
        print(f"\n{sort_col} range: {reduced[sort_col].min():.6f} to "
              f"{reduced[sort_col].max():.6f}")
        print(f"Best configuration:")
        best = reduced.iloc[0]
        for col in group_cols:
            if col in best.index:
                print(f"  {col}: {best[col]}")
        print(f"  {sort_col}: {best[sort_col]:.6f} ± {best[std_col]:.6f}")
        print(f"  Sections: {int(best['n_sections'])}, Total cells: {int(best['total_cells']):,}")
    
    print(f"\nSaved to: {output_csv}")
    
    return reduced
