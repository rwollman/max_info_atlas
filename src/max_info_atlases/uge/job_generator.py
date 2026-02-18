"""
Job list and chunk file generation.

Creates job lists and splits them into chunks for parallel processing.

Job List Format:
    Each line in a job list file contains parameters for one unit of work.
    The format is step-specific (tab or space separated).
    
    Examples:
        # Features: input_file output_dir feature_type
        /data/test.h5ad\t/results/features\tpca50
        
        # Clustering: graph_file output_dir resolution_idx sections_file
        /results/FEL.npy\t/results/clusters\t25\t/data/Sections.npy
        
        # Percolation: type_file output_dir edge_list_dir max_k
        res_25/section1.npy\t/results/perc\t/results/edges\t100

Workflow:
    1. Generate job list: max-info jobs create-list --step <step> --params ...
    2. Chunk job list:    max-info jobs chunk --job-list jobs.txt --chunk-size 100
    3. Submit array job:  max-info jobs submit --worker-command "max-info run-<step> ..."
    4. Monitor:           max-info jobs monitor --job-id 12345
"""

import os
import math
import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union
from datetime import datetime


def read_job_list(job_list_file: Union[str, Path]) -> List[str]:
    """
    Read job list from a file.
    
    Args:
        job_list_file: Path to job list file (one job per line)
        
    Returns:
        List of job lines (stripped, non-empty, non-comment)
    """
    job_list_file = Path(job_list_file)
    jobs = []
    with open(job_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                jobs.append(line)
    return jobs


def write_job_list(jobs: List[str], output_file: Union[str, Path]) -> Path:
    """
    Write job list to a file.
    
    Args:
        jobs: List of job lines
        output_file: Path to output file
        
    Returns:
        Path to created file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for job in jobs:
            f.write(job + '\n')
    
    print(f"Created job list with {len(jobs)} jobs: {output_file}")
    return output_file


def scan_input_files(
    input_dir: Union[str, Path],
    pattern: str = "**/*.npy",
    output_dir: Optional[Union[str, Path]] = None,
    output_extension: str = ".npz",
) -> List[str]:
    """
    Scan for input files, optionally filtering out already-completed ones.
    
    Args:
        input_dir: Directory to scan for input files
        pattern: Glob pattern for input files
        output_dir: If provided, filter out files that have corresponding outputs
        output_extension: Extension of output files
        
    Returns:
        List of file paths (relative to input_dir)
    """
    input_dir = Path(input_dir)
    
    # Find all matching files
    all_files = list(input_dir.glob(pattern))
    
    if output_dir is None:
        return [str(f.relative_to(input_dir)) for f in all_files]
    
    # Filter out completed files
    output_dir = Path(output_dir)
    pending_files = []
    
    for input_file in all_files:
        rel_path = input_file.relative_to(input_dir)
        
        # Construct expected output path
        output_path = output_dir / rel_path.with_suffix(output_extension)
        
        if not output_path.exists():
            pending_files.append(str(rel_path))
    
    return pending_files


# =============================================================================
# Step-specific job list generators
# =============================================================================

def generate_features_jobs(
    input_files: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path],
    feature_types: Union[str, List[str]] = "pca50",
) -> List[str]:
    """
    Generate job list for feature extraction.
    
    Supports multiple input files and/or multiple feature types.
    Generates all combinations.
    
    Args:
        input_files: Input h5ad file(s) - single path or list
        output_dir: Output directory for features
        feature_types: Feature type(s) - single string or list (e.g., ['pca15', 'pca50'])
        
    Returns:
        List of job lines (tab-separated: input output type)
        
    Example:
        >>> generate_features_jobs('data.h5ad', 'out/', ['pca15', 'pca50'])
        ['data.h5ad\\tout/\\tpca15', 'data.h5ad\\tout/\\tpca50']
    """
    # Normalize to lists
    if isinstance(input_files, (str, Path)):
        input_files = [input_files]
    if isinstance(feature_types, str):
        feature_types = [feature_types]
    
    jobs = []
    for input_file in input_files:
        for feature_type in feature_types:
            jobs.append(f"{input_file}\t{output_dir}\t{feature_type}")
    return jobs


def generate_graph_jobs(
    input_files: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path],
    k_values: Union[int, List[int]] = 15,
    metrics: Union[str, List[str]] = "correlation",
) -> List[str]:
    """
    Generate job list for graph construction.
    
    Supports multiple input files, k values, and/or metrics.
    Generates all combinations.
    
    Args:
        input_files: Input features .npy file(s) - single path or list
        output_dir: Output directory for edge lists
        k_values: Number of neighbors - single int or list (e.g., [10, 15, 20])
        metrics: Distance metric(s) - single string or list (e.g., ['cosine', 'correlation'])
        
    Returns:
        List of job lines (tab-separated: input output_file k metric)
        
    Example:
        >>> generate_graph_jobs('feat.npy', 'out/', [10, 15], ['cosine', 'correlation'])
        # Returns 4 jobs (2 k values × 2 metrics)
    """
    # Normalize to lists
    if isinstance(input_files, (str, Path)):
        input_files = [input_files]
    if isinstance(k_values, int):
        k_values = [k_values]
    if isinstance(metrics, str):
        metrics = [metrics]
    
    jobs = []
    for input_file in input_files:
        input_path = Path(input_file)
        for k in k_values:
            for metric in metrics:
                # Generate output filename based on parameters
                output_file = Path(output_dir) / f"FEL_k{k}_{metric}.npy"
                jobs.append(f"{input_file}\t{output_file}\t{k}\t{metric}")
    return jobs


def generate_clustering_jobs(
    input_files: Union[str, Path, List[Union[str, Path]]],
    output_dir: Union[str, Path],
    sections_file: Union[str, Path],
    resolution_indices: Union[int, List[int]],
    method: str = 'leiden',
) -> List[str]:
    """
    Generate job list for clustering at multiple resolutions.
    
    Supports multiple input files (graphs) and/or multiple resolutions.
    Generates all combinations.
    
    Args:
        input_files: Input graph file(s) (FEL.npy) - single path or list
        output_dir: Output directory for cluster assignments
        sections_file: Sections .npy file
        resolution_indices: Resolution index or list of indices
        method: Clustering method ('leiden' or 'phenograph')
        
    Returns:
        List of job lines (tab-separated: method input output res_idx sections)
        
    Example:
        >>> generate_clustering_jobs('FEL.npy', 'out/', 'sec.npy', [0, 10, 25])
        # Returns 3 jobs (one per resolution)
        >>> generate_clustering_jobs('FEL.npy', 'out/', 'sec.npy', [0, 10], method='phenograph')
        # Returns 2 PhenoGraph jobs
    """
    # Normalize to lists
    if isinstance(input_files, (str, Path)):
        input_files = [input_files]
    if isinstance(resolution_indices, int):
        resolution_indices = [resolution_indices]
    
    jobs = []
    for input_file in input_files:
        for res_idx in resolution_indices:
            jobs.append(f"{method}\t{input_file}\t{output_dir}\t{res_idx}\t{sections_file}")
    return jobs


def generate_percolation_jobs(
    type_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    edge_list_dir: Union[str, Path],
    pattern: str = "res_*/*.npy",
    max_k: int = 100,
    xy_data_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Generate job list for percolation analysis.
    
    Args:
        type_data_dir: Directory containing type vector files
        output_dir: Output directory for percolation results
        edge_list_dir: Directory for edge list cache
        pattern: Glob pattern for type vector files
        max_k: Maximum k for percolation
        xy_data_dir: Directory with XY coordinate files ({section_name}_XY.npy format)
        
    Returns:
        List of job lines (tab-separated: type_file output_dir edge_dir max_k xy_dir type_data_dir)
    """
    type_data_dir = Path(type_data_dir)
    type_files = sorted(type_data_dir.glob(pattern))
    
    jobs = []
    xy_dir = xy_data_dir or type_data_dir
    
    for type_file in type_files:
        rel_path = type_file.relative_to(type_data_dir)
        jobs.append(f"{rel_path}\t{output_dir}\t{edge_list_dir}\t{max_k}\t{xy_dir}\t{type_data_dir}")
    
    return jobs


def generate_aggregation_jobs(
    results_dir: Union[str, Path],
    output_file: Union[str, Path],
    pattern: str = "**/*.score",
) -> List[str]:
    """
    Generate job list for score aggregation.
    
    Args:
        results_dir: Directory containing .score files
        output_file: Output CSV file
        pattern: Glob pattern for score files
        
    Returns:
        List of job lines (tab-separated: results_dir output_file pattern)
    """
    return [f"{results_dir}\t{output_file}\t{pattern}"]


def create_chunk_files(
    job_list: List[str],
    output_dir: Union[str, Path],
    chunk_size: int = 6000,
) -> Tuple[List[Path], Path]:
    """
    Split a job list into chunk files.
    
    If chunk_size >= len(job_list), creates a single chunk (single job).
    
    Args:
        job_list: List of job entries (file paths or commands)
        output_dir: Directory to save chunk files
        chunk_size: Number of jobs per chunk (use large number for single job)
        
    Returns:
        Tuple of (list of chunk file paths, summary file path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_jobs = len(job_list)
    
    # If chunk_size >= total, make it a single chunk
    if chunk_size >= total_jobs:
        chunk_size = total_jobs
    
    num_chunks = math.ceil(total_jobs / chunk_size)
    
    print(f"Creating {num_chunks} chunk(s) from {total_jobs} jobs")
    print(f"Jobs per chunk: {chunk_size}")
    print(f"Output directory: {output_dir}")
    
    chunk_files = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_jobs)
        
        chunk_jobs = job_list[start_idx:end_idx]
        
        # Use 4-digit zero-padded naming
        chunk_filename = f"chunk_{i+1:04d}_of_{num_chunks:04d}.txt"
        chunk_path = output_dir / chunk_filename
        
        with open(chunk_path, 'w') as f:
            for job in chunk_jobs:
                f.write(job + '\n')
        
        chunk_files.append(chunk_path)
        print(f"Created {chunk_filename} ({len(chunk_jobs)} jobs)")
    
    # Create summary file
    summary_path = output_dir / "chunks_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Total jobs: {total_jobs}\n")
        f.write(f"Jobs per chunk: {chunk_size}\n")
        f.write(f"Number of chunks: {num_chunks}\n")
        f.write(f"Created: {datetime.now().isoformat()}\n")
        f.write("\nChunk files:\n")
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_jobs)
            chunk_filename = f"chunk_{i+1:04d}_of_{num_chunks:04d}.txt"
            f.write(f"{chunk_filename}: jobs {start_idx+1}-{end_idx}\n")
    
    print(f"\nCreated summary: {summary_path}")
    
    return chunk_files, summary_path


def chunk_job_list_file(
    job_list_file: Union[str, Path],
    chunks_dir: Union[str, Path],
    chunk_size: int = 6000,
) -> Tuple[int, Path]:
    """
    Read a job list file and create chunks.
    
    This is the main entry point for chunking an existing job list.
    
    Args:
        job_list_file: Path to job list file
        chunks_dir: Directory to save chunk files
        chunk_size: Number of jobs per chunk (use large number for single job)
        
    Returns:
        Tuple of (number of chunks, chunks directory path)
    """
    job_list = read_job_list(job_list_file)
    
    if not job_list:
        print("No jobs to process!")
        return 0, Path(chunks_dir)
    
    chunk_files, _ = create_chunk_files(
        job_list=job_list,
        output_dir=chunks_dir,
        chunk_size=chunk_size,
    )
    
    return len(chunk_files), Path(chunks_dir)


def prepare_jobs(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    chunks_dir: Union[str, Path],
    chunk_size: int = 6000,
    pattern: str = "**/*.npy",
) -> Tuple[int, Path]:
    """
    Prepare job list and create chunk files.
    
    This is the main entry point for job preparation.
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory for outputs (used to filter completed jobs)
        chunks_dir: Directory to save chunk files
        chunk_size: Number of jobs per chunk
        pattern: Glob pattern for input files
        
    Returns:
        Tuple of (number of chunks, chunks directory path)
    """
    print(f"Scanning for jobs in: {input_dir}")
    
    # Scan for pending jobs
    job_list = scan_input_files(
        input_dir=input_dir,
        pattern=pattern,
        output_dir=output_dir,
    )
    
    print(f"Found {len(job_list)} pending jobs")
    
    if not job_list:
        print("No jobs to process!")
        return 0, Path(chunks_dir)
    
    # Create chunks
    chunk_files, _ = create_chunk_files(
        job_list=job_list,
        output_dir=chunks_dir,
        chunk_size=chunk_size,
    )
    
    return len(chunk_files), Path(chunks_dir)


def count_chunks(chunks_dir: Union[str, Path]) -> int:
    """
    Count the number of chunk files in a directory.
    
    Args:
        chunks_dir: Directory containing chunk files
        
    Returns:
        Number of chunk files
    """
    chunks_dir = Path(chunks_dir)
    chunk_files = list(chunks_dir.glob("chunk_*_of_*.txt"))
    return len(chunk_files)


def get_chunk_file(chunks_dir: Union[str, Path], task_id: int) -> Optional[Path]:
    """
    Get the chunk file path for a given task ID.
    
    Args:
        chunks_dir: Directory containing chunk files
        task_id: Task ID (1-indexed)
        
    Returns:
        Path to chunk file, or None if not found
    """
    chunks_dir = Path(chunks_dir)
    pattern = f"chunk_{task_id:04d}_of_*.txt"
    matches = list(chunks_dir.glob(pattern))
    
    return matches[0] if matches else None


def calculate_optimal_chunk_size(
    time_per_file: float,
    target_runtime_hours: float = 1.0,
    safety_factor: float = 0.8,
    min_chunk_size: int = 100,
    max_chunk_size: int = 50000,
) -> int:
    """
    Calculate optimal chunk size based on measured time per file.
    
    Args:
        time_per_file: Measured seconds per file from calibration run
        target_runtime_hours: Target runtime for each chunk (default: 1 hour)
        safety_factor: Safety margin (0.8 = 80% of theoretical max)
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
        
    Returns:
        Recommended chunk size
        
    Example:
        >>> # If each file takes 0.1 seconds
        >>> calculate_optimal_chunk_size(time_per_file=0.1, target_runtime_hours=1.0)
        28800  # ~8 hours / 0.1s = 28800 files, with 80% safety = 23040
    """
    target_runtime_seconds = target_runtime_hours * 3600
    
    # Calculate theoretical max
    theoretical_max = target_runtime_seconds / time_per_file
    
    # Apply safety factor
    recommended = int(theoretical_max * safety_factor)
    
    # Clamp to min/max
    recommended = max(min_chunk_size, min(max_chunk_size, recommended))
    
    return recommended


def create_calibration_chunk(
    job_list: List[str],
    output_dir: Union[str, Path],
    n_files: int = 10,
) -> Path:
    """
    Create a small calibration chunk for timing estimation.
    
    Args:
        job_list: Full list of jobs
        output_dir: Directory to save calibration chunk
        n_files: Number of files to include (default: 10)
        
    Returns:
        Path to calibration chunk file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample files from different parts of the list
    n_files = min(n_files, len(job_list))
    step = max(1, len(job_list) // n_files)
    sampled = [job_list[i * step] for i in range(n_files)]
    
    calibration_file = output_dir / "calibration_chunk.txt"
    with open(calibration_file, 'w') as f:
        for job in sampled:
            f.write(job + '\n')
    
    print(f"Created calibration chunk with {len(sampled)} files: {calibration_file}")
    
    return calibration_file


def parse_calibration_results(log_file: Union[str, Path]) -> dict:
    """
    Parse calibration job log to extract timing information.
    
    Args:
        log_file: Path to calibration job log file
        
    Returns:
        Dictionary with timing statistics
    """
    import re
    
    log_file = Path(log_file)
    content = log_file.read_text()
    
    # Extract timing from CSV output or log messages
    times = []
    
    # Look for total_time_seconds in output
    for match in re.finditer(r'total_time_seconds["\s:]+(\d+\.?\d*)', content):
        times.append(float(match.group(1)))
    
    if not times:
        # Try alternative patterns
        for match in re.finditer(r'Processed .* in (\d+\.?\d*) seconds', content):
            times.append(float(match.group(1)))
    
    if times:
        return {
            'n_files': len(times),
            'total_time': sum(times),
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
        }
    
    return {'error': 'Could not parse timing from log'}
