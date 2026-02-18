"""UGE/SGE job management for Hoffman2 HPC cluster."""

from .job_generator import (
    # Core functions
    prepare_jobs, 
    create_chunk_files,
    chunk_job_list_file,
    read_job_list,
    write_job_list,
    calculate_optimal_chunk_size,
    create_calibration_chunk,
    # Step-specific job list generators
    generate_features_jobs,
    generate_graph_jobs,
    generate_clustering_jobs,
    generate_percolation_jobs,
    generate_aggregation_jobs,
)
from .submit import submit_array_job, submit_single_job
from .monitor import JobMonitor

__all__ = [
    # Core functions
    'prepare_jobs', 
    'create_chunk_files',
    'chunk_job_list_file',
    'read_job_list',
    'write_job_list',
    'calculate_optimal_chunk_size',
    'create_calibration_chunk',
    # Step-specific generators
    'generate_features_jobs',
    'generate_graph_jobs',
    'generate_clustering_jobs',
    'generate_percolation_jobs',
    'generate_aggregation_jobs',
    # Submission and monitoring
    'submit_array_job', 
    'submit_single_job', 
    'JobMonitor',
]
