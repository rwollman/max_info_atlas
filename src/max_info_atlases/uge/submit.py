"""
UGE job submission.

Generates and submits qsub scripts for array and single jobs.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

from .templates import generate_array_job_script, generate_single_job_script
from .job_generator import count_chunks


def submit_array_job(
    chunks_dir: Union[str, Path],
    base_dir: Union[str, Path],
    temp_dir: Union[str, Path],
    logs_dir: Union[str, Path],
    job_name: str = "percolation",
    memory: str = "16G",
    runtime: str = "4:00:00",
    conda_env: str = "max_info_atlases",
    worker_command: Optional[str] = None,
    dry_run: bool = False,
    hold_jid: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Submit an array job for chunked processing.
    
    Args:
        chunks_dir: Directory containing chunk files
        base_dir: Base directory for input data
        temp_dir: Directory for temporary outputs
        logs_dir: Directory for log files
        job_name: Name for the job
        memory: Memory request (e.g., "16G")
        runtime: Runtime request (e.g., "4:00:00")
        conda_env: Conda environment name
        worker_command: Custom worker command (default: max-info worker)
        dry_run: If True, don't actually submit
        hold_jid: Job ID to wait for before starting (UGE dependency)
        
    Returns:
        Tuple of (job_id, script_content) where job_id is None if dry_run
    """
    chunks_dir = Path(chunks_dir)
    
    # Count chunks
    num_chunks = count_chunks(chunks_dir)
    if num_chunks == 0:
        raise ValueError(f"No chunk files found in {chunks_dir}")
    
    print(f"Found {num_chunks} chunks")
    
    # Ensure directories exist
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate script
    script_content = generate_array_job_script(
        chunks_dir=str(chunks_dir),
        num_chunks=num_chunks,
        job_name=job_name,
        base_dir=str(base_dir),
        temp_dir=str(temp_dir),
        logs_dir=str(logs_dir),
        memory=memory,
        runtime=runtime,
        conda_env=conda_env,
        worker_command=worker_command,
    )
    
    if dry_run:
        print("Dry run - not submitting")
        print("\n=== Generated Script ===")
        print(script_content)
        return None, script_content
    
    # Write script to temp file and submit
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Build qsub command
        qsub_cmd = ['qsub']
        if hold_jid:
            qsub_cmd.extend(['-hold_jid', hold_jid])
        qsub_cmd.append(script_path)
        
        result = subprocess.run(
            qsub_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Parse job ID from output (format: "Your job-array XXXX.1-N:1 ...")
        output = result.stdout.strip()
        job_id = output.split()[2].split('.')[0]  # Get just the numeric ID
        
        print(f"Submitted array job: {job_id}")
        print(f"Tasks: 1-{num_chunks}")
        if hold_jid:
            print(f"Depends on job: {hold_jid}")
        print(f"Log pattern: {logs_dir}/{job_name}.$JOB_ID.$TASK_ID.log")
        
        return job_id, script_content
        
    finally:
        os.unlink(script_path)


def submit_single_job(
    chunk_file: Union[str, Path],
    chunk_id: str,
    base_dir: Union[str, Path],
    temp_dir: Union[str, Path],
    logs_dir: Union[str, Path],
    memory: str = "16G",
    runtime: str = "2:00:00",
    conda_env: str = "max_info_atlases",
    dry_run: bool = False,
) -> Tuple[Optional[str], str]:
    """
    Submit a single job (typically for resubmitting a failed chunk).
    
    Args:
        chunk_file: Path to the chunk file
        chunk_id: Identifier for this chunk
        base_dir: Base directory for input data
        temp_dir: Directory for temporary outputs
        logs_dir: Directory for log files
        memory: Memory request
        runtime: Runtime request
        conda_env: Conda environment name
        dry_run: If True, don't actually submit
        
    Returns:
        Tuple of (job_id, script_content) where job_id is None if dry_run
    """
    # Ensure directories exist
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate script
    script_content = generate_single_job_script(
        chunk_file=str(chunk_file),
        chunk_id=chunk_id,
        base_dir=str(base_dir),
        temp_dir=str(temp_dir),
        logs_dir=str(logs_dir),
        memory=memory,
        runtime=runtime,
        conda_env=conda_env,
    )
    
    if dry_run:
        print(f"Dry run - not submitting resubmit job for {chunk_id}")
        return None, script_content
    
    # Write script to temp file and submit
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        result = subprocess.run(
            ['qsub', script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Parse job ID
        output = result.stdout.strip()
        job_id = output.split()[2]
        
        print(f"Submitted resubmit job for chunk {chunk_id}: {job_id}")
        
        return job_id, script_content
        
    finally:
        os.unlink(script_path)


def check_qsub_available() -> bool:
    """Check if qsub is available on this system."""
    try:
        subprocess.run(['which', 'qsub'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
