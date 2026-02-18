"""
Timing analysis utilities for HPC job logs.

This module provides functions to analyze UGE log files and estimate
per-chunk task durations, helping optimize chunk sizes for different pipeline steps.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np


def parse_log_file(log_path: Path) -> Optional[Dict]:
    """
    Parse a UGE log file to extract timing and task information.
    
    Args:
        log_path: Path to log file
        
    Returns:
        Dict with timing info, or None if log couldn't be parsed:
            {
                'start_time': datetime,
                'end_time': datetime or None,
                'duration_seconds': float or None,
                'chunk_path': str,
                'n_tasks_in_chunk': int,
                'task_id': int,
                'job_id': str,
            }
    """
    if not log_path.exists():
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    info = {}
    
    # Extract start time
    # Format: "Start time: Sun Feb  8 21:06:05 PST 2026"
    start_match = re.search(r'Start time: (.+)', content)
    if start_match:
        time_str = start_match.group(1)
        try:
            # Parse the datetime (this might need adjustment based on actual format)
            info['start_time'] = datetime.strptime(time_str, '%a %b %d %H:%M:%S %Z %Y')
        except ValueError:
            # Fallback: use file creation time
            info['start_time'] = datetime.fromtimestamp(log_path.stat().st_ctime)
    else:
        info['start_time'] = datetime.fromtimestamp(log_path.stat().st_ctime)
    
    # Extract end time (if present)
    # Format: "End time: ..." or use file modification time
    end_match = re.search(r'End time: (.+)', content)
    if end_match:
        time_str = end_match.group(1)
        try:
            info['end_time'] = datetime.strptime(time_str, '%a %b %d %H:%M:%S %Z %Y')
        except ValueError:
            info['end_time'] = datetime.fromtimestamp(log_path.stat().st_mtime)
    else:
        # Only use file mtime if job has actually completed
        # Look for completion indicators from the UGE template and worker scripts
        if (re.search(r'Done SGE_TASK_ID', content) or 
            re.search(r'Completed \d+ .* jobs', content)):
            info['end_time'] = datetime.fromtimestamp(log_path.stat().st_mtime)
        else:
            info['end_time'] = None
    
    # Calculate duration
    if info['end_time']:
        info['duration_seconds'] = (info['end_time'] - info['start_time']).total_seconds()
    else:
        info['duration_seconds'] = None
    
    # Extract chunk path
    chunk_match = re.search(r'Processing chunk: (.+)', content)
    if chunk_match:
        info['chunk_path'] = chunk_match.group(1)
    else:
        info['chunk_path'] = None
    
    # Extract number of tasks in chunk
    # Format: "Processing 1 clustering jobs from chunk" or "Processing 10 feature extraction jobs"
    tasks_match = re.search(r'Processing (\d+) .+ jobs from chunk', content)
    if tasks_match:
        info['n_tasks_in_chunk'] = int(tasks_match.group(1))
    else:
        info['n_tasks_in_chunk'] = 1  # Default to 1 if not found
    
    # Extract task ID from log filename
    # Format: {run_name}_{step}.{job_id}.{task_id}.log
    filename = log_path.name
    task_match = re.search(r'\.(\d+)\.(\d+)\.log$', filename)
    if task_match:
        info['job_id'] = task_match.group(1)
        info['task_id'] = int(task_match.group(2))
    else:
        info['job_id'] = 'unknown'
        info['task_id'] = 0
    
    return info


def analyze_step_logs(log_dir: Path, step_name: str, chunk_size: int) -> Dict:
    """
    Analyze all log files for a given step.
    
    Args:
        log_dir: Directory containing log files for this step
        step_name: Name of the step (e.g., 'features', 'clustering')
        chunk_size: Configured chunk size for this step (tasks per job)
        
    Returns:
        Dict with timing statistics:
            {
                'n_logs': int,
                'n_completed': int,
                'n_running': int,
                'avg_duration_seconds': float,
                'min_duration_seconds': float,
                'max_duration_seconds': float,
                'estimated_per_task_seconds': float,
                'chunk_size': int,
                'durations': List[float],
            }
    """
    if not log_dir.exists():
        return {
            'n_logs': 0,
            'n_completed': 0,
            'n_running': 0,
            'avg_duration_seconds': None,
            'min_duration_seconds': None,
            'max_duration_seconds': None,
            'estimated_per_task_seconds': None,
            'chunk_size': chunk_size,
            'durations': [],
        }
    
    log_files = list(log_dir.glob('*.log'))
    
    completed_durations = []
    n_running = 0
    
    for log_file in log_files:
        info = parse_log_file(log_file)
        if info and info['duration_seconds'] is not None:
            completed_durations.append(info['duration_seconds'])
        elif info and info['end_time'] is None:
            n_running += 1
    
    n_completed = len(completed_durations)
    
    if completed_durations:
        avg_duration = np.mean(completed_durations)
        min_duration = np.min(completed_durations)
        max_duration = np.max(completed_durations)
        
        # Estimate per-task time by dividing average job time by chunk_size
        estimated_per_task = avg_duration / chunk_size if chunk_size > 0 else avg_duration
    else:
        avg_duration = None
        min_duration = None
        max_duration = None
        estimated_per_task = None
    
    return {
        'n_logs': len(log_files),
        'n_completed': n_completed,
        'n_running': n_running,
        'avg_duration_seconds': avg_duration,
        'min_duration_seconds': min_duration,
        'max_duration_seconds': max_duration,
        'estimated_per_task_seconds': estimated_per_task,
        'chunk_size': chunk_size,
        'durations': completed_durations,
    }


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds is None:
        return 'N/A'
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def analyze_log_timings(config, step_name: str):
    """
    Analyze log timings for a specific step and print summary.
    
    Args:
        config: RunConfig object
        step_name: Step to analyze ('features', 'graphs', etc.)
    """
    from ..run import RunConfig
    
    log_dir = Path(config.logs_dir) / step_name
    
    # Get chunk_size from config
    step_config = config._data['steps'].get(step_name, {})
    resources = step_config.get('resources', {})
    chunk_size = resources.get('chunk_size', 1)
    
    print(f"=== {step_name.upper()} ===")
    print(f"Log directory: {log_dir}")
    
    if not log_dir.exists():
        print(f"  No logs found (directory doesn't exist)\n")
        return
    
    stats = analyze_step_logs(log_dir, step_name, chunk_size)
    
    print(f"Total log files: {stats['n_logs']}")
    print(f"Completed jobs: {stats['n_completed']}")
    print(f"Running jobs: {stats['n_running']}")
    print(f"Chunk size (tasks/job): {stats['chunk_size']}")
    print()
    
    if stats['avg_duration_seconds'] is not None:
        print(f"Job Duration Statistics:")
        print(f"  Average: {format_duration(stats['avg_duration_seconds'])}")
        print(f"  Min:     {format_duration(stats['min_duration_seconds'])}")
        print(f"  Max:     {format_duration(stats['max_duration_seconds'])}")
        print()
        
        if stats['estimated_per_task_seconds'] is not None:
            print(f"Estimated Per-Task Duration (atomic):")
            print(f"  {format_duration(stats['estimated_per_task_seconds'])} per task")
            print(f"  (calculated as avg_job_time / chunk_size)")
    else:
        print("No completed jobs to analyze")
    
    print()
