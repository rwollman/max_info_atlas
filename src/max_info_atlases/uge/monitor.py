"""
UGE job monitoring with stuck detection and auto-resubmit.

Monitors running jobs, detects stuck tasks, and optionally resubmits them.
"""

import os
import re
import glob
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

from .submit import submit_single_job
from .job_generator import get_chunk_file


@dataclass
class TaskStatus:
    """Status information for a single task."""
    task_id: int
    state: str  # 'r' = running, 'qw' = queued, 'Eqw' = error
    start_time: Optional[datetime] = None
    output_file: Optional[Path] = None
    output_mtime: Optional[datetime] = None


class JobMonitor:
    """
    Monitor UGE jobs with stuck detection and auto-resubmit capability.
    """
    
    def __init__(
        self,
        job_id: str,
        chunks_dir: Union[str, Path],
        temp_dir: Union[str, Path],
        base_dir: Union[str, Path],
        logs_dir: Union[str, Path],
    ):
        """
        Initialize job monitor.
        
        Args:
            job_id: UGE job ID to monitor
            chunks_dir: Directory containing chunk files
            temp_dir: Directory where output CSVs are written
            base_dir: Base directory for input data
            logs_dir: Directory for log files
        """
        self.job_id = job_id
        self.chunks_dir = Path(chunks_dir)
        self.temp_dir = Path(temp_dir)
        self.base_dir = Path(base_dir)
        self.logs_dir = Path(logs_dir)
        
        # Track output file modification times
        self.last_output_mtime: Dict[int, datetime] = {}
        self.stuck_detected_at: Dict[int, datetime] = {}
        
        # Track resubmitted tasks
        self.resubmitted_jobs: Dict[int, str] = {}  # task_id -> new_job_id
    
    def parse_qstat(self) -> Dict[int, TaskStatus]:
        """
        Parse qstat output to get status of all tasks.
        
        Returns:
            Dictionary mapping task_id to TaskStatus
        """
        try:
            # Get username from environment or use default
            username = os.environ.get('USER', 'rwollman')
            
            result = subprocess.run(
                ['qstat', '-j', self.job_id],
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                # Job not found - all tasks completed or failed
                return {}
            
            # Parse output for task information
            # This is a simplified parser - may need adjustment for specific UGE versions
            tasks = {}
            
            # Use qstat -u $USERNAME to get task states for specific user
            result = subprocess.run(
                ['qstat', '-u', username],
                capture_output=True,
                text=True,
            )
            
            for line in result.stdout.strip().split('\n'):
                if self.job_id in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        # Parse job_id.task_id format or array job format
                        job_str = parts[0]
                        state = parts[4]
                        
                        # Extract task ID if present
                        if '.' in job_str:
                            # Format: job_id.task_id (individual task)
                            task_id = int(job_str.split('.')[1])
                            tasks[task_id] = TaskStatus(
                                task_id=task_id,
                                state=state,
                            )
                        else:
                            # Array job showing summary (no dot in job_str)
                            # Task ID is in the last column (ja-task-ID)
                            # Example: 12296902 0.00000 features rwollman qw 02/02/2026 19:03:59 1 1
                            # Note: empty columns (queue, jclass) don't appear when split, so we get ~9 fields
                            if len(parts) >= 8:
                                # The ja-task-ID field is the last column
                                task_id_str = parts[-1]
                                # Handle both single task IDs and ranges (e.g., "1" or "1-10")
                                if '-' in task_id_str:
                                    # Range of tasks
                                    start, end = task_id_str.split('-')
                                    for tid in range(int(start), int(end) + 1):
                                        tasks[tid] = TaskStatus(
                                            task_id=tid,
                                            state=state,
                                        )
                                else:
                                    # Single task
                                    try:
                                        task_id = int(task_id_str)
                                        tasks[task_id] = TaskStatus(
                                            task_id=task_id,
                                            state=state,
                                        )
                                    except ValueError:
                                        # Not a valid task ID, skip
                                        pass
            
            return tasks
            
        except Exception as e:
            print(f"Error parsing qstat: {e}")
            return {}
    
    def get_output_file(self, task_id: int) -> Optional[Path]:
        """
        Get the output CSV file for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Path to output file if exists, None otherwise
        """
        pattern = f"chunk_result_chunk_{task_id:04d}_*.csv"
        matches = list(self.temp_dir.glob(pattern))
        return matches[0] if matches else None
    
    def count_completed(self) -> Tuple[int, int]:
        """
        Count completed tasks based on output files.
        
        Returns:
            Tuple of (completed_count, total_count)
        """
        # Count chunk files for total
        total = len(list(self.chunks_dir.glob("chunk_*_of_*.txt")))
        
        # Count output files for completed
        completed = len(list(self.temp_dir.glob("chunk_result_*.csv")))
        
        return completed, total
    
    def detect_stuck_tasks(self, timeout_minutes: int = 60) -> List[int]:
        """
        Detect tasks that appear to be stuck.
        
        A task is considered stuck if:
        1. It's still running according to qstat AND
        2. Either:
           a) No output file exists and task has been running > timeout_minutes
           b) Output file exists but hasn't been modified in > timeout_minutes
        
        Args:
            timeout_minutes: Minutes without progress before marking as stuck
            
        Returns:
            List of stuck task IDs
        """
        stuck = []
        now = datetime.now()
        timeout = timedelta(minutes=timeout_minutes)
        
        # Get current task states
        tasks = self.parse_qstat()
        
        for task_id, status in tasks.items():
            if status.state != 'r':  # Only check running tasks
                continue
            
            output_file = self.get_output_file(task_id)
            
            if output_file is None:
                # No output yet
                if task_id not in self.stuck_detected_at:
                    # First time seeing this task with no output
                    self.stuck_detected_at[task_id] = now
                elif now - self.stuck_detected_at[task_id] > timeout:
                    stuck.append(task_id)
            else:
                # Has output - check if it's growing
                mtime = datetime.fromtimestamp(output_file.stat().st_mtime)
                
                if task_id in self.last_output_mtime:
                    if mtime == self.last_output_mtime[task_id]:
                        # Output hasn't changed
                        if task_id not in self.stuck_detected_at:
                            self.stuck_detected_at[task_id] = now
                        elif now - self.stuck_detected_at[task_id] > timeout:
                            stuck.append(task_id)
                    else:
                        # Output is growing - not stuck
                        self.stuck_detected_at.pop(task_id, None)
                
                self.last_output_mtime[task_id] = mtime
        
        return stuck
    
    def kill_task(self, task_id: int) -> bool:
        """
        Kill a specific task.
        
        Args:
            task_id: Task ID to kill
            
        Returns:
            True if successful
        """
        try:
            subprocess.run(
                ['qdel', f'{self.job_id}.{task_id}'],
                capture_output=True,
                check=True,
            )
            print(f"Killed task {task_id}")
            return True
        except Exception as e:
            print(f"Failed to kill task {task_id}: {e}")
            return False
    
    def resubmit_task(
        self,
        task_id: int,
        memory: str = "16G",
        runtime: str = "2:00:00",
        conda_env: str = "max_info_atlases",
    ) -> Optional[str]:
        """
        Resubmit a task as a single job.
        
        Args:
            task_id: Task ID to resubmit
            memory: Memory request for resubmit
            runtime: Runtime request for resubmit
            conda_env: Conda environment
            
        Returns:
            New job ID if successful, None otherwise
        """
        chunk_file = get_chunk_file(self.chunks_dir, task_id)
        if chunk_file is None:
            print(f"Could not find chunk file for task {task_id}")
            return None
        
        job_id, _ = submit_single_job(
            chunk_file=chunk_file,
            chunk_id=str(task_id),
            base_dir=self.base_dir,
            temp_dir=self.temp_dir,
            logs_dir=self.logs_dir,
            memory=memory,
            runtime=runtime,
            conda_env=conda_env,
        )
        
        if job_id:
            self.resubmitted_jobs[task_id] = job_id
        
        return job_id
    
    def kill_and_resubmit(
        self,
        task_ids: List[int],
        memory: str = "16G",
        runtime: str = "2:00:00",
        conda_env: str = "max_info_atlases",
    ) -> Dict[int, Optional[str]]:
        """
        Kill stuck tasks and resubmit them.
        
        Args:
            task_ids: List of task IDs to kill and resubmit
            memory: Memory request for resubmits
            runtime: Runtime request for resubmits
            conda_env: Conda environment
            
        Returns:
            Dictionary mapping task_id to new job_id (or None if failed)
        """
        results = {}
        
        for task_id in task_ids:
            print(f"\nProcessing stuck task {task_id}")
            
            # Kill the stuck task
            self.kill_task(task_id)
            
            # Resubmit
            new_job_id = self.resubmit_task(
                task_id=task_id,
                memory=memory,
                runtime=runtime,
                conda_env=conda_env,
            )
            
            results[task_id] = new_job_id
            
            # Clear stuck tracking
            self.stuck_detected_at.pop(task_id, None)
        
        return results
    
    def is_job_complete(self) -> bool:
        """Check if all tasks have completed."""
        tasks = self.parse_qstat()
        return len(tasks) == 0
    
    def monitor_loop(
        self,
        timeout_minutes: int = 60,
        poll_interval: int = 60,
        auto_resubmit: bool = False,
        extra_runtime: str = "2:00:00",
        on_complete: Optional[callable] = None,
    ) -> bool:
        """
        Main monitoring loop.
        
        Args:
            timeout_minutes: Minutes without progress to consider stuck
            poll_interval: Seconds between status checks
            auto_resubmit: If True, automatically kill and resubmit stuck tasks
            extra_runtime: Runtime for resubmitted jobs
            on_complete: Optional callback when all jobs complete
            
        Returns:
            True if all tasks completed successfully, False if some failed
        """
        print(f"Starting monitor for job {self.job_id}")
        print(f"Timeout: {timeout_minutes} minutes")
        print(f"Poll interval: {poll_interval} seconds")
        print(f"Auto-resubmit: {auto_resubmit}")
        print("")
        
        while True:
            completed, total = self.count_completed()
            tasks = self.parse_qstat()
            running = sum(1 for t in tasks.values() if t.state == 'r')
            queued = sum(1 for t in tasks.values() if t.state == 'qw')
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Completed: {completed}/{total}, "
                  f"Running: {running}, Queued: {queued}")
            
            # Check for stuck tasks
            stuck = self.detect_stuck_tasks(timeout_minutes)
            if stuck:
                print(f"  Stuck tasks detected: {stuck}")
                
                if auto_resubmit:
                    results = self.kill_and_resubmit(
                        stuck,
                        runtime=extra_runtime,
                    )
                    for task_id, new_job in results.items():
                        if new_job:
                            print(f"  Resubmitted task {task_id} as job {new_job}")
            
            # Check if complete (no more tasks in queue)
            if self.is_job_complete() and not self.resubmitted_jobs:
                # All jobs have left the queue - determine success/failure
                # Note: completed count may be 0 for non-percolation jobs that 
                # don't create chunk_result_*.csv files. In that case, we can't
                # determine success here - the calling script should verify outputs.
                if completed == total:
                    print(f"\n=== All tasks completed successfully ===")
                    print(f"Final count: {completed}/{total}")
                    if on_complete:
                        on_complete()
                    return True
                elif completed == 0:
                    # No completion markers found - might be a different job type
                    # that doesn't use CSV output files. Print warning but don't fail.
                    print(f"\n=== All tasks finished (job left queue) ===")
                    print(f"Final count: {completed}/{total}")
                    print(f"WARNING: No completion markers found. This may be normal for")
                    print(f"non-percolation jobs. Verify outputs manually or check logs at:")
                    print(f"  {self.logs_dir}")
                    if on_complete:
                        on_complete()
                    # Return True but caller should verify actual outputs
                    return True
                else:
                    # Some tasks completed, some failed
                    print(f"\n=== PARTIAL COMPLETION - SOME TASKS FAILED ===")
                    print(f"Final count: {completed}/{total} ({total - completed} failed)")
                    print(f"Check logs at: {self.logs_dir}")
                    return False
            
            # Wait before next check
            time.sleep(poll_interval)
