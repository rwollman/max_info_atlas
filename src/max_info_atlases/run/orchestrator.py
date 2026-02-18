"""
Run orchestrator - generates job lists for all pipeline steps from a RunConfig.

This reads the run config YAML and produces:
  - Job list files for each step
  - Chunk files for HPC array jobs
  - A manifest of expected outputs

Usage:
    orchestrator = RunOrchestrator(config)
    orchestrator.prepare_all()     # Generate all job lists + chunks
    orchestrator.submit_all()      # Submit to UGE with dependencies
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .manifest import RunConfig, RunManifest
from ..uge.job_generator import write_job_list, create_chunk_files
from ..features.graphs import build_folder_name
from ..clustering.base import get_resolution_values, format_resolution_dirname


class RunOrchestrator:
    """
    Generates job lists and manages submission for a full pipeline run.
    
    Given a RunConfig, this class knows how to:
    1. Generate job lists for every step
    2. Chunk them for HPC array jobs
    3. Submit them with proper inter-step dependencies
    4. Report on what was generated
    
    Example:
        config = RunConfig.from_yaml('config/run_cell_type_opt.yaml')
        orch = RunOrchestrator(config)
        orch.prepare_all()
    """
    
    def __init__(self, config: RunConfig, sections: Optional[List[str]] = None, skip_completed: bool = True):
        """
        Initialize orchestrator.
        
        Args:
            config: Run configuration
            sections: Section names (loaded from config if not provided)
            skip_completed: If True, skip jobs for already-existing outputs (default: True)
        """
        self.config = config
        self.manifest = RunManifest(config, sections=sections)
        self._sections = self.manifest.sections
        self.skip_completed = skip_completed
    
    # =========================================================================
    # Job list generators for each step
    # =========================================================================
    
    def generate_features_jobs(self) -> List[str]:
        """
        Generate job list for feature extraction.
        
        Supports three types of features:
        1. Expression features (raw, log1p, pca, scvi) - from adata.X
        2. Local frequency features (localfreq_k{k}) - from spatial neighborhoods
        3. Derived features (localfreq_pca{n}_k{k}) - from other feature matrices
        
        Jobs are ordered to ensure dependencies are satisfied:
        - Expression and local frequency features first (independent)
        - Derived features second (depend on their source features)
        
        Format varies by feature type:
        - Expression: adata_path\\toutput_dir\\tfeature_type
        - Local frequency: adata_path\\toutput_dir\\tfeature_type\\txy_dir\\tsections_file\\ttype_column
        - Derived: operation\\tsource_feature_path\\toutput_dir\\tderived_name\\tparams_json
        """
        config = self.config
        
        # Get feature dependencies
        dependencies = config.feature_dependencies
        
        # Separate features into base (no dependencies) and derived (have dependencies)
        base_jobs = []
        derived_jobs = []
        
        for dt in config.feature_data_types:
            # Check if output already exists
            output_file = config.features_dir / f"features_{dt}.npy"
            if self.skip_completed and output_file.exists():
                print(f"  Skipping {dt} (already exists: {output_file})")
                continue
            
            # Check if this is a derived feature
            if dt in dependencies:
                # Derived feature - schedule after its source
                source_dt = dependencies[dt]
                source_file = config.features_dir / f"features_{source_dt}.npy"
                
                # Parse operation and params from derived config
                derived_config = config.derived_features_config
                if derived_config:
                    from ..features.derived import parse_derived_config
                    parsed_specs = parse_derived_config(derived_config)
                    
                    # Find the spec that matches this derived feature
                    import json
                    for spec in parsed_specs:
                        operation = spec.get('operation')
                        # Check if this spec would generate this derived name
                        from ..features.derived import build_derived_feature_name
                        
                        # Try to match by checking if source matches and operation matches
                        test_name = build_derived_feature_name(source_dt, operation, spec)
                        if test_name == dt:
                            params_json = json.dumps(spec)
                            derived_jobs.append(
                                f"derived\t{source_file}\t{config.features_dir}\t{dt}\t{params_json}"
                            )
                            break
            
            elif dt.startswith('localfreq_'):
                # Local frequency feature
                # Extract k and weighted from name
                # localfreq_k50 or localfreq_w_k50
                lf_config = config.local_frequency_config
                if lf_config:
                    type_column = lf_config.get('type_column')
                    
                    base_jobs.append(
                        f"localfreq\t{config.adata_path}\t{config.features_dir}\t{dt}\t"
                        f"{config.xy_dir}\t{config.sections_file}\t{type_column}"
                    )
            
            else:
                # Expression feature (raw, log1p, pca, scvi)
                base_jobs.append(
                    f"{config.adata_path}\t{config.features_dir}\t{dt}"
                )
        
        # Combine: base features first, then derived features
        # This ensures sources exist before derived features are computed
        jobs = base_jobs + derived_jobs
        
        return jobs
    
    def extract_preexisting_annotations(self):
        """
        Extract preexisting annotations from AnnData .obs columns
        and save as per-section .npy type vector files.
        
        These files go into {clustering_dir}/Preexisting_{level}/{section}.npy
        so the percolation step can process them like any other clustering.
        
        This step requires loading the AnnData file, so it runs locally
        (not as a UGE job).
        """
        config = self.config
        preexisting_columns = config.preexisting_obs_columns
        
        if not preexisting_columns:
            print("  No preexisting annotation columns configured, skipping.")
            return
        
        # Check if already extracted
        all_exist = True
        for level in preexisting_columns:
            level_dir = config.clustering_dir / f"Preexisting_{level}"
            if not level_dir.exists():
                all_exist = False
                break
            # Check if it has the right number of section files
            npy_files = list(level_dir.glob("*.npy"))
            if len(npy_files) < len(self._sections):
                all_exist = False
                break
        
        if all_exist:
            print("  Preexisting annotations already extracted, skipping.")
            return
        
        print(f"  Loading AnnData from {config.adata_path} ...")
        print(f"    Warning: This loads the full .X matrix. For large files, submit as HPC job.")
        import scanpy as sc
        
        try:
            adata = sc.read_h5ad(config.adata_path)
        except (MemoryError, Exception) as e:
            if 'memory' in str(e).lower() or 'allocate' in str(e).lower():
                print(f"\n  ERROR: Not enough memory to load {config.adata_path}")
                print(f"  Submit this as an HPC job with more memory allocation.")
            raise
        
        from ..features.celltype import extract_preexisting_annotations
        
        # Determine how to get section labels
        if config.section_column:
            print(f"  Using section column '{config.section_column}' from adata.obs")
            section_column = config.section_column
            sections_array = None
        else:
            print(f"  Loading sections from {config.sections_file} ...")
            sections_array = np.load(config.sections_file)
            section_column = None
        
        print(f"  Extracting preexisting annotations from .obs columns: {preexisting_columns}")
        summary = extract_preexisting_annotations(
            adata,
            output_dir=config.clustering_dir,
            obs_columns=preexisting_columns,
            section_column=section_column,
            sections_array=sections_array,
        )
        
        print(f"  Extracted {len(summary)} preexisting annotation levels")
        return summary
    
    def generate_graph_jobs(self) -> List[str]:
        """
        Generate job list for graph construction.
        
        Format: input_file\\toutput_file\\tk\\tmetric
        """
        config = self.config
        jobs = []
        
        for dt in config.feature_data_types:
            for dist in config.graph_distances:
                input_file = config.features_dir / f"features_{dt}.npy"
                output_file = config.graphs_dir / f"FEL_{dt}_{dist}.npy"
                
                # Check if output already exists
                if self.skip_completed and output_file.exists():
                    print(f"  Skipping {dt}/{dist} (already exists: {output_file})")
                    continue
                    
                jobs.append(
                    f"{input_file}\t{output_file}\t{config.graph_k}\t{dist}"
                )
        
        return jobs
    
    def generate_clustering_jobs(self) -> List[str]:
        """
        Generate job list for clustering.
        
        Format: method\\tinput_file\\toutput_dir\\tres_idx\\tsections_file
        
        Note: the 'method' field is new (prepended) to distinguish
        Leiden vs PhenoGraph in the clustering worker.
        """
        config = self.config
        jobs = []
        skipped = 0
        
        for method in config.methods:
            if method.name not in ('leiden', 'phenograph'):
                continue  # Preexisting has no clustering step
            
            for dt in method.data_types:
                for dist in method.distances:
                    # Input graph file
                    graph_file = config.graphs_dir / f"FEL_{dt}_{dist}.npy"
                    
                    # Output folder name
                    method_str = 'Leiden' if method.name == 'leiden' else 'PhenoGraph'
                    folder = build_folder_name(method_str, dt, dist)
                    output_dir = config.clustering_dir / folder
                    
                    # Get actual resolution values for this method
                    resolution_values = get_resolution_values(method.n_resolutions)
                    
                    for res_idx in range(method.n_resolutions):
                        resolution = resolution_values[res_idx]
                        res_dirname = format_resolution_dirname(resolution)
                        
                        # Check if all section outputs already exist
                        if self.skip_completed:
                            res_dir = output_dir / res_dirname
                            all_exist = True
                            if res_dir.exists():
                                for section in self._sections:
                                    section_file = res_dir / f"{section}.npy"
                                    if not section_file.exists():
                                        all_exist = False
                                        break
                            else:
                                all_exist = False
                            
                            if all_exist:
                                skipped += 1
                                continue
                        
                        jobs.append(
                            f"{method.name}\t{graph_file}\t{output_dir}\t{res_idx}\t{config.sections_file}"
                        )
        
        if skipped > 0:
            print(f"  Skipped {skipped} completed clustering jobs")
        
        return jobs
    
    def generate_percolation_jobs(self) -> List[str]:
        """
        Generate job list for percolation analysis.
        
        Format: type_file_rel\\toutput_dir\\tedge_list_dir\\tmax_k\\txy_dir\\ttype_data_dir
        
        This generates one job per type_vector file (method x data_type x distance x resolution x section).
        """
        config = self.config
        jobs = []
        skipped = 0
        
        for method in config.methods:
            if method.name in ('leiden', 'phenograph'):
                for dt in method.data_types:
                    for dist in method.distances:
                        method_str = 'Leiden' if method.name == 'leiden' else 'PhenoGraph'
                        folder = build_folder_name(method_str, dt, dist)
                        
                        # Get actual resolution values for this method
                        resolution_values = get_resolution_values(method.n_resolutions)
                        
                        for res_idx in range(method.n_resolutions):
                            resolution = resolution_values[res_idx]
                            res_dirname = format_resolution_dirname(resolution)
                            
                            for section in self._sections:
                                # Relative path within clustering_dir
                                rel_path = f"{folder}/{res_dirname}/{section}.npy"
                                
                                # Check if score file already exists
                                if self.skip_completed:
                                    score_file = config.percolation_dir / rel_path.replace('.npy', '.score')
                                    if score_file.exists():
                                        skipped += 1
                                        continue
                                
                                jobs.append(
                                    f"{rel_path}\t{config.percolation_dir}\t"
                                    f"{config.edge_lists_dir}\t{config.percolation_max_k}\t"
                                    f"{config.xy_dir}\t{config.clustering_dir}"
                                )
            
            elif method.name == 'preexisting':
                preexisting_dir = config.preexisting_dir
                if preexisting_dir is None:
                    continue
                
                for level in method.levels:
                    for section in self._sections:
                        rel_path = f"Preexisting_{level}/{section}.npy"
                        
                        # Check if score file already exists
                        if self.skip_completed:
                            score_file = config.percolation_dir / rel_path.replace('.npy', '.score')
                            if score_file.exists():
                                skipped += 1
                                continue
                        
                        jobs.append(
                            f"{rel_path}\t{config.percolation_dir}\t"
                            f"{config.edge_lists_dir}\t{config.percolation_max_k}\t"
                            f"{config.xy_dir}\t{preexisting_dir}"
                        )
        
        if skipped > 0:
            print(f"  Skipped {skipped} completed percolation jobs")
        
        return jobs
    
    def generate_aggregation_jobs(self) -> List[str]:
        """
        Generate job list for score aggregation (and optional reduction).
        
        Format: results_dir\\toutput_file\\tpattern\\txy_dir\\treduced_output
        """
        config = self.config
        return [
            f"{config.percolation_dir}\t{config.scores_csv}\t**/*.score\t{config.xy_dir}\t{config.reduced_scores_csv}"
        ]
    
    # =========================================================================
    # Prepare: generate all job lists and chunks
    # =========================================================================
    
    def prepare_step(self, step: str) -> Tuple[Path, Path, int]:
        """
        Prepare job list and chunks for a single step.
        
        Args:
            step: Pipeline step name ('features', 'graphs', 'clustering', 'percolation', 'aggregation')
        
        Returns:
            Tuple of (job_list_path, chunks_dir, num_chunks)
        """
        config = self.config
        
        # Generate jobs
        generators = {
            'features': self.generate_features_jobs,
            'graphs': self.generate_graph_jobs,
            'clustering': self.generate_clustering_jobs,
            'percolation': self.generate_percolation_jobs,
            'aggregation': self.generate_aggregation_jobs,
        }
        
        if step not in generators:
            raise ValueError(f"Unknown step: {step}. Must be one of {list(generators.keys())}")
        
        jobs = generators[step]()
        
        if not jobs:
            print(f"  No jobs for step '{step}'")
            # Clean up stale chunk files so submit won't find old chunks
            step_dir = config.jobs_dir / step
            chunks_dir = step_dir / "chunks"
            if chunks_dir.exists():
                import shutil
                shutil.rmtree(chunks_dir)
            return None, None, 0
        
        # Create step directory
        step_dir = config.jobs_dir / step
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # Write job list
        job_list_path = step_dir / "jobs.txt"
        write_job_list(jobs, job_list_path)
        
        # Create chunks
        resources = config.hpc_resources(step)
        chunk_size = resources.get('chunk_size', 6000)
        
        chunks_dir = step_dir / "chunks"
        chunk_files, summary_path = create_chunk_files(
            job_list=jobs,
            output_dir=chunks_dir,
            chunk_size=chunk_size,
        )
        
        return job_list_path, chunks_dir, len(chunk_files)
    
    def prepare_all(self) -> Dict[str, Dict]:
        """
        Prepare job lists and chunks for ALL pipeline steps.
        
        Returns:
            Dict mapping step name to preparation info.
        """
        config = self.config
        steps = ['features', 'graphs', 'clustering', 'percolation', 'aggregation']
        
        print(f"{'=' * 70}")
        print(f"Preparing run: {config.run_name}")
        print(f"{'=' * 70}")
        
        # Print manifest summary first
        self.manifest.print_summary()
        print()
        
        # Extract preexisting annotations if configured
        if config.preexisting_obs_columns:
            print(f"\n--- Preexisting annotation extraction ---")
            try:
                self.extract_preexisting_annotations()
            except Exception as e:
                print(f"  WARNING: Could not extract preexisting annotations: {e}")
                print(f"  (Preexisting percolation jobs will still be generated, but will fail")
                print(f"   unless the .npy files are created before the percolation step runs.)")
        
        results = {}
        
        for step in steps:
            print(f"\n--- Step: {step} ---")
            job_list_path, chunks_dir, num_chunks = self.prepare_step(step)
            
            results[step] = {
                'job_list': str(job_list_path) if job_list_path else None,
                'chunks_dir': str(chunks_dir) if chunks_dir else None,
                'num_chunks': num_chunks,
            }
        
        # Save manifest
        print(f"\n--- Saving manifest ---")
        self.manifest.save_manifest()
        
        # Print summary
        print(f"\n{'=' * 70}")
        print(f"PREPARATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"  {'Step':<20} {'Jobs':>10} {'Chunks':>10}")
        print(f"  {'-' * 20} {'-' * 10} {'-' * 10}")
        for step, info in results.items():
            if info['job_list']:
                # Count jobs from file
                with open(info['job_list']) as f:
                    n_jobs = sum(1 for line in f if line.strip() and not line.startswith('#'))
            else:
                n_jobs = 0
            print(f"  {step:<20} {n_jobs:>10,} {info['num_chunks']:>10}")
        
        print(f"\nAll job files saved to: {config.jobs_dir}")
        print(f"Manifest saved to: {config.jobs_dir / 'manifest.json'}")
        
        return results
    
    # =========================================================================
    # Submit: submit all steps to UGE with dependencies
    # =========================================================================
    
    def submit_step(
        self,
        step: str,
        hold_jid: Optional[str] = None,
        dry_run: bool = False,
    ) -> Optional[str]:
        """
        Submit a single step to UGE.
        
        Args:
            step: Pipeline step name
            hold_jid: Job ID to wait for (dependency)
            dry_run: If True, print script without submitting
            
        Returns:
            UGE job ID (or None if dry run / no jobs)
        """
        from ..uge.submit import submit_array_job, check_qsub_available
        
        config = self.config
        step_dir = config.jobs_dir / step
        chunks_dir = step_dir / "chunks"
        
        if not chunks_dir.exists():
            print(f"  No chunks found for step '{step}'. Run prepare first.")
            return None
        
        # Get resources
        resources = config.hpc_resources(step)
        
        # Build worker command based on step
        worker_commands = {
            'features': 'max-info run-features --chunk-file "$chunk_file"',
            'graphs': 'max-info run-graph --chunk-file "$chunk_file"',
            'clustering': 'max-info run-clustering --chunk-file "$chunk_file"',
            'percolation': 'max-info run-percolation --chunk-file "$chunk_file"',
            'aggregation': 'max-info run-aggregation --chunk-file "$chunk_file" --reduce',
        }
        worker_cmd = worker_commands.get(step)
        
        if not dry_run and not check_qsub_available():
            print("Error: qsub not available. Use --dry-run to see scripts.")
            return None
        
        job_id, script = submit_array_job(
            chunks_dir=str(chunks_dir),
            base_dir=str(config.base_dir),
            temp_dir=str(config.base_dir / "temp"),
            logs_dir=str(config.logs_dir / step),
            job_name=f"{config.run_name}_{step}",
            memory=resources['memory'],
            runtime=resources['runtime'],
            worker_command=worker_cmd,
            hold_jid=hold_jid,
            dry_run=dry_run,
        )
        
        return job_id
    
    def submit_all(self, dry_run: bool = False) -> Dict[str, Optional[str]]:
        """
        Submit all pipeline steps with inter-step dependencies.
        
        Step N+1 waits for step N to complete before starting.
        
        Args:
            dry_run: If True, print scripts without submitting
            
        Returns:
            Dict mapping step name to UGE job ID.
        """
        config = self.config
        steps = ['features', 'graphs', 'clustering', 'percolation', 'aggregation']
        
        print(f"{'=' * 70}")
        print(f"Submitting run: {config.run_name}")
        if dry_run:
            print("(DRY RUN - no jobs will be submitted)")
        print(f"{'=' * 70}")
        
        job_ids = {}
        prev_jid = None
        
        for step in steps:
            print(f"\n--- Submitting: {step} ---")
            
            if prev_jid:
                print(f"  Depends on: {prev_jid}")
            
            jid = self.submit_step(step, hold_jid=prev_jid, dry_run=dry_run)
            job_ids[step] = jid
            
            if jid:
                print(f"  Job ID: {jid}")
                prev_jid = jid
        
        # Print summary
        print(f"\n{'=' * 70}")
        print(f"SUBMISSION COMPLETE")
        print(f"{'=' * 70}")
        print(f"  {'Step':<20} {'Job ID':>15}")
        print(f"  {'-' * 20} {'-' * 15}")
        for step, jid in job_ids.items():
            print(f"  {step:<20} {jid or 'N/A':>15}")
        
        if not dry_run:
            print(f"\nMonitor with: max-info run status --config <config.yaml>")
        
        return job_ids
    
    # =========================================================================
    # Utility: filter out already-completed jobs
    # =========================================================================
    
    def filter_completed_percolation_jobs(self) -> List[str]:
        """
        Generate percolation job list, filtering out already-completed scores.
        
        This is useful for restarting a partially-completed run.
        
        Returns:
            List of job lines for only the missing percolation outputs.
        """
        config = self.config
        all_jobs = self.generate_percolation_jobs()
        
        pending = []
        for job in all_jobs:
            parts = job.split('\t')
            rel_path = parts[0]
            
            # Check if .score file already exists
            score_path = config.percolation_dir / rel_path.replace('.npy', '.score')
            if not score_path.exists():
                pending.append(job)
        
        skipped = len(all_jobs) - len(pending)
        if skipped > 0:
            print(f"Filtered: {skipped} already completed, {len(pending)} pending")
        
        return pending
    
    def prepare_pending_only(self, step: str = 'percolation') -> Tuple[Path, Path, int]:
        """
        Prepare job list for only the pending (not yet completed) jobs.
        
        Useful for resuming a partially-completed run.
        
        Args:
            step: Currently only 'percolation' is supported.
            
        Returns:
            Tuple of (job_list_path, chunks_dir, num_chunks)
        """
        config = self.config
        
        if step != 'percolation':
            raise NotImplementedError(f"Pending-only filtering not yet supported for step '{step}'")
        
        jobs = self.filter_completed_percolation_jobs()
        
        if not jobs:
            print(f"  All {step} jobs are complete!")
            return None, None, 0
        
        # Create step directory
        step_dir = config.jobs_dir / f"{step}_pending"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # Write job list
        job_list_path = step_dir / "jobs.txt"
        write_job_list(jobs, job_list_path)
        
        # Create chunks
        resources = config.hpc_resources(step)
        chunk_size = resources.get('chunk_size', 6000)
        
        chunks_dir = step_dir / "chunks"
        chunk_files, _ = create_chunk_files(
            job_list=jobs,
            output_dir=chunks_dir,
            chunk_size=chunk_size,
        )
        
        return job_list_path, chunks_dir, len(chunk_files)
    
    # =========================================================================
    # Cleanup: remove job lists, chunks, and logs
    # =========================================================================
    
    def clean_jobs(self, step: Optional[str] = None) -> Dict[str, int]:
        """
        Clean job files and chunks for a step or all steps.
        
        Args:
            step: Pipeline step name (None = all steps)
            
        Returns:
            Dict with counts of removed items
        """
        import shutil
        
        config = self.config
        removed = {'job_lists': 0, 'chunk_files': 0, 'directories': 0}
        
        if step:
            # Clean specific step
            step_dir = config.jobs_dir / step
            if step_dir.exists():
                # Count files before removing
                job_list = step_dir / "jobs.txt"
                if job_list.exists():
                    removed['job_lists'] += 1
                
                chunks_dir = step_dir / "chunks"
                if chunks_dir.exists():
                    removed['chunk_files'] += len(list(chunks_dir.glob("chunk_*.txt")))
                
                shutil.rmtree(step_dir)
                removed['directories'] += 1
                print(f"Removed: {step_dir}")
        else:
            # Clean all steps
            if config.jobs_dir.exists():
                for step_dir in config.jobs_dir.iterdir():
                    if step_dir.is_dir():
                        # Count files
                        job_list = step_dir / "jobs.txt"
                        if job_list.exists():
                            removed['job_lists'] += 1
                        
                        chunks_dir = step_dir / "chunks"
                        if chunks_dir.exists():
                            removed['chunk_files'] += len(list(chunks_dir.glob("chunk_*.txt")))
                        
                        shutil.rmtree(step_dir)
                        removed['directories'] += 1
                        print(f"Removed: {step_dir}")
                
                # Remove manifest if present
                manifest_file = config.jobs_dir / "manifest.json"
                if manifest_file.exists():
                    manifest_file.unlink()
                    print(f"Removed: {manifest_file}")
        
        return removed
    
    def clean_logs(self, step: Optional[str] = None) -> Dict[str, int]:
        """
        Clean log files for a step or all steps.
        
        Args:
            step: Pipeline step name (None = all steps)
            
        Returns:
            Dict with counts of removed items
        """
        import shutil
        
        config = self.config
        removed = {'log_files': 0, 'directories': 0}
        
        if step:
            # Clean specific step
            step_log_dir = config.logs_dir / step
            if step_log_dir.exists():
                removed['log_files'] += len(list(step_log_dir.glob("*.log")))
                shutil.rmtree(step_log_dir)
                removed['directories'] += 1
                print(f"Removed: {step_log_dir}")
        else:
            # Clean all logs
            if config.logs_dir.exists():
                for log_dir in config.logs_dir.iterdir():
                    if log_dir.is_dir():
                        removed['log_files'] += len(list(log_dir.glob("*.log")))
                        shutil.rmtree(log_dir)
                        removed['directories'] += 1
                        print(f"Removed: {log_dir}")
        
        return removed
