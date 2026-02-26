"""
Command-line interface for max_info_atlases.

UGE Pipeline Pattern (same for all steps):
    1. max-info jobs create-list --step <step> --output jobs.txt ...
    2. max-info jobs chunk --job-list jobs.txt --chunks-dir chunks/ --chunk-size <N>
    3. max-info jobs submit --chunks-dir chunks/ --worker-command "max-info run-<step> ..."
    4. max-info jobs monitor --job-id <JOB_ID> ...

Available Steps:
    features    - Extract cell type features (PCA)
    graph       - Build k-NN graph
    clustering  - Leiden clustering at multiple resolutions
    percolation - Run percolation analysis
    aggregation - Aggregate .score files into CSV

Workers (called by UGE jobs):
    max-info run-features    --chunk-file chunk.txt
    max-info run-graph       --chunk-file chunk.txt
    max-info run-clustering  --chunk-file chunk.txt
    max-info run-percolation --chunk-file chunk.txt --type-data-dir ... --output-dir ...
    max-info run-aggregation --chunk-file chunk.txt

Interactive Commands (for development/testing):
    max-info features celltype --input data.h5ad --output features/ --type pca50
    max-info features graph --input features.npy --output FEL.npy --k 15
    max-info cluster leiden --input FEL.npy --output clusters/ --resolution-idx 25
    max-info results aggregate-scores --base-dir results/ --output scores.csv
"""

import click
from pathlib import Path


@click.group()
@click.version_option()
def cli():
    """Max Info Atlases - Cell type clustering and percolation analysis pipeline."""
    pass


# ============================================================================
# Features Commands
# ============================================================================

@cli.group()
def features():
    """Feature extraction commands."""
    pass


@features.command('celltype')
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True),
              help='Input AnnData file (.h5ad)')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory for features')
@click.option('--type', '-t', 'data_type', default='raw',
              type=click.Choice(['raw', 'log1p', 'pca', 'pca15', 'pca50', 'scvi']),
              help='Feature type to extract')
@click.option('--n-pcs', default=50, help='Number of PCA components')
def features_celltype(input_file, output_dir, data_type, n_pcs):
    """Extract cell type features from AnnData."""
    import scanpy as sc
    from ..features.celltype import extract_celltype_features
    
    click.echo(f"Loading {input_file}")
    adata = sc.read_h5ad(input_file)
    
    click.echo(f"Extracting {data_type} features")
    features = extract_celltype_features(adata, data_type, n_pcs)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import numpy as np
    output_file = output_dir / f"features_{data_type}.npy"
    np.save(output_file, features)
    
    click.echo(f"Saved features to {output_file}")


@features.command('localfreq')
@click.option('--adata', '-a', 'adata_path', required=True, type=click.Path(exists=True),
              help='Input AnnData file (.h5ad)')
@click.option('--xy-dir', '-x', required=True, type=click.Path(exists=True),
              help='Directory containing {section}_XY.npy files')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory for features')
@click.option('--type-column', '-t', required=True,
              help='Column in adata.obs with cell type labels')
@click.option('--k', '-k', required=True, type=int, help='Number of neighbors')
@click.option('--weighted/--no-weighted', default=False, help='Use Gaussian-weighted counts')
@click.option('--sections-file', type=click.Path(exists=True),
              help='Path to Sections.npy file (optional)')
@click.option('--section-column', help='Column in adata.obs with section labels (optional)')
def features_localfreq(adata_path, xy_dir, output_dir, type_column, k, weighted, sections_file, section_column):
    """Extract local frequency features from spatial data."""
    from ..features.local_frequency import extract_and_save_local_frequency
    
    click.echo(f"Extracting local frequency features:")
    click.echo(f"  k={k}, weighted={weighted}, type_column={type_column}")
    
    output_file = extract_and_save_local_frequency(
        adata_path=adata_path,
        xy_dir=xy_dir,
        output_dir=output_dir,
        type_column=type_column,
        k=k,
        weighted=weighted,
        sections_file=sections_file,
        section_column=section_column,
    )
    
    click.echo(f"✓ Saved to {output_file}")


@features.command('env')
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True),
              help='Input AnnData file or XY coordinates')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory')
@click.option('--k', '-k', required=True, type=int, help='Number of neighbors')
@click.option('--weighted/--no-weighted', default=True, help='Use Gaussian-weighted counts')
def features_env(input_file, output_dir, k, weighted):
    """Create environment features from spatial data (deprecated, use localfreq)."""
    click.echo(f"Creating environment features with k={k}, weighted={weighted}")
    click.echo("(Implementation requires XY coordinates and type vectors)")
    click.echo("NOTE: Use 'max-info features localfreq' for the new implementation")


@features.command('derived')
@click.option('--input', '-i', 'source_file', required=True, type=click.Path(exists=True),
              help='Input .npy feature file (source feature)')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory')
@click.option('--operation', required=True, type=click.Choice(['pca']),
              help='Derived feature operation')
@click.option('--n-components', '-n', type=int, help='Number of components (for PCA)')
@click.option('--scale/--no-scale', default=True, help='Standardize features before PCA')
def features_derived(source_file, output_dir, operation, n_components, scale):
    """Compute derived features from existing feature matrices."""
    from ..features.derived import extract_and_save_derived_feature
    
    if operation == 'pca' and n_components is None:
        click.echo("ERROR: --n-components required for PCA operation", err=True)
        return
    
    params = {'n_components': n_components, 'scale': scale}
    
    click.echo(f"Computing {operation} derived features:")
    click.echo(f"  Source: {source_file}")
    click.echo(f"  Params: {params}")
    
    output_file = extract_and_save_derived_feature(
        source_feature_path=source_file,
        output_dir=output_dir,
        operation=operation,
        params=params,
    )
    
    click.echo(f"✓ Saved to {output_file}")


@features.command('graph')
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True),
              help='Input .npy feature file (from features celltype)')
@click.option('--output', '-o', 'output_file', required=True, type=click.Path(),
              help='Output FEL.npy file')
@click.option('--k', default=15, help='Number of nearest neighbors')
@click.option('--metric', default='cosine', 
              type=click.Choice(['cosine', 'correlation', 'euclidean']),
              help='Distance metric')
def features_graph(input_file, output_file, k, metric):
    """Build k-NN graph from features."""
    import numpy as np
    from ..features.graphs import build_knn_graph_from_file, save_edge_list
    
    click.echo(f"Building k-NN graph with k={k}, metric={metric}")
    
    edge_list = build_knn_graph_from_file(input_file, k=k, metric=metric)
    save_edge_list(edge_list, output_file)
    
    click.echo(f"Saved {len(edge_list)} edges to {output_file}")


# ============================================================================
# Clustering Commands
# ============================================================================

@cli.group()
def cluster():
    """Clustering commands."""
    pass


@cluster.command('leiden')
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True),
              help='Input FEL.npy edge list file')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory')
@click.option('--resolution', '-r', type=float, help='Resolution parameter')
@click.option('--resolution-idx', type=int, help='Resolution index (0-49)')
@click.option('--sections', type=click.Path(exists=True),
              help='Sections.npy file for splitting output')
def cluster_leiden(input_file, output_dir, resolution, resolution_idx, sections):
    """Run Leiden clustering on an edge list graph."""
    import numpy as np
    from ..clustering.leiden import LeidenClustering
    
    click.echo(f"Loading edge list from {input_file}")
    edge_list = np.load(input_file)
    
    from ..clustering.base import format_resolution_dirname
    
    clustering = LeidenClustering(resolution=resolution, resolution_idx=resolution_idx)
    click.echo(f"Running Leiden with resolution={clustering.resolution} (idx={resolution_idx})")
    
    assignments = clustering.fit(edge_list)
    click.echo(f"Found {len(np.unique(assignments))} clusters")
    
    # Create resolution-specific subdirectory
    base_output_dir = Path(output_dir)
    if resolution_idx is not None:
        res_dirname = format_resolution_dirname(clustering.resolution)
        output_dir = base_output_dir / res_dirname
    else:
        output_dir = base_output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if sections:
        section_labels = np.load(sections, allow_pickle=True)
        for section in np.unique(section_labels):
            mask = section_labels == section
            section_file = output_dir / f"{section}.npy"
            np.save(section_file, assignments[mask])
        click.echo(f"Saved {len(np.unique(section_labels))} sections to {output_dir}")
    else:
        np.save(output_dir / "clusters.npy", assignments)
        click.echo(f"Saved clusters to {output_dir / 'clusters.npy'}")


@cluster.command('kmeans')
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True),
              help='Input CSV file')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory')
@click.option('--k', '-k', required=True, type=int, help='Number of clusters')
@click.option('--pca', type=int, help='Number of PCA components (optional)')
def cluster_kmeans(input_file, output_dir, k, pca):
    """Run K-means clustering on features."""
    import numpy as np
    import pandas as pd
    from ..clustering.kmeans import KMeansClustering
    
    click.echo(f"Loading data from {input_file}")
    df = pd.read_csv(input_file, index_col=None)
    sections = df.iloc[:, 0]
    features = df.iloc[:, 1:].select_dtypes(include=[np.number]).values
    
    clustering = KMeansClustering(n_clusters=k, n_pcs=pca)
    click.echo(f"Running K-means with k={k}, pca={pca}")
    
    assignments = clustering.fit(features)
    
    output_dir = Path(output_dir) / f"k_{k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for section in pd.unique(sections):
        mask = sections == section
        np.save(output_dir / f"{section}.npy", assignments[mask])
    
    click.echo(f"Saved clusters to {output_dir}")


@cluster.command('lda')
@click.option('--input', '-i', 'input_file', required=True, type=click.Path(exists=True),
              help='Input CSV file')
@click.option('--output', '-o', 'output_dir', required=True, type=click.Path(),
              help='Output directory')
@click.option('--n-topics', '-n', required=True, type=int, help='Number of topics')
def cluster_lda(input_file, output_dir, n_topics):
    """Run LDA clustering on features."""
    import numpy as np
    import pandas as pd
    from ..clustering.lda import LDAClustering
    
    click.echo(f"Loading data from {input_file}")
    df = pd.read_csv(input_file, index_col=None)
    sections = df.iloc[:, 0]
    features = df.iloc[:, 1:].select_dtypes(include=[np.number]).values
    
    clustering = LDAClustering(n_topics=n_topics)
    click.echo(f"Running LDA with n_topics={n_topics}")
    
    assignments = clustering.fit(features)
    
    output_dir = Path(output_dir) / f"k_{n_topics}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for section in pd.unique(sections):
        mask = sections == section
        np.save(output_dir / f"{section}.npy", assignments[mask])
    
    click.echo(f"Saved clusters to {output_dir}")


# ============================================================================
# Jobs Commands (UGE management)
# ============================================================================

@cli.group()
def jobs():
    """UGE job management commands."""
    pass


@jobs.command('create-list')
@click.option('--step', '-s', required=True, 
              type=click.Choice(['features', 'graph', 'clustering', 'percolation', 'aggregation']),
              help='Pipeline step to create job list for')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output job list file')
# Features options
@click.option('--input-file', '-i', type=click.Path(), help='Input file (features/graph)')
@click.option('--output-dir', type=click.Path(), help='Output directory')
@click.option('--feature-type', default='pca50', help='Feature type (features step)')
# Graph options
@click.option('--output-file', type=click.Path(), help='Output file (graph step)')
@click.option('--k', default=15, help='Number of neighbors (graph step)')
@click.option('--metric', default='correlation', help='Distance metric (graph step)')
# Clustering options
@click.option('--sections-file', type=click.Path(), help='Sections file (clustering step)')
@click.option('--resolutions', help='Comma-separated resolution indices (clustering step)')
# Percolation options
@click.option('--type-data-dir', type=click.Path(), help='Type data directory (percolation step)')
@click.option('--edge-list-dir', type=click.Path(), help='Edge list directory (percolation step)')
@click.option('--xy-data-dir', type=click.Path(), help='XY data directory (percolation step)')
@click.option('--pattern', default='res_*/*.npy', help='File pattern (percolation step)')
@click.option('--max-k', default=100, help='Max k for percolation')
# Aggregation options
@click.option('--results-dir', type=click.Path(), help='Results directory (aggregation step)')
@click.option('--score-pattern', default='**/*.score', help='Score file pattern (aggregation step)')
def jobs_create_list(step, output, input_file, output_dir, feature_type, output_file,
                     k, metric, sections_file, resolutions, type_data_dir, edge_list_dir,
                     xy_data_dir, pattern, max_k, results_dir, score_pattern):
    """Create a job list file for a pipeline step."""
    from ..uge.job_generator import (
        generate_features_jobs, generate_graph_jobs, generate_clustering_jobs,
        generate_percolation_jobs, generate_aggregation_jobs, write_job_list
    )
    
    if step == 'features':
        if not input_file or not output_dir:
            raise click.UsageError("--input-file and --output-dir required for features step")
        jobs = generate_features_jobs(input_file, output_dir, feature_type)
        
    elif step == 'graph':
        if not input_file or not output_file:
            raise click.UsageError("--input-file and --output-file required for graph step")
        jobs = generate_graph_jobs(input_file, output_file, k, metric)
        
    elif step == 'clustering':
        if not input_file or not output_dir or not sections_file or not resolutions:
            raise click.UsageError("--input-file, --output-dir, --sections-file, --resolutions required for clustering step")
        res_list = [int(r.strip()) for r in resolutions.split(',')]
        jobs = generate_clustering_jobs(input_file, output_dir, sections_file, res_list)
        
    elif step == 'percolation':
        if not type_data_dir or not output_dir or not edge_list_dir:
            raise click.UsageError("--type-data-dir, --output-dir, --edge-list-dir required for percolation step")
        jobs = generate_percolation_jobs(type_data_dir, output_dir, edge_list_dir, 
                                         pattern, max_k, xy_data_dir)
        
    elif step == 'aggregation':
        if not results_dir or not output_file:
            raise click.UsageError("--results-dir and --output-file required for aggregation step")
        jobs = generate_aggregation_jobs(results_dir, output_file, score_pattern)
    
    write_job_list(jobs, output)
    click.echo(f"Created job list with {len(jobs)} jobs")
    click.echo(f"Chunk with: max-info jobs chunk --job-list {output} --chunks-dir <dir>")


@jobs.command('chunk')
@click.option('--job-list', '-j', required=True, type=click.Path(exists=True),
              help='Job list file to chunk')
@click.option('--chunks-dir', '-c', required=True, type=click.Path(),
              help='Directory for chunk files')
@click.option('--chunk-size', '-s', default=1000000, 
              help='Jobs per chunk (use large number for single job, default: 1000000)')
def jobs_chunk(job_list, chunks_dir, chunk_size):
    """Chunk a job list file into smaller pieces for array job processing."""
    from ..uge.job_generator import chunk_job_list_file
    
    num_chunks, chunks_path = chunk_job_list_file(
        job_list_file=job_list,
        chunks_dir=chunks_dir,
        chunk_size=chunk_size,
    )
    
    click.echo(f"\nCreated {num_chunks} chunk(s) in {chunks_path}")
    click.echo(f"Submit with: max-info jobs submit --chunks-dir {chunks_path} --worker-command '...'")


@jobs.command('prepare')
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True),
              help='Directory containing input files')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Directory for outputs (to filter completed)')
@click.option('--chunks-dir', '-c', required=True, type=click.Path(),
              help='Directory for chunk files')
@click.option('--chunk-size', default=6000, help='Number of files per chunk')
@click.option('--pattern', default='**/*.npy', help='Glob pattern for input files')
def jobs_prepare(input_dir, output_dir, chunks_dir, chunk_size, pattern):
    """Prepare job list by scanning directory and create chunk files (legacy command)."""
    from ..uge.job_generator import prepare_jobs
    
    num_chunks, chunks_path = prepare_jobs(
        input_dir=input_dir,
        output_dir=output_dir,
        chunks_dir=chunks_dir,
        chunk_size=chunk_size,
        pattern=pattern,
    )
    
    click.echo(f"\nCreated {num_chunks} chunks in {chunks_path}")
    click.echo(f"Ready for submission with: max-info jobs submit --chunks-dir {chunks_path}")


@jobs.command('submit')
@click.option('--chunks-dir', '-c', required=True, type=click.Path(exists=True),
              help='Directory containing chunk files')
@click.option('--base-dir', '-b', required=True, type=click.Path(),
              help='Base directory for input data (may not exist yet if using --hold-jid)')
@click.option('--temp-dir', '-t', required=True, type=click.Path(),
              help='Directory for temporary outputs')
@click.option('--logs-dir', '-l', required=True, type=click.Path(),
              help='Directory for log files')
@click.option('--job-name', '-n', default='percolation', help='Job name')
@click.option('--memory', '-m', default='16G', help='Memory request')
@click.option('--runtime', '-r', default='4:00:00', help='Runtime request')
@click.option('--worker-command', help='Custom worker command (default: max-info worker)')
@click.option('--hold-jid', help='Job ID to wait for before starting (creates UGE dependency)')
@click.option('--dry-run', is_flag=True, help='Print script without submitting')
def jobs_submit(chunks_dir, base_dir, temp_dir, logs_dir, job_name, memory, runtime, worker_command, hold_jid, dry_run):
    """Submit array job to UGE."""
    from ..uge.submit import submit_array_job, check_qsub_available
    
    if not dry_run and not check_qsub_available():
        click.echo("Error: qsub not available on this system", err=True)
        raise SystemExit(1)
    
    job_id, script = submit_array_job(
        chunks_dir=chunks_dir,
        base_dir=base_dir,
        temp_dir=temp_dir,
        logs_dir=logs_dir,
        job_name=job_name,
        memory=memory,
        runtime=runtime,
        worker_command=worker_command,
        hold_jid=hold_jid,
        dry_run=dry_run,
    )
    
    if job_id:
        click.echo(f"\nJob submitted: {job_id}")
        if hold_jid:
            click.echo(f"Depends on: {hold_jid} (will wait until that job completes)")
        click.echo(f"Monitor with: max-info jobs monitor --job-id {job_id} ...")


@jobs.command('monitor')
@click.option('--job-id', '-j', required=True, help='UGE job ID')
@click.option('--chunks-dir', '-c', required=True, type=click.Path(exists=True),
              help='Directory containing chunk files')
@click.option('--temp-dir', '-t', required=True, type=click.Path(),
              help='Directory for temporary outputs')
@click.option('--base-dir', '-b', required=True, type=click.Path(exists=True),
              help='Base directory for input data')
@click.option('--logs-dir', '-l', required=True, type=click.Path(),
              help='Directory for log files')
@click.option('--timeout-minutes', default=60, help='Minutes without progress to consider stuck')
@click.option('--poll-interval', default=60, help='Seconds between status checks')
@click.option('--auto-resubmit', is_flag=True, help='Automatically resubmit stuck tasks')
@click.option('--extra-runtime', default='2:00:00', help='Runtime for resubmitted jobs')
@click.option('--combine-on-complete', is_flag=True, help='Combine results when complete')
@click.option('--output', '-o', type=click.Path(), help='Output file for combined results')
def jobs_monitor(job_id, chunks_dir, temp_dir, base_dir, logs_dir,
                 timeout_minutes, poll_interval, auto_resubmit, extra_runtime,
                 combine_on_complete, output):
    """Monitor job and optionally auto-resubmit stuck tasks."""
    import sys
    from ..uge.monitor import JobMonitor
    from ..percolation.analysis import combine_chunk_results
    
    def on_complete():
        if combine_on_complete and output:
            click.echo("\nCombining results...")
            combine_chunk_results(temp_dir, output, cleanup=False)
    
    monitor = JobMonitor(
        job_id=job_id,
        chunks_dir=chunks_dir,
        temp_dir=temp_dir,
        base_dir=base_dir,
        logs_dir=logs_dir,
    )
    
    success = monitor.monitor_loop(
        timeout_minutes=timeout_minutes,
        poll_interval=poll_interval,
        auto_resubmit=auto_resubmit,
        extra_runtime=extra_runtime,
        on_complete=on_complete if combine_on_complete else None,
    )
    
    if not success:
        sys.exit(1)


@jobs.command('status')
@click.option('--job-id', '-j', required=True, help='UGE job ID')
@click.option('--chunks-dir', '-c', required=True, type=click.Path(exists=True),
              help='Directory containing chunk files')
@click.option('--temp-dir', '-t', required=True, type=click.Path(),
              help='Directory for temporary outputs')
def jobs_status(job_id, chunks_dir, temp_dir):
    """Quick status check for a job."""
    from ..uge.monitor import JobMonitor
    
    monitor = JobMonitor(
        job_id=job_id,
        chunks_dir=chunks_dir,
        temp_dir=temp_dir,
        base_dir=".",
        logs_dir=".",
    )
    
    completed, total = monitor.count_completed()
    tasks = monitor.parse_qstat()
    running = sum(1 for t in tasks.values() if t.state == 'r')
    queued = sum(1 for t in tasks.values() if t.state == 'qw')
    
    click.echo(f"Job {job_id} Status:")
    click.echo(f"  Completed: {completed}/{total}")
    click.echo(f"  Running: {running}")
    click.echo(f"  Queued: {queued}")


@jobs.command('calibrate')
@click.option('--input-dir', '-i', required=True, type=click.Path(exists=True),
              help='Directory containing input files')
@click.option('--output-dir', '-o', required=True, type=click.Path(),
              help='Directory for calibration output')
@click.option('--n-files', default=10, help='Number of files for calibration (default: 10)')
@click.option('--pattern', default='**/*.npy', help='Glob pattern for input files')
def jobs_calibrate_prepare(input_dir, output_dir, n_files, pattern):
    """Create a calibration chunk with a few files to measure timing."""
    from ..uge.job_generator import scan_input_files, create_calibration_chunk
    
    click.echo(f"Scanning for files in: {input_dir}")
    job_list = scan_input_files(input_dir, pattern)
    
    if not job_list:
        click.echo("No files found!", err=True)
        raise SystemExit(1)
    
    click.echo(f"Found {len(job_list)} total files")
    
    chunk_file = create_calibration_chunk(job_list, output_dir, n_files)
    
    click.echo(f"\nCalibration chunk created: {chunk_file}")
    click.echo(f"\nNext steps:")
    click.echo(f"  1. Submit calibration job:")
    click.echo(f"     max-info worker --chunk-file {chunk_file} --base-dir {input_dir} --output-dir {output_dir}")
    click.echo(f"  2. Or submit via UGE and check timing in log")
    click.echo(f"  3. Then run: max-info jobs calc-chunk-size --time-per-file <seconds>")


@jobs.command('calc-chunk-size')
@click.option('--time-per-file', '-t', required=True, type=float,
              help='Measured time per file in seconds')
@click.option('--target-runtime', '-r', default=1.0,
              help='Target runtime per chunk in hours (default: 1.0)')
@click.option('--safety-factor', '-s', default=0.8,
              help='Safety factor (default: 0.8 = 80%% of max)')
def jobs_calc_chunk_size(time_per_file, target_runtime, safety_factor):
    """Calculate optimal chunk size based on calibration timing."""
    from ..uge.job_generator import calculate_optimal_chunk_size
    
    chunk_size = calculate_optimal_chunk_size(
        time_per_file=time_per_file,
        target_runtime_hours=target_runtime,
        safety_factor=safety_factor,
    )
    
    files_per_hour = 3600 / time_per_file
    estimated_runtime = chunk_size * time_per_file / 3600
    
    click.echo(f"\n=== Chunk Size Calculator ===")
    click.echo(f"Time per file: {time_per_file:.3f} seconds")
    click.echo(f"Files per hour: {files_per_hour:.0f}")
    click.echo(f"Target runtime: {target_runtime:.1f} hours")
    click.echo(f"Safety factor: {safety_factor:.0%}")
    click.echo(f"")
    click.echo(f"Recommended chunk size: {chunk_size}")
    click.echo(f"Estimated runtime per chunk: {estimated_runtime:.2f} hours")
    click.echo(f"")
    click.echo(f"Use with:")
    click.echo(f"  max-info jobs prepare --chunk-size {chunk_size} ...")


# ============================================================================
# Results Commands
# ============================================================================

@cli.group()
def results():
    """Result aggregation commands."""
    pass


@results.command('combine')
@click.option('--temp-dir', '-t', required=True, type=click.Path(exists=True),
              help='Directory containing chunk_result_*.csv files')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output CSV file')
@click.option('--cleanup', is_flag=True, help='Remove temp files after combining')
def results_combine(temp_dir, output, cleanup):
    """Combine chunked results into final CSV."""
    from ..percolation.analysis import combine_chunk_results
    
    df = combine_chunk_results(temp_dir, output, cleanup)
    
    if len(df) > 0:
        click.echo(f"\nCombined {len(df)} results")
    else:
        click.echo("No results to combine", err=True)


@results.command('aggregate-scores')
@click.option('--base-dir', '-b', required=True, type=click.Path(exists=True),
              help='Base directory containing .score files')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output CSV file')
def results_aggregate_scores(base_dir, output):
    """Aggregate .score files into CSV (efficient for distributed workflows)."""
    from ..percolation.analysis import aggregate_score_files_to_csv
    
    df = aggregate_score_files_to_csv(base_dir, output)
    
    if len(df) > 0:
        click.echo(f"\nAggregated {len(df)} scores")
    else:
        click.echo("No scores found", err=True)


@results.command('reduce-scores')
@click.option('--scores-csv', '-s', required=True, type=click.Path(exists=True),
              help='Input CSV with per-section scores (from aggregate-scores)')
@click.option('--xy-dir', '-x', required=True, type=click.Path(exists=True),
              help='Directory containing {section}_XY.npy files')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output CSV file with reduced scores')
@click.option('--section-column', default='section_name',
              help='Name of column containing section names (default: section_name)')
def results_reduce_scores(scores_csv, xy_dir, output, section_column):
    """Reduce per-section scores to weighted averages across sections.
    
    This computes weighted average scores where weights are the number of cells per section.
    The output contains one row per algorithm/configuration with aggregated statistics.
    """
    from ..percolation.analysis import reduce_scores_by_section
    
    df = reduce_scores_by_section(scores_csv, xy_dir, output, section_column)
    
    if len(df) > 0:
        click.echo(f"\n✓ Reduced to {len(df)} configurations")
    else:
        click.echo("No results produced", err=True)


# ============================================================================
# Run Commands (end-to-end pipeline orchestration)
# ============================================================================

@cli.group()
def run():
    """End-to-end pipeline run management."""
    pass


@run.command('prepare')
@click.option('--config', '-c', 'config_path', required=True, type=click.Path(exists=True),
              help='Run config YAML file')
@click.option('--pending-only', is_flag=True,
              help='Only generate jobs for missing outputs (resume mode)')
@click.option('--redo', is_flag=True,
              help='Regenerate all jobs, ignoring existing outputs')
def run_prepare(config_path, pending_only, redo):
    """Generate job lists and chunks for all pipeline steps."""
    from ..run import RunConfig, RunOrchestrator
    
    if pending_only and redo:
        click.echo("Error: --pending-only and --redo are mutually exclusive", err=True)
        return
    
    config = RunConfig.from_yaml(config_path)
    
    # Determine skip_completed behavior
    # Default: skip completed (smart continuation)
    # --pending-only: same as default (skip completed)
    # --redo: don't skip anything
    skip_completed = not redo
    
    orch = RunOrchestrator(config, skip_completed=skip_completed)
    
    if pending_only:
        click.echo("Preparing pending (incomplete) jobs only...")
        job_list, chunks_dir, n_chunks = orch.prepare_pending_only('percolation')
        if n_chunks > 0:
            click.echo(f"\nCreated {n_chunks} chunks in {chunks_dir}")
        else:
            click.echo("Nothing to do - all jobs complete!")
    else:
        orch.prepare_all()


@run.command('status')
@click.option('--config', '-c', 'config_path', required=True, type=click.Path(exists=True),
              help='Run config YAML file')
@click.option('--show-missing', '-m', default=5, type=int,
              help='Number of missing files to show per step (0=none)')
def run_status(config_path, show_missing):
    """Check progress of a pipeline run against expected outputs."""
    from ..run import RunConfig, RunManifest
    
    config = RunConfig.from_yaml(config_path)
    manifest = RunManifest(config)
    manifest.print_progress(show_missing=show_missing)


@run.command('analyze-error')
@click.option('--config', '-c', 'config_path', required=True, type=click.Path(exists=True),
              help='Run config YAML file')
@click.option('--step', type=click.Choice(['features', 'graphs', 'clustering', 'percolation', 'aggregation']),
              help='Analyze a single step only (default: all steps)')
@click.option('--max-logs', default=20, type=int,
              help='Maximum suspicious logs to print')
@click.option('--max-jobs-per-log', default=5, type=int,
              help='Maximum job parameter lines to show per suspicious log')
@click.option('--max-missing-per-log', default=5, type=int,
              help='Maximum missing outputs to show per suspicious log')
def run_analyze_error(config_path, step, max_logs, max_jobs_per_log, max_missing_per_log):
    """Find suspicious logs and show associated run parameters + missing outputs."""
    from ..run import RunConfig
    from ..run.error_analysis import analyze_run_errors

    config = RunConfig.from_yaml(config_path)
    analyze_run_errors(
        config=config,
        step=step,
        max_logs=max_logs,
        max_jobs_per_log=max_jobs_per_log,
        max_missing_per_log=max_missing_per_log,
    )


@run.command('summary')
@click.option('--config', '-c', 'config_path', required=True, type=click.Path(exists=True),
              help='Run config YAML file')
def run_summary(config_path):
    """Print summary of what a run will produce (without generating jobs)."""
    from ..run import RunConfig, RunManifest
    
    config = RunConfig.from_yaml(config_path)
    manifest = RunManifest(config)
    manifest.print_summary()


@run.command('submit')
@click.option('--config', '-c', 'config_path', required=True, type=click.Path(exists=True),
              help='Run config YAML file')
@click.option('--step', '-s', type=click.Choice(['features', 'graphs', 'clustering', 'percolation', 'aggregation']),
              help='Submit only a specific step (default: all steps)')
@click.option('--hold-jid', help='Job ID to wait for before starting')
@click.option('--dry-run', is_flag=True, help='Print scripts without submitting')
def run_submit(config_path, step, hold_jid, dry_run):
    """Submit pipeline steps to UGE (requires prepare first)."""
    from ..run import RunConfig, RunOrchestrator
    
    config = RunConfig.from_yaml(config_path)
    orch = RunOrchestrator(config)
    
    if step:
        click.echo(f"Submitting step: {step}")
        jid = orch.submit_step(step, hold_jid=hold_jid, dry_run=dry_run)
        if jid:
            click.echo(f"Job ID: {jid}")
    else:
        orch.submit_all(dry_run=dry_run)


@run.command('manifest')
@click.option('--config', '-c', 'config_path', required=True, type=click.Path(exists=True),
              help='Run config YAML file')
@click.option('--output', '-o', type=click.Path(),
              help='Output manifest JSON path (default: {jobs_dir}/manifest.json)')
def run_manifest(config_path, output):
    """Save full manifest of expected outputs to JSON."""
    from ..run import RunConfig, RunManifest
    
    config = RunConfig.from_yaml(config_path)
    manifest = RunManifest(config)
    manifest.save_manifest(output)


@run.command('extract-preexisting')
@click.option('--config', '-c', 'config_path', required=True, type=click.Path(exists=True),
              help='Run config YAML file')
def run_extract_preexisting(config_path):
    """Extract preexisting annotations from AnnData .obs into per-section .npy files."""
    from ..run import RunConfig, RunOrchestrator
    
    config = RunConfig.from_yaml(config_path)
    orch = RunOrchestrator(config)
    
    summary = orch.extract_preexisting_annotations()
    if summary:
        click.echo(f"\nExtracted {len(summary)} preexisting annotation levels:")
        for level, info in summary.items():
            click.echo(f"  Preexisting_{level}: {info['n_categories']} categories, "
                       f"{info['n_sections']} sections")


@run.command('timing')
@click.option('--config', '-c', 'config_path', required=True, type=click.Path(exists=True),
              help='Run config YAML file')
@click.option('--step', type=click.Choice(['features', 'graphs', 'clustering', 'percolation', 'aggregation']),
              help='Analyze specific step (default: all steps)')
def run_timing(config_path, step):
    """Analyze log files to estimate per-chunk task timings."""
    from ..run import RunConfig
    from ..utils.timing import analyze_log_timings
    
    config = RunConfig.from_yaml(config_path)
    
    click.echo(f"Analyzing log timings for run: {config.run_name}")
    click.echo()
    
    steps_to_analyze = [step] if step else ['features', 'graphs', 'clustering', 'percolation', 'aggregation']
    
    for step_name in steps_to_analyze:
        analyze_log_timings(config, step_name)


@run.command('clean')
@click.option('--config', '-c', 'config_path', required=True, type=click.Path(exists=True),
              help='Run config YAML file')
@click.option('--step', type=click.Choice(['features', 'graphs', 'clustering', 'percolation', 'aggregation']),
              help='Clean specific step (default: all steps)')
@click.option('--jobs/--no-jobs', default=True,
              help='Clean job files and chunks')
@click.option('--logs/--no-logs', default=True,
              help='Clean log files')
def run_clean(config_path, step, jobs, logs):
    """Clean job files, chunks, and logs for a run."""
    from ..run import RunConfig, RunOrchestrator
    
    config = RunConfig.from_yaml(config_path)
    orch = RunOrchestrator(config)
    
    click.echo(f"Cleaning run: {config.run_name}")
    if step:
        click.echo(f"Step: {step}")
    else:
        click.echo(f"Step: all")
    click.echo()
    
    total_removed = {'job_lists': 0, 'chunk_files': 0, 'log_files': 0, 'directories': 0}
    
    if jobs:
        click.echo("--- Cleaning job files and chunks ---")
        removed = orch.clean_jobs(step=step)
        total_removed['job_lists'] += removed['job_lists']
        total_removed['chunk_files'] += removed['chunk_files']
        total_removed['directories'] += removed['directories']
        click.echo(f"  Removed {removed['job_lists']} job list(s)")
        click.echo(f"  Removed {removed['chunk_files']} chunk file(s)")
        click.echo()
    
    if logs:
        click.echo("--- Cleaning log files ---")
        removed = orch.clean_logs(step=step)
        total_removed['log_files'] += removed['log_files']
        total_removed['directories'] += removed['directories']
        click.echo(f"  Removed {removed['log_files']} log file(s)")
        click.echo()
    
    click.echo("Done!")


# ============================================================================
# Worker Commands (called by UGE jobs)
# ============================================================================

@cli.command('worker')
@click.option('--chunk-file', '-f', required=True, type=click.Path(exists=True),
              help='Chunk file to process')
@click.option('--base-dir', '-b', required=True, type=click.Path(exists=True),
              help='Base directory for input data')
@click.option('--output-dir', '-o', required=True, type=click.Path(),
              help='Output directory for results')
def worker(chunk_file, base_dir, output_dir):
    """Worker process for score aggregation (processes existing .npz files)."""
    import os
    from ..percolation.analysis import process_chunk
    
    click.echo(f"Processing chunk: {chunk_file}")
    
    df = process_chunk(chunk_file, base_dir, output_dir)
    
    if len(df) > 0:
        # Save results
        chunk_name = Path(chunk_file).stem
        output_file = Path(output_dir) / f"chunk_result_{chunk_name}_{os.getpid()}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        click.echo(f"Saved {len(df)} results to {output_file}")
    else:
        click.echo("No results generated", err=True)


@cli.command('run-features')
@click.option('--chunk-file', '-f', required=True, type=click.Path(exists=True),
              help='Chunk file with job parameters (supports expression, local frequency, and derived features)')
def run_features(chunk_file):
    """
    Worker for feature extraction (processes chunk of feature jobs).
    
    Supports three job formats:
    1. Expression: input_file\\toutput_dir\\tfeature_type
    2. Local frequency: localfreq\\tinput_file\\toutput_dir\\tfeature_type\\txy_dir\\tsections_file\\ttype_column
    3. Derived: derived\\tsource_file\\toutput_dir\\tderived_name\\tparams_json
    """
    from ..uge.job_generator import read_job_list
    from ..features.celltype import extract_celltype_features
    from ..features.local_frequency import extract_and_save_local_frequency
    from ..features.derived import extract_and_save_derived_feature
    import scanpy as sc
    import numpy as np
    import json
    
    jobs = read_job_list(chunk_file)
    click.echo(f"Processing {len(jobs)} feature jobs from chunk")
    
    for job in jobs:
        parts = job.split('\t')
        if len(parts) < 3:
            click.echo(f"  Skipping malformed job: {job}", err=True)
            continue
        
        # Detect job type
        if parts[0] == 'localfreq':
            # Local frequency feature
            # Format: localfreq\tinput_file\toutput_dir\tfeature_type\txy_dir\tsections_file\ttype_column
            if len(parts) < 7:
                click.echo(f"  Skipping malformed localfreq job: {job}", err=True)
                continue
            
            _, input_file, output_dir, feature_type, xy_dir, sections_file, type_column = parts[:7]
            
            click.echo(f"  Processing localfreq: {feature_type}")
            
            # Extract k and weighted from feature_type name
            # localfreq_k50 or localfreq_w_k50
            parts_name = feature_type.split('_')
            weighted = 'w' in parts_name
            
            # Find k value
            k = None
            for part in parts_name:
                if part.startswith('k') and part[1:].isdigit():
                    k = int(part[1:])
                    break
            
            if k is None:
                click.echo(f"  ERROR: Could not parse k from feature_type: {feature_type}", err=True)
                continue
            
            try:
                output_file = extract_and_save_local_frequency(
                    adata_path=input_file,
                    xy_dir=xy_dir,
                    output_dir=output_dir,
                    type_column=type_column,
                    k=k,
                    weighted=weighted,
                    sections_file=sections_file,
                )
                click.echo(f"    Saved to {output_file}")
            except Exception as e:
                click.echo(f"  ERROR processing localfreq {feature_type}: {e}", err=True)
                continue
        
        elif parts[0] == 'derived':
            # Derived feature
            # Format: derived\tsource_file\toutput_dir\tderived_name\tparams_json
            if len(parts) < 5:
                click.echo(f"  Skipping malformed derived job: {job}", err=True)
                continue
            
            _, source_file, output_dir, derived_name, params_json = parts[:5]
            
            click.echo(f"  Processing derived: {derived_name}")
            
            try:
                params = json.loads(params_json)
                operation = params.get('operation')
                
                output_file = extract_and_save_derived_feature(
                    source_feature_path=source_file,
                    output_dir=output_dir,
                    operation=operation,
                    params=params,
                    output_name=derived_name,
                )
                click.echo(f"    Saved to {output_file}")
            except Exception as e:
                click.echo(f"  ERROR processing derived {derived_name}: {e}", err=True)
                continue
        
        else:
            # Expression feature (traditional)
            # Format: input_file\toutput_dir\tfeature_type
            input_file, output_dir, feature_type = parts[0], parts[1], parts[2]
            click.echo(f"  Processing expression: {input_file} -> {output_dir} ({feature_type})")
            
            try:
                # Create output directory
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # Load AnnData
                adata = sc.read_h5ad(input_file)
                
                # Extract features
                features = extract_celltype_features(adata, data_type=feature_type)
                
                # Save features
                output_file = Path(output_dir) / f"features_{feature_type}.npy"
                np.save(output_file, features)
                click.echo(f"    Saved features to {output_file} (shape: {features.shape})")
            except Exception as e:
                click.echo(f"  ERROR processing expression {feature_type}: {e}", err=True)
                continue
    
    click.echo(f"✓ Completed {len(jobs)} feature jobs")


@cli.command('run-graph')
@click.option('--chunk-file', '-f', required=True, type=click.Path(exists=True),
              help='Chunk file with graph job parameters')
def run_graph(chunk_file):
    """Worker for graph construction (processes chunk of graph jobs)."""
    from ..uge.job_generator import read_job_list
    from ..features.graphs import build_knn_graph_from_file
    from ..clustering.phenograph import compute_jaccard_graph
    import numpy as np
    
    jobs = read_job_list(chunk_file)
    click.echo(f"Processing {len(jobs)} graph jobs from chunk")
    
    for job in jobs:
        parts = job.split('\t')
        if len(parts) >= 7 and parts[0] == 'knn_phenograph':
            _, input_file, fel_output, pg_output, k, metric, k_jaccard = parts[:7]
            k = int(k)
            k_jaccard = int(k_jaccard)

            click.echo(
                f"  Building kNN + PhenoGraph graph: {input_file} "
                f"-> {fel_output}, {pg_output} (k={k}, metric={metric}, k_jaccard={k_jaccard})"
            )

            Path(fel_output).parent.mkdir(parents=True, exist_ok=True)
            Path(pg_output).parent.mkdir(parents=True, exist_ok=True)

            # Base kNN graph (Leiden input)
            edge_list = build_knn_graph_from_file(input_file, k=k, metric=metric)
            np.save(fel_output, edge_list)

            # PhenoGraph weighted graph (computed once in graph step)
            jaccard_edges, jaccard_weights = compute_jaccard_graph(
                edge_list=edge_list,
                k_jaccard=k_jaccard,
            )
            n_nodes = int(edge_list.max()) + 1 if edge_list.size > 0 else 0
            np.savez(
                pg_output,
                edges=jaccard_edges,
                weights=jaccard_weights,
                n_nodes=n_nodes,
                k_jaccard=k_jaccard,
                source_fel=fel_output,
            )

        elif len(parts) >= 5 and parts[0] == 'knn':
            _, input_file, output_file, k, metric = parts[:5]
            k = int(k)
            click.echo(f"  Building kNN graph: {input_file} -> {output_file} (k={k}, metric={metric})")

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            edge_list = build_knn_graph_from_file(input_file, k=k, metric=metric)
            np.save(output_file, edge_list)

        elif len(parts) >= 4:
            # Legacy format: input_file\\toutput_file\\tk\\tmetric
            input_file, output_file, k, metric = parts[0], parts[1], int(parts[2]), parts[3]
            click.echo(f"  Building graph: {input_file} -> {output_file} (k={k}, metric={metric})")

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            edge_list = build_knn_graph_from_file(input_file, k=k, metric=metric)
            np.save(output_file, edge_list)
        else:
            click.echo(f"  Skipping malformed job: {job}", err=True)
            continue
    
    click.echo(f"✓ Completed {len(jobs)} graph jobs")


@cli.command('run-clustering')
@click.option('--chunk-file', '-f', required=True, type=click.Path(exists=True),
              help='Chunk file with job parameters. Supports two formats:\n'
                   'v1 (4 fields): input_file\\toutput_dir\\tres_idx\\tsections_file\n'
                   'v2 (5 fields): method\\tinput_file\\toutput_dir\\tres_idx\\tsections_file')
def run_clustering(chunk_file):
    """Worker for clustering (processes chunk of Leiden or PhenoGraph jobs)."""
    from ..uge.job_generator import read_job_list
    from ..clustering.leiden import run_leiden_on_file
    from ..clustering.phenograph import run_phenograph_on_file
    
    jobs = read_job_list(chunk_file)
    click.echo(f"Processing {len(jobs)} clustering jobs from chunk")
    
    for job in jobs:
        parts = job.split('\t')
        
        # Detect format: 5 fields = v2 (method included), 4 fields = v1 (Leiden default)
        if len(parts) >= 5:
            method, input_file, output_dir, res_idx, sections_file = (
                parts[0], parts[1], parts[2], int(parts[3]), parts[4]
            )
        elif len(parts) >= 4:
            method = 'leiden'  # backward compatible default
            input_file, output_dir, res_idx, sections_file = (
                parts[0], parts[1], int(parts[2]), parts[3]
            )
        else:
            click.echo(f"  Skipping malformed job: {job}", err=True)
            continue
        
        # Get resolution value for display
        from ..clustering.base import get_resolution_values, format_resolution_dirname
        resolution_values = get_resolution_values()  # Uses default n_resolutions=50
        resolution = resolution_values[res_idx]
        res_dirname = format_resolution_dirname(resolution)
        
        click.echo(f"  {method}: {input_file} -> {output_dir}/{res_dirname}")
        
        # Dispatch to the right clustering method
        if method == 'leiden':
            run_leiden_on_file(
                input_npy=input_file,
                output_folder=output_dir,
                resolution_idx=res_idx,
                sections_path=sections_file,
            )
        elif method == 'phenograph':
            run_phenograph_on_file(
                input_npy=input_file,
                output_folder=output_dir,
                resolution_idx=res_idx,
                sections_path=sections_file,
            )
        else:
            click.echo(f"  Unknown clustering method: {method}", err=True)
            continue
        
        click.echo(f"    Saved to {output_dir}/{res_dirname}")
    
    click.echo(f"Completed {len(jobs)} clustering jobs")


@cli.command('run-percolation')
@click.option('--chunk-file', '-f', required=True, type=click.Path(exists=True),
              help='Chunk file with job parameters (type_file\\toutput_dir\\tedge_list_dir\\tmax_k\\txy_data_dir\\ttype_data_dir)')
def run_percolation(chunk_file):
    """Worker for running percolation analysis on type vectors (creates .npz and .score files)."""
    from ..uge.job_generator import read_job_list
    from ..percolation.graph_percolation import GraphPercolation, EdgeListManager
    import numpy as np
    
    jobs = read_job_list(chunk_file)
    click.echo(f"Processing {len(jobs)} percolation jobs from chunk")
    
    processed = 0
    
    for job in jobs:
        parts = job.split('\t')
        if len(parts) < 6:
            click.echo(f"  Skipping malformed job: {job}", err=True)
            continue
        
        type_file_rel = parts[0]
        output_dir = Path(parts[1])
        edge_list_dir = Path(parts[2])
        max_k = int(parts[3])
        xy_data_dir = Path(parts[4])
        type_data_dir = Path(parts[5])
        
        try:
            # Full path to type vector file
            type_file = type_data_dir / type_file_rel
            
            if not type_file.exists():
                click.echo(f"  Skipping missing file: {type_file}")
                continue
            
            # Load type vector
            type_vec = np.load(type_file)
            
            # Load XY coordinates by section name
            section_name = type_file.stem  # Get filename without extension
            xy_file = xy_data_dir / f"{section_name}_XY.npy"
            
            if not xy_file.exists():
                click.echo(f"  Skipping (no XY file): {type_file_rel}")
                click.echo(f"    Expected: {xy_file}")
                continue
            
            XY = np.load(xy_file)
            
            # Verify XY and type_vec have same length
            if len(XY) != len(type_vec):
                click.echo(f"  Skipping (XY/type mismatch): {type_file_rel}")
                click.echo(f"    XY shape: {XY.shape}, type_vec length: {len(type_vec)}")
                continue
            
            # Initialize edge list manager
            edge_list_dir.mkdir(parents=True, exist_ok=True)
            elm = EdgeListManager(base_dir=str(edge_list_dir))
            
            # Run percolation
            gp = GraphPercolation(XY, type_vec, maxK=max_k)
            gp.percolation(edge_list_manager=elm, xy_name=type_file.stem)
            
            # Create output path matching input structure
            output_file = output_dir / type_file_rel
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save full results
            gp.save(str(output_file))
            
            # Save raw + normalized scores separately for efficient aggregation
            score = gp.raw_score()
            normalized_score = gp.normalized_score()
            score_file = output_file.with_suffix('.score')
            with open(score_file, 'w') as f:
                f.write(f"{score}\t{normalized_score}\n")
            
            click.echo(
                f"  Processed: {type_file_rel} "
                f"(raw_score: {score:.4f}, normalized_score: {normalized_score:.4f})"
            )
            processed += 1
            
        except Exception as e:
            click.echo(f"  ERROR processing {type_file_rel}: {e}", err=True)
            continue
    
    click.echo(f"✓ Completed {processed}/{len(jobs)} percolation jobs")


@cli.command('run-aggregation')
@click.option('--chunk-file', '-f', required=True, type=click.Path(exists=True),
              help='Chunk file with job parameters (results_dir\\toutput_file\\tpattern[\\txy_dir\\treduced_output])')
@click.option('--reduce/--no-reduce', default=False,
              help='Automatically run reduce-scores after aggregation (requires xy_dir in job params)')
def run_aggregation(chunk_file, reduce):
    """Worker for score aggregation (processes chunk of aggregation jobs).
    
    Job format: results_dir\\toutput_file\\tpattern[\\txy_dir\\treduced_output]
    - First 3 fields are required for aggregation
    - Last 2 fields (xy_dir, reduced_output) are optional and used if --reduce is set
    """
    from ..uge.job_generator import read_job_list
    from ..percolation.analysis import aggregate_score_files_to_csv, reduce_scores_by_section
    
    jobs = read_job_list(chunk_file)
    click.echo(f"Processing {len(jobs)} aggregation jobs from chunk")
    
    for job in jobs:
        parts = job.split('\t')
        if len(parts) < 3:
            click.echo(f"  Skipping malformed job: {job}", err=True)
            continue
        
        results_dir, output_file, pattern = parts[0], parts[1], parts[2]
        click.echo(f"  Aggregating: {results_dir} -> {output_file}")
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        aggregate_score_files_to_csv(results_dir, output_file)
        
        # Optionally run reduce step
        if reduce and len(parts) >= 5:
            xy_dir, reduced_output = parts[3], parts[4]
            if Path(xy_dir).exists() and Path(output_file).exists():
                click.echo(f"  Reducing: {output_file} -> {reduced_output}")
                try:
                    reduce_scores_by_section(output_file, xy_dir, reduced_output)
                except Exception as e:
                    click.echo(f"  WARNING: Reduce failed: {e}", err=True)
            else:
                click.echo(f"  Skipping reduce: missing xy_dir or aggregated scores", err=True)
    
    click.echo(f"✓ Completed {len(jobs)} aggregation jobs")


if __name__ == '__main__':
    cli()
