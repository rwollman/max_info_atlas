#!/bin/bash
set -e

# ==============================================================================
# UGE Pipeline for Max Info Atlases
# ==============================================================================
#
# ARCHITECTURE OVERVIEW
# ---------------------
# This pipeline uses UGE job dependencies (--hold-jid) to chain jobs:
#
#   Step 1 (Features) → Step 2 (Graph) → Step 3 (Clustering)
#        ↓ (wait for outputs to scan)
#   Step 4 (Percolation)
#        ↓ (wait for outputs to scan)
#   Step 5 (Aggregation)
#
# Steps 1-3 are submitted immediately with UGE dependencies - no waiting needed!
# Steps 4-5 require scanning the outputs of previous steps, so we wait for
# Step 3 to complete before submitting Step 4, and wait for Step 4 before Step 5.
#
# Each step uses a uniform pattern:
#   1. CREATE-LIST: Generate a job list file (one line = one unit of work)
#   2. CHUNK:       Split job list into chunks (chunk-size controls parallelism)
#   3. SUBMIT:      Submit UGE array job with step-specific worker + --hold-jid
#
# PARAMETER COMBINATIONS
# ----------------------
# Steps 1-3 (Features, Graph, Clustering) DEFINE parameter combinations:
#   - Features:   Can specify multiple feature types (pca15, pca30, pca50)
#   - Graph:      Can specify multiple k values and metrics (cosine, correlation)
#   - Clustering: Can specify multiple resolution indices (0, 10, 25, 40, 49)
#   
# Steps 4-5 (Percolation, Aggregation) DISCOVER files:
#   - Percolation: Scans clustering output directory for all .npy files
#   - Aggregation: Scans percolation output directory for all .score files
#
# CHUNK SIZE = PARALLELISM CONTROL
# --------------------------------
#   - chunk-size >= total_jobs  → Single UGE job (all work in one task)
#   - chunk-size = 1            → Maximum parallelism (one task per job line)
#   - chunk-size = N            → N job lines per UGE task
#
#   For small test datasets: use large chunk-size (single jobs)
#   For production:          use smaller chunk-size to parallelize
#
# USAGE
# -----
#   ./tests/run_uge_pipeline.sh           # Run full pipeline
#   ./tests/run_uge_pipeline.sh --clean   # Clean results and re-run
#   ./tests/run_uge_pipeline.sh --dry-run # Show what would be submitted
#
# MONITORING
# ----------
# While the pipeline is running, you can check job status with:
#   qstat -u $USER                         # See all your jobs
#   max-info jobs monitor --job-id <id>    # Detailed monitoring for a job
#
# ==============================================================================

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Directories
TEST_DIR=/u/home/r/rwollman/project-rwollman/max_info_cart_v2/data/test_data
RESULTS_DIR=/u/home/r/rwollman/project-rwollman/max_info_cart_v2/data/test_results_uge
LOGS_DIR=/u/home/r/rwollman/project-rwollman/max_info_cart_v2/data/uge_logs
JOBS_DIR=/u/home/r/rwollman/project-rwollman/max_info_cart_v2/data/job_lists

# ------------------------------------------------------------------------------
# Pipeline Parameters
# ------------------------------------------------------------------------------
# These define what parameter combinations to run.
# Modify these to run different experiments.

# Feature extraction: which PCA dimensions to compute
# Can be comma-separated for multiple: "pca15,pca30,pca50"
FEATURE_TYPES="pca50"

# Graph construction: k-nearest neighbors and distance metric
# Can be comma-separated for multiple: "10,15,20" or "cosine,correlation"
K_NEIGHBORS=15
METRIC=correlation

# Clustering: which resolution indices to run
# Comma-separated list - each creates a separate clustering job
RESOLUTIONS="0,10,25,40,49"

# Percolation parameters (applied to ALL discovered type vectors)
MAX_K=100
XY_DIR="$TEST_DIR/xy_data"  # Directory containing XY coordinate files

# ------------------------------------------------------------------------------
# Chunk Sizes (Controls Parallelism)
# ------------------------------------------------------------------------------
# CHUNK_SIZE_SINGLE: Large value = all jobs in one UGE task (sequential)
# CHUNK_SIZE_PARALLEL: Small value = more UGE tasks (parallel)
#
# For this test script, we use single jobs. For production with thousands
# of files, decrease chunk size to parallelize across cluster nodes.

CHUNK_SIZE_SINGLE=1000000  # Effectively puts all jobs in one chunk
CHUNK_SIZE_PARALLEL=1      # One job per chunk = max parallelism

# ------------------------------------------------------------------------------
# Derived Paths (computed from parameters above)
# ------------------------------------------------------------------------------
N_FEATURES=$(echo $FEATURE_TYPES | cut -d',' -f1 | sed 's/pca//')
FEATURES_DIR="$RESULTS_DIR/features"
TYPE_DATA_DIR="$RESULTS_DIR/type_data/LeidenPCA${N_FEATURES}${METRIC^}"
PERC_OUTPUT_DIR="$RESULTS_DIR/percolation_results/LeidenPCA${N_FEATURES}${METRIC^}"
EDGE_LIST_DIR="$RESULTS_DIR/edge_lists"
FINAL_CSV="$RESULTS_DIR/percolation_scores.csv"

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

DRY_RUN=""
for arg in "$@"; do
    case $arg in
        --clean)
            echo "Cleaning previous results..."
            rm -rf "$RESULTS_DIR" "$LOGS_DIR" "$JOBS_DIR"
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            echo "DRY RUN MODE: Will show commands without submitting"
            ;;
    esac
done

# ==============================================================================
# VALIDATION
# ==============================================================================

if [ ! -f "$TEST_DIR/test_data.h5ad" ]; then
    echo "ERROR: Test dataset not found at $TEST_DIR/test_data.h5ad"
    echo ""
    echo "Create it first with:"
    echo "  python tests/create_test_dataset.py \\"
    echo "    --input /path/to/source.h5ad \\"
    echo "    --output-dir $TEST_DIR"
    exit 1
fi

mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$JOBS_DIR"

# ==============================================================================
# PIPELINE START
# ==============================================================================

echo "=============================================================="
echo "Max Info Atlases - UGE Pipeline"
echo "=============================================================="
echo ""
echo "Configuration:"
echo "  Test data:      $TEST_DIR"
echo "  Results:        $RESULTS_DIR"
echo "  Logs:           $LOGS_DIR"
echo "  Job lists:      $JOBS_DIR"
echo ""
echo "Parameters:"
echo "  Feature types:  $FEATURE_TYPES"
echo "  k neighbors:    $K_NEIGHBORS"
echo "  Metric:         $METRIC"
echo "  Resolutions:    $RESOLUTIONS"
echo ""

# ==============================================================================
# STEP 1: FEATURE EXTRACTION
# ==============================================================================
# This step DEFINES what features to compute.
# Job list contains: input_file, output_dir, feature_type
#
# For multiple feature types (e.g., "pca15,pca30,pca50"), create-list
# generates one job line per type.
# ==============================================================================

echo "=== Step 1: Feature Extraction ==="
echo "Creating job list for feature types: $FEATURE_TYPES"

# 1a: Create job list
#     --step features: tells create-list to expect features parameters
#     --feature-type: can be comma-separated for multiple types
max-info jobs create-list --step features \
    --input-file "$TEST_DIR/test_data.h5ad" \
    --output-dir "$FEATURES_DIR" \
    --feature-type "$FEATURE_TYPES" \
    --output "$JOBS_DIR/features_jobs.txt"

echo "Job list created. Contents:"
cat "$JOBS_DIR/features_jobs.txt" | head -5
echo ""

# 1b: Chunk the job list
#     Large chunk-size = single job (all feature types in one task)
max-info jobs chunk \
    --job-list "$JOBS_DIR/features_jobs.txt" \
    --chunks-dir "$JOBS_DIR/features_chunks" \
    --chunk-size $CHUNK_SIZE_SINGLE

# 1c: Submit array job
#     --worker-command: must match the --step used in create-list
#     run-features reads chunk file and processes each line
echo "Submitting UGE job..."
SUBMIT_OUT=$(max-info jobs submit \
    --chunks-dir "$JOBS_DIR/features_chunks" \
    --base-dir "$TEST_DIR" \
    --temp-dir "$RESULTS_DIR/temp" \
    --logs-dir "$LOGS_DIR" \
    --job-name features \
    --memory 8G \
    --runtime 1:00:00 \
    --worker-command 'max-info run-features --chunk-file "$chunk_file"' \
    $DRY_RUN)

echo "$SUBMIT_OUT"

# Extract job ID for dependency chain
JOB1_ID=$(echo "$SUBMIT_OUT" | grep -oP 'Submitted array job: \K\d+' || echo "$SUBMIT_OUT" | grep -oP '\d+' | head -1)
echo "Step 1 submitted: Job $JOB1_ID"
echo ""

# ==============================================================================
# STEP 2: GRAPH CONSTRUCTION
# ==============================================================================
# This step DEFINES what graphs to build.
# Job list contains: input_file, output_file, k, metric
#
# For multiple k values or metrics, create-list generates all combinations.
# Example: k=[10,15] × metrics=[cosine,correlation] = 4 job lines
# ==============================================================================

echo "=== Step 2: Graph Construction ==="
echo "Building k-NN graph with k=$K_NEIGHBORS, metric=$METRIC"

# 2a: Create job list
#     For multiple graphs, use comma-separated --k and --metric values
#     Note: --output-file is actually a directory; filenames are generated as FEL_k{k}_{metric}.npy
max-info jobs create-list --step graph \
    --input-file "$FEATURES_DIR/features_pca${N_FEATURES}.npy" \
    --output-file "$TYPE_DATA_DIR" \
    --k $K_NEIGHBORS \
    --metric $METRIC \
    --output "$JOBS_DIR/graph_jobs.txt"

echo "Job list created. Contents:"
cat "$JOBS_DIR/graph_jobs.txt" | head -5
echo ""

# 2b: Chunk
max-info jobs chunk \
    --job-list "$JOBS_DIR/graph_jobs.txt" \
    --chunks-dir "$JOBS_DIR/graph_chunks" \
    --chunk-size $CHUNK_SIZE_SINGLE

# 2c: Submit with dependency on Step 1
echo "Submitting UGE job (depends on job $JOB1_ID)..."
HOLD_JID_OPT=""
if [ -n "$JOB1_ID" ] && [ -z "$DRY_RUN" ]; then
    HOLD_JID_OPT="--hold-jid $JOB1_ID"
fi

SUBMIT_OUT=$(max-info jobs submit \
    --chunks-dir "$JOBS_DIR/graph_chunks" \
    --base-dir "$FEATURES_DIR" \
    --temp-dir "$RESULTS_DIR/temp" \
    --logs-dir "$LOGS_DIR" \
    --job-name graph \
    --memory 8G \
    --runtime 1:00:00 \
    --worker-command 'max-info run-graph --chunk-file "$chunk_file"' \
    $HOLD_JID_OPT \
    $DRY_RUN)

echo "$SUBMIT_OUT"

# Extract job ID for dependency chain
JOB2_ID=$(echo "$SUBMIT_OUT" | grep -oP 'Submitted array job: \K\d+' || echo "$SUBMIT_OUT" | grep -oP '\d+' | head -1)
echo "Step 2 submitted: Job $JOB2_ID (depends on $JOB1_ID)"
echo ""

# ==============================================================================
# STEP 3: CLUSTERING
# ==============================================================================
# This step DEFINES what resolutions to cluster at.
# Job list contains: input_file, output_dir, resolution_idx, sections_file
#
# For 5 resolutions (0,10,25,40,49), create-list generates 5 job lines.
# Each creates type vectors in: output_dir/res_X/section_name.npy
# ==============================================================================

echo "=== Step 3: Leiden Clustering ==="
echo "Clustering at resolution indices: $RESOLUTIONS"

# 3a: Create job list (one line per resolution)
#     Input is the edge list file created by step 2
max-info jobs create-list --step clustering \
    --input-file "$TYPE_DATA_DIR/FEL_k${K_NEIGHBORS}_${METRIC}.npy" \
    --output-dir "$TYPE_DATA_DIR" \
    --sections-file "$TEST_DIR/Sections.npy" \
    --resolutions "$RESOLUTIONS" \
    --output "$JOBS_DIR/clustering_jobs.txt"

echo "Job list created. Contents:"
cat "$JOBS_DIR/clustering_jobs.txt"
echo ""

# 3b: Chunk
#     With chunk-size=1000000, all 5 resolutions run in one UGE task
#     With chunk-size=1, each resolution gets its own UGE task (parallel)
max-info jobs chunk \
    --job-list "$JOBS_DIR/clustering_jobs.txt" \
    --chunks-dir "$JOBS_DIR/clustering_chunks" \
    --chunk-size $CHUNK_SIZE_SINGLE

# 3c: Submit with dependency on Step 2
echo "Submitting UGE job (depends on job $JOB2_ID)..."
HOLD_JID_OPT=""
if [ -n "$JOB2_ID" ] && [ -z "$DRY_RUN" ]; then
    HOLD_JID_OPT="--hold-jid $JOB2_ID"
fi

SUBMIT_OUT=$(max-info jobs submit \
    --chunks-dir "$JOBS_DIR/clustering_chunks" \
    --base-dir "$TYPE_DATA_DIR" \
    --temp-dir "$RESULTS_DIR/temp" \
    --logs-dir "$LOGS_DIR" \
    --job-name clustering \
    --memory 16G \
    --runtime 2:00:00 \
    --worker-command 'max-info run-clustering --chunk-file "$chunk_file"' \
    $HOLD_JID_OPT \
    $DRY_RUN)

echo "$SUBMIT_OUT"

# Extract job ID for dependency chain
JOB3_ID=$(echo "$SUBMIT_OUT" | grep -oP 'Submitted array job: \K\d+' || echo "$SUBMIT_OUT" | grep -oP '\d+' | head -1)
echo "Step 3 submitted: Job $JOB3_ID (depends on $JOB2_ID)"
echo ""

# ------------------------------------------------------------------------------
# WAIT FOR STEP 3: Percolation needs to scan clustering outputs
# ------------------------------------------------------------------------------
# Step 4 creates its job list by scanning the output of Step 3.
# We must wait for Step 3 to complete before we can create the job list.
# This is the ONLY place we wait - all other dependencies use --hold-jid.
echo "=== Waiting for Steps 1-3 to complete ==="
echo "Jobs $JOB1_ID -> $JOB2_ID -> $JOB3_ID are queued with dependencies."
echo "Waiting for job $JOB3_ID to finish so we can discover clustering outputs..."
echo ""

if [ -z "$DRY_RUN" ]; then
    # Simple wait loop using qstat
    while true; do
        # Check if job is still in queue
        if ! qstat -j "$JOB3_ID" > /dev/null 2>&1; then
            echo "Job $JOB3_ID completed."
            break
        fi
        
        # Show status
        STATUS=$(qstat -u "$USER" 2>/dev/null | grep "$JOB3_ID" | awk '{print $5}' | head -1)
        echo "  $(date '+%Y-%m-%d %H:%M:%S') - Job $JOB3_ID status: ${STATUS:-checking...}"
        sleep 30
    done
    
    # Brief pause for NFS
    echo "Waiting for NFS to sync..."
    sleep 5
fi

# Verify Step 3 outputs
CLUSTERING_COUNT=$(find "$TYPE_DATA_DIR" -name "res_*" -type d 2>/dev/null | wc -l)
if [ "$CLUSTERING_COUNT" -eq 0 ] && [ -z "$DRY_RUN" ]; then
    echo ""
    echo "ERROR: Step 3 failed - no clustering output directories found in: $TYPE_DATA_DIR"
    echo "Check UGE logs at: $LOGS_DIR/clustering.*.log"
    exit 1
fi
echo "Found $CLUSTERING_COUNT resolution directories in $TYPE_DATA_DIR"
echo ""

# ==============================================================================
# STEP 4: PERCOLATION ANALYSIS
# ==============================================================================
# This step DISCOVERS files created by clustering.
# It scans TYPE_DATA_DIR for all .npy files matching pattern "res_*/*.npy"
# and creates one job line per file found.
#
# No parameter combinations here - just processes whatever exists.
# Output: .npz (full results) and .score (just the score) files
# ==============================================================================

echo "=== Step 4: Percolation Analysis ==="
echo "Scanning for type vectors in: $TYPE_DATA_DIR"

# 4a: Create job list by scanning directory
#     --pattern: glob pattern to find type vector files
#     Creates one job per file found
max-info jobs create-list --step percolation \
    --type-data-dir "$TYPE_DATA_DIR" \
    --output-dir "$PERC_OUTPUT_DIR" \
    --edge-list-dir "$EDGE_LIST_DIR" \
    --xy-data-dir "$XY_DIR" \
    --pattern "res_*/*.npy" \
    --max-k $MAX_K \
    --output "$JOBS_DIR/percolation_jobs.txt"

echo "Job list created. Found $(wc -l < "$JOBS_DIR/percolation_jobs.txt") type vectors"
echo "First few jobs:"
head -5 "$JOBS_DIR/percolation_jobs.txt"
echo ""

# 4b: Chunk
#     For large datasets, use smaller chunk-size to parallelize
#     Example: 10000 files with chunk-size=100 = 100 parallel UGE tasks
max-info jobs chunk \
    --job-list "$JOBS_DIR/percolation_jobs.txt" \
    --chunks-dir "$JOBS_DIR/percolation_chunks" \
    --chunk-size $CHUNK_SIZE_SINGLE

# 4c: Submit
#     All parameters are in the chunk file, so worker only needs chunk-file
#     No dependency needed here since we already waited for Step 3
echo "Submitting UGE job..."
SUBMIT_OUT=$(max-info jobs submit \
    --chunks-dir "$JOBS_DIR/percolation_chunks" \
    --base-dir "$TYPE_DATA_DIR" \
    --temp-dir "$RESULTS_DIR/temp" \
    --logs-dir "$LOGS_DIR" \
    --job-name percolation \
    --memory 32G \
    --runtime 4:00:00 \
    --worker-command "max-info run-percolation --chunk-file \"\$chunk_file\"" \
    $DRY_RUN)

echo "$SUBMIT_OUT"

# Extract job ID for dependency chain
JOB4_ID=$(echo "$SUBMIT_OUT" | grep -oP 'Submitted array job: \K\d+' || echo "$SUBMIT_OUT" | grep -oP '\d+' | head -1)
echo "Step 4 submitted: Job $JOB4_ID"
echo ""

# ==============================================================================
# STEP 5: SCORE AGGREGATION
# ==============================================================================
# This step DISCOVERS .score files created by percolation.
# It scans PERC_OUTPUT_DIR for all .score files and combines them into
# a single CSV with columns: resolution, section, score, etc.
#
# The .score files are lightweight (just the percolation score) for
# efficient aggregation. Full results are in .npz files.
# ==============================================================================

# ------------------------------------------------------------------------------
# WAIT FOR STEP 4: Aggregation needs to scan percolation outputs
# ------------------------------------------------------------------------------
echo "=== Waiting for Step 4 (Percolation) to complete ==="
echo "Waiting for job $JOB4_ID to finish so we can discover score files..."
echo ""

if [ -z "$DRY_RUN" ]; then
    # Simple wait loop using qstat
    while true; do
        # Check if job is still in queue
        if ! qstat -j "$JOB4_ID" > /dev/null 2>&1; then
            echo "Job $JOB4_ID completed."
            break
        fi
        
        # Show status
        STATUS=$(qstat -u "$USER" 2>/dev/null | grep "$JOB4_ID" | awk '{print $5}' | head -1)
        echo "  $(date '+%Y-%m-%d %H:%M:%S') - Job $JOB4_ID status: ${STATUS:-checking...}"
        sleep 30
    done
    
    # Brief pause for NFS
    echo "Waiting for NFS to sync..."
    sleep 5
fi

# Verify Step 4 outputs
SCORE_COUNT=$(find "$PERC_OUTPUT_DIR" -name "*.score" 2>/dev/null | wc -l)
if [ "$SCORE_COUNT" -eq 0 ] && [ -z "$DRY_RUN" ]; then
    echo ""
    echo "ERROR: Step 4 failed - no .score files found in: $PERC_OUTPUT_DIR"
    echo "Check UGE logs at: $LOGS_DIR/percolation.*.log"
    exit 1
fi
echo "Found $SCORE_COUNT score files in $PERC_OUTPUT_DIR"
echo ""

echo "=== Step 5: Score Aggregation ==="
echo "Scanning for score files in: $PERC_OUTPUT_DIR"

# 5a: Create job list (usually just one job for aggregation)
max-info jobs create-list --step aggregation \
    --results-dir "$PERC_OUTPUT_DIR" \
    --output-file "$FINAL_CSV" \
    --score-pattern "**/*.score" \
    --output "$JOBS_DIR/aggregation_jobs.txt"

echo "Job list created. Contents:"
cat "$JOBS_DIR/aggregation_jobs.txt"
echo ""

# 5b: Chunk
max-info jobs chunk \
    --job-list "$JOBS_DIR/aggregation_jobs.txt" \
    --chunks-dir "$JOBS_DIR/aggregation_chunks" \
    --chunk-size $CHUNK_SIZE_SINGLE

# 5c: Submit (no dependency needed since we waited for Step 4)
echo "Submitting UGE job..."
SUBMIT_OUT=$(max-info jobs submit \
    --chunks-dir "$JOBS_DIR/aggregation_chunks" \
    --base-dir "$PERC_OUTPUT_DIR" \
    --temp-dir "$RESULTS_DIR/temp" \
    --logs-dir "$LOGS_DIR" \
    --job-name aggregation \
    --memory 8G \
    --runtime 1:00:00 \
    --worker-command 'max-info run-aggregation --chunk-file "$chunk_file"' \
    $DRY_RUN)

echo "$SUBMIT_OUT"

# Extract job ID
JOB5_ID=$(echo "$SUBMIT_OUT" | grep -oP 'Submitted array job: \K\d+' || echo "$SUBMIT_OUT" | grep -oP '\d+' | head -1)
echo "Step 5 submitted: Job $JOB5_ID"
echo ""

# Wait for Step 5 (final step) to show results
echo "=== Waiting for Step 5 (Aggregation) to complete ==="
if [ -z "$DRY_RUN" ]; then
    while true; do
        if ! qstat -j "$JOB5_ID" > /dev/null 2>&1; then
            echo "Job $JOB5_ID completed."
            break
        fi
        STATUS=$(qstat -u "$USER" 2>/dev/null | grep "$JOB5_ID" | awk '{print $5}' | head -1)
        echo "  $(date '+%Y-%m-%d %H:%M:%S') - Job $JOB5_ID status: ${STATUS:-checking...}"
        sleep 15
    done
    sleep 5  # NFS sync
fi

# Validate Step 5 outputs
if [ ! -f "$FINAL_CSV" ] && [ -z "$DRY_RUN" ]; then
    echo ""
    echo "ERROR: Step 5 failed - final CSV not found: $FINAL_CSV"
    echo "Check UGE logs at: $LOGS_DIR/aggregation.*.log"
    exit 1
fi
echo "Step 5 complete"
echo ""

# ==============================================================================
# SUMMARY
# ==============================================================================

echo "=============================================================="
echo "Pipeline Complete!"
echo "=============================================================="
echo ""
echo "Output locations:"
echo "  Job lists:      $JOBS_DIR/"
echo "  Features:       $FEATURES_DIR/"
echo "  Clustering:     $TYPE_DATA_DIR/"
echo "  Percolation:    $PERC_OUTPUT_DIR/"
echo "  Final CSV:      $FINAL_CSV"
echo "  Logs:           $LOGS_DIR/"
echo ""

if [ -z "$DRY_RUN" ] && [ -f "$FINAL_CSV" ]; then
    echo "Results preview:"
    echo "----------------"
    head -20 "$FINAL_CSV"
    echo ""
    echo "Total rows: $(wc -l < "$FINAL_CSV")"
fi

echo ""
echo "=============================================================="
echo "To customize this pipeline:"
echo "  1. Modify FEATURE_TYPES, K_NEIGHBORS, METRIC, RESOLUTIONS"
echo "  2. For parallelism, change CHUNK_SIZE_SINGLE to smaller values"
echo "  3. For production, use CHUNK_SIZE_PARALLEL=1 for percolation step"
echo "=============================================================="
