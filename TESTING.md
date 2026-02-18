# Testing and Development Guide

## 1. Running Unit Tests

### Prerequisites

```bash
cd /u/home/r/rwollman/project-rwollman/max_info_cart_v2/code

# Install package with dev dependencies
pip install -e ".[dev]"

# Verify Cython module compiled successfully
python -c "from max_info_atlases.cython import ConnectedComponentEntropy; print('✓ Cython module loaded!')"
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Test path parsing
pytest tests/test_path_utils.py -v

# Test clustering methods
pytest tests/test_clustering.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=max_info_atlases --cov-report=html
# Open htmlcov/index.html in browser
```

---

## 2. End-to-End Test with Subsampled Data

### Quick Start: Automated Test Scripts

**Step 1: Create test dataset (one-time setup)**

```bash
cd /u/home/r/rwollman/project-rwollman/max_info_cart_v2/code

python tests/create_test_dataset.py \
    --input /u/project/rwollman/data/external/mouse_brain/allen_merscope.h5ad \
    --output-dir /u/home/r/rwollman/project-rwollman/max_info_cart_v2/data/test_data \
    --n-sections 3 \
    --cells-per-section 1000
```

**Step 2A: Run automated pipeline (LOCAL)**

```bash
# Make script executable (first time only)
chmod +x tests/run_end_to_end_test.sh

# Run the complete pipeline locally (no UGE)
bash tests/run_end_to_end_test.sh

# Or clean results and re-run
bash tests/run_end_to_end_test.sh --clean
```

**Step 2B: Run automated pipeline (UGE)**

```bash
# Make script executable (first time only)
chmod +x tests/run_uge_pipeline.sh

# Run the complete pipeline with UGE job submission
bash tests/run_uge_pipeline.sh

# Or clean results and re-run
bash tests/run_uge_pipeline.sh --clean
```

| Feature | Local Script | UGE Script |
|---------|-------------|------------|
| **Execution** | Sequential on current node | UGE job submission |
| **Monitoring** | None (blocking) | Uses `max-info jobs monitor` |
| **Job Management** | N/A | Uses `max-info jobs prepare/submit` |
| **Percolation** | Inline Python | Array job with `max-info run-percolation` |
| **Resources** | Uses current node | Requests cluster resources |
| **Good for** | Quick testing, development | Production, large datasets |
| **Requires** | Local compute | UGE/qsub access |

**What both scripts do:**
1. Validates test dataset exists
2. Extracts PCA50 features
3. Builds k-NN graph (k=15, correlation metric)
4. Runs Leiden clustering at 5 resolutions (0, 10, 25, 40, 49)
5. Runs percolation analysis on **all resolutions and all sections**
   - Saves `.npz` files (full results for archival)
   - Saves `.score` files (just the score for efficient aggregation)
6. Aggregates scores with `max-info results aggregate-scores` (reads lightweight .score files)
7. Prints comprehensive summary with all file locations

**Output:**
- Final results saved to: `$RESULTS_DIR/percolation_scores.csv`
- Contains: all combinations of resolutions × sections
- Each row: algorithm, resolution, section, percolation score, metadata
- **Efficient:** Uses .score files for aggregation (scalable to millions of results)

### UGE Pipeline Details

The UGE pipeline (`run_uge_pipeline.sh`) uses a **uniform pattern for ALL steps**:

```
create-list -> chunk -> submit -> monitor
```

**Step Pattern (same for all steps):**

```bash
# 1. Create job list for the step
max-info jobs create-list --step <step_type> --output jobs.txt ...

# 2. Chunk the job list (chunk-size controls parallelism)
max-info jobs chunk --job-list jobs.txt --chunks-dir chunks/ --chunk-size <N>

# 3. Submit array job with step-specific worker
max-info jobs submit --chunks-dir chunks/ --worker-command "max-info run-<step> ..."

# 4. Monitor until completion
max-info jobs monitor --job-id <JOB_ID> --chunks-dir chunks/ ...
```

**Chunk Size Controls Parallelism:**
- `chunk-size >= total_jobs` → Single job (no parallelism)
- `chunk-size = 1` → Maximum parallelism (one UGE task per job line)
- For test dataset: use large chunk size (single jobs)
- For production: decrease chunk size to parallelize

**Pipeline Steps and Workers:**

| Step | Job List Generator | Worker Command |
|------|-------------------|----------------|
| 1. Features | `--step features` | `max-info run-features` |
| 2. Graph | `--step graph` | `max-info run-graph` |
| 3. Clustering | `--step clustering` | `max-info run-clustering` |
| 4. Percolation | `--step percolation` | `max-info run-percolation` |
| 5. Aggregation | `--step aggregation` | `max-info run-aggregation` |

**Key max-info commands:**

| Command | Purpose |
|---------|---------|
| `max-info jobs create-list` | Create job list for a pipeline step |
| `max-info jobs chunk` | Split job list into chunks |
| `max-info jobs submit` | Submit UGE array job |
| `max-info jobs monitor` | Monitor job, detect stuck tasks, auto-resubmit |
| `max-info jobs status` | Quick status check |
| `max-info jobs calibrate` | Create calibration chunk for timing |
| `max-info run-<step>` | Step-specific worker (features, graph, clustering, percolation, aggregation) |

**Resource Allocation:**
- Small (8GB, 1hr): Features, graph, aggregation
- Medium (16GB, 2hr): Clustering
- Large (32GB, 4hr): Percolation

---

### Manual Steps (for reference)

If you prefer to run steps individually, here are the detailed commands:

#### Step 1: Create Subsampled Dataset

Create a small test dataset with ~3 sections and ~1000 cells each:

```bash
python tests/create_test_dataset.py \
    --input /u/project/rwollman/data/external/mouse_brain/allen_merscope.h5ad \
    --output-dir /purpledata/RoyTemp/max_info_atlases/v2/test_data \
    --n-sections 3 \
    --cells-per-section 1000
```

**Options:**
- `--n-sections`: Number of sections to include (default: 3)
- `--cells-per-section`: Maximum cells per section (default: 1000)
- `--section-column`: Column name in adata.obs for section IDs (default: 'brain_section_label')
- `--output-name`: Name for output h5ad file (default: 'test_allen_3sections.h5ad')
- `--random-seed`: Random seed for reproducibility (default: 42)

**Or use as a Python function:**

```python
from tests.create_test_dataset import create_test_dataset

output_file = create_test_dataset(
    input_h5ad='/u/project/rwollman/data/external/mouse_brain/allen_merscope.h5ad',
    output_dir='/purpledata/RoyTemp/max_info_atlases/v2/test_data',
    n_sections=3,
    cells_per_section=1000
)
```

#### Step 2: Run Feature Extraction

```bash
TEST_DIR=/u/home/r/rwollman/project-rwollman/max_info_cart_v2/data/test_data
RESULTS_DIR=/u/home/r/rwollman/project-rwollman/max_info_cart_v2/data/test_results

# Extract cell type features (PCA)
max-info features celltype \
    --input $TEST_DIR/test_allen_3sections.h5ad \
    --output $RESULTS_DIR/features \
    --type pca50

# Build k-NN graph
max-info features graph \
    --input $RESULTS_DIR/features/features_pca50.npy \
    --output $RESULTS_DIR/type_data/LeidenPCA50Correlation/FEL.npy \
    --k 15 \
    --metric correlation
```

#### Step 3: Run Clustering

```bash
# Leiden clustering at multiple resolutions
for res_idx in 0 10 25 40 49; do
    max-info cluster leiden \
        --input $RESULTS_DIR/type_data/LeidenPCA50Correlation/FEL.npy \
        --output $RESULTS_DIR/type_data/LeidenPCA50Correlation \
        --resolution-idx $res_idx \
        --sections $TEST_DIR/Sections.npy
done
```

#### Step 4: Run Percolation Analysis and Create Results Table

**Option A: Quick test (single resolution, print scores only)**

```python
import numpy as np
from pathlib import Path
from max_info_atlases.percolation.graph_percolation import GraphPercolation, EdgeListManager

results_dir = Path('/u/home/r/rwollman/project-rwollman/max_info_cart_v2/data/test_results')
type_dir = results_dir / 'type_data/LeidenPCA50Correlation/res_31p623'  # Change resolution as needed
section_files = sorted(type_dir.glob('*.npy'))

elm = EdgeListManager(base_dir=str(results_dir / 'edge_lists'))

for section_file in section_files:
    type_vec = np.load(section_file)
    XY = np.random.rand(len(type_vec), 2) * 1000  # Use real coordinates in production
    
    gp = GraphPercolation(XY, type_vec, maxK=100)
    gp.percolation(edge_list_manager=elm, xy_name=section_file.stem)
    print(f"{section_file.name}: {gp.score():.4f}")
```

**Option B: Full workflow with results table (recommended)**

```bash
RESULTS_DIR=/u/home/r/rwollman/project-rwollman/max_info_cart_v2/data/test_results
RESOLUTIONS=(0 10 25 40 49)  # All resolutions to process

# 1. Run percolation on all resolutions and save results
for res_idx in "${RESOLUTIONS[@]}"; do
    echo "Processing resolution $res_idx..."
    
    OUTPUT_DIR="$RESULTS_DIR/percolation_results/LeidenPCA50Correlation/res_${res_idx}"
    mkdir -p "$OUTPUT_DIR"
    
    # Process each section
    python -c "
import numpy as np
from pathlib import Path
from max_info_atlases.percolation.graph_percolation import GraphPercolation, EdgeListManager

results_dir = Path('$RESULTS_DIR')
type_dir = results_dir / 'type_data/LeidenPCA50Correlation/res_${res_idx}'
output_dir = Path('$OUTPUT_DIR')

elm = EdgeListManager(base_dir=str(results_dir / 'edge_lists'))

for section_file in sorted(type_dir.glob('*.npy')):
    type_vec = np.load(section_file)
    XY = np.random.rand(len(type_vec), 2) * 1000
    
    gp = GraphPercolation(XY, type_vec, maxK=100)
    gp.percolation(edge_list_manager=elm, xy_name=section_file.stem)
    
    # Save full results (archival)
    npz_file = output_dir / f'{section_file.stem}.npz'
    gp.save(str(npz_file))
    
    # Save just the score (efficient aggregation)
    score = gp.score()
    score_file = output_dir / f'{section_file.stem}.score'
    with open(score_file, 'w') as f:
        f.write(f'{score}\n')
"
done

# 2. Aggregate scores efficiently (reads .score files, not .npz!)
max-info results aggregate-scores \
    --base-dir $RESULTS_DIR/percolation_results \
    --output $RESULTS_DIR/percolation_scores.csv

# 3. View results
head -20 $RESULTS_DIR/percolation_scores.csv
```

**Output CSV columns:**
- `algorithm` - Clustering algorithm (e.g., 'Leiden')
- `data_type` - Feature type (e.g., 'PCA50')
- `metric` - Distance metric (e.g., 'Correlation')
- `resolution_idx` - Resolution index (0, 10, 25, 40, 49)
- `section_name` - Section identifier
- `percolation_score` - Computed percolation score
- `load_time_seconds` - Time to load .npz file
- `score_time_seconds` - Time to calculate score
- `total_time_seconds` - Total processing time
- `file_path` - Path to .npz file

**Result Summary:**
- For 5 resolutions × 3 sections = 15 rows total
- Each row contains scores and metadata for one section at one resolution
- Easy to analyze: group by resolution, compare across sections, etc.

**Why .score files?**
- **Memory efficient**: Each .score file is ~10 bytes vs .npz files which are KB-MB
- **Scalable**: Can aggregate millions of scores without loading large arrays
- **Distributed-friendly**: Perfect for UGE workflows where aggregation happens separately
- **Archival**: .npz files preserved for detailed analysis if needed

**Note:** Each resolution creates its own subdirectory (`res_0p100/`, `res_3p162/`, `res_31p623/`, etc.) with section files inside, where the directory name contains the actual resolution value with 'p' replacing the decimal point.

---

## 3. Calibrating Chunk Size with Test Job

Instead of guessing chunk sizes, run a single calibration job first to measure actual processing time.

### Step 1: Create Calibration Chunk

```bash
# Create a small calibration chunk with 10 files sampled from your data
max-info jobs calibrate \
    --input-dir /path/to/type_vectors \
    --output-dir /path/to/calibration \
    --n-files 10 \
    --pattern "**/*.npy"
```

This creates `calibration_chunk.txt` with 10 representative files.

### Step 2: Run Calibration Job

**Option A: Run locally (quick test)**
```bash
# Run locally to time processing
time max-info worker \
    --chunk-file /path/to/calibration/calibration_chunk.txt \
    --base-dir /path/to/type_vectors \
    --output-dir /path/to/calibration
```

**Option B: Submit to UGE (realistic timing)**
```bash
# Submit calibration as single UGE job
max-info jobs submit \
    --chunks-dir /path/to/calibration \
    --base-dir /path/to/type_vectors \
    --temp-dir /path/to/calibration/temp \
    --logs-dir /path/to/calibration/logs \
    --memory 16G \
    --runtime 1:00:00
```

### Step 3: Calculate Optimal Chunk Size

After the calibration job completes, note the time per file and calculate:

```bash
# Calculate chunk size for 1-hour target runtime
# Example: if 10 files took 1.5 seconds total = 0.15 seconds/file
max-info jobs calc-chunk-size \
    --time-per-file 0.15 \
    --target-runtime 1.0 \
    --safety-factor 0.8
```

Output:
```
=== Chunk Size Calculator ===
Time per file: 0.150 seconds
Files per hour: 24000
Target runtime: 1.0 hours
Safety factor: 80%

Recommended chunk size: 19200
Estimated runtime per chunk: 0.80 hours

Use with:
  max-info jobs prepare --chunk-size 19200 ...
```

### Step 4: Prepare Jobs with Calibrated Size

```bash
max-info jobs prepare \
    --input-dir /path/to/type_vectors \
    --chunks-dir chunked_jobs/ \
    --chunk-size 19200  # Use calculated value from Step 3
```

### Quick Reference: Typical Values

| File Type | Time/File | 1hr Chunk Size |
|-----------|-----------|----------------|
| Small .npy (percolation) | 0.1-0.2s | 15,000-25,000 |
| Medium .npz | 0.5-1.0s | 2,500-5,000 |
| Large analysis | 5-10s | 300-500 |

---

## Full Pipeline Script

**Note:** An automated end-to-end test script is available at `tests/run_end_to_end_test.sh`.

See the top of **Section 2** for usage instructions, or run:

```bash
bash tests/run_end_to_end_test.sh --clean
```

This script includes all steps from dataset creation through percolation analysis with proper error handling and a comprehensive summary at the end.
