# Max Info Atlases v2

Refactored pipeline for cell type clustering and percolation analysis.

## Installation

```bash
cd /u/home/r/rwollman/project-rwollman/max_info_cart_v2/code

# Install build dependencies first
pip install numpy cython

# Install package in editable mode (compiles Cython extensions automatically)
pip install -e .
```

### Verify Installation

After installation, verify the Cython module compiled successfully:
```bash
python -c "from max_info_atlases.cython import ConnectedComponentEntropy; print('✓ Cython module loaded successfully!')"
```

### Cython Module (for percolation analysis)

The percolation module includes the `ConnectedComponentEntropy` Cython extension, which is automatically compiled during installation.

**Requirements:**
- Cython >= 0.29
- A C compiler (gcc/clang)
- NumPy >= 1.20

If you encounter compilation issues, ensure you have the build dependencies:
```bash
pip install numpy cython
```

## Compatibility with v1 Data

This package is fully compatible with data created by v1. You can:

- Parse existing folder structures with `path_utils.parse_path()`
- Calculate scores from existing `.npz` files with `percolation.analysis.calculate_percolation_score()`
- Use `extract_metadata_from_path()` for backward-compatible dict format

```python
from max_info_atlases.path_utils import parse_path, extract_metadata_from_path
from max_info_atlases.percolation.analysis import calculate_percolation_score

# Parse v1 paths
metadata = parse_path('/path/to/v1/LeidenRawCorrelation_envenv_w_k_20/res_1p000/section.npy', '/path/to/v1')

# Calculate score from v1 .npz file
score, timing = calculate_percolation_score('/path/to/v1/results/section.npz')
```

## Quick Start

### Feature Extraction

```bash
# Cell type features from AnnData
max-info features celltype --input data.h5ad --output features/ --type pca

# Environment features (k-NN neighborhood composition)
max-info features env --input data.h5ad --k 50 --weighted --output env_k_50.csv

# Build k-NN graph for clustering
max-info features graph --input features.csv --output FEL.npy --k 15 --metric cosine
```

### Clustering

```bash
# Leiden clustering on graph
max-info cluster leiden --input FEL.npy --resolution 1.0 --output clusters/

# K-means clustering on features
max-info cluster kmeans --input env.csv --k 100 --pca 50 --output clusters/

# LDA clustering on features
max-info cluster lda --input env.csv --n-topics 50 --output clusters/
```

### UGE Job Management (Hoffman2)

```bash
# 1. Prepare job chunks
max-info jobs prepare \
    --input-dir /path/to/type_vectors \
    --chunks-dir chunked_jobs/ \
    --chunk-size 6000

# 2. Submit array job
max-info jobs submit \
    --chunks-dir chunked_jobs/ \
    --base-dir /path/to/data \
    --temp-dir /path/to/temp \
    --logs-dir /path/to/logs \
    --memory 16G \
    --runtime 4:00:00

# 3. Monitor with auto-resubmit
max-info jobs monitor \
    --job-id 12345 \
    --chunks-dir chunked_jobs/ \
    --temp-dir /path/to/temp \
    --base-dir /path/to/data \
    --logs-dir /path/to/logs \
    --timeout-minutes 60 \
    --auto-resubmit \
    --combine-on-complete \
    --output final_scores.csv
```

### Result Aggregation

```bash
# Combine chunked results
max-info results combine \
    --temp-dir /path/to/temp \
    --output final_scores.csv \
    --cleanup
```

## Package Structure

```
max_info_atlases/
├── path_utils.py      # Centralized path parsing
├── clustering/        # Clustering methods
│   ├── base.py        # Abstract base class
│   ├── leiden.py      # Leiden clustering
│   ├── kmeans.py      # K-means clustering
│   └── lda.py         # LDA clustering
├── features/          # Feature extraction
│   ├── celltype.py    # Cell type features
│   ├── environment.py # Environment features
│   └── graphs.py      # k-NN graph construction
├── percolation/       # Percolation analysis
│   ├── graph_percolation.py  # Core algorithm
│   └── analysis.py    # Score calculation & aggregation
├── uge/               # UGE job management
│   ├── job_generator.py  # Job list & chunking
│   ├── submit.py      # Job submission
│   ├── monitor.py     # Monitoring & auto-resubmit
│   └── templates.py   # UGE script templates
└── cli/               # Command-line interface
    └── main.py        # Click CLI entry point
```

## Configuration

Run configs (e.g., `config/small_cell_type_opt.yaml`, `config/local_frequency_example.yaml`) define all paths and parameters for a pipeline run.

## Migration from v1

The v2 package is designed to work alongside existing code:

1. **Folder structure unchanged**: Results still use the same folder-based metadata encoding
2. **Centralized parsing**: Use `path_utils.parse_path()` instead of scattered parsing code
3. **UGE templates**: Replace shell scripts with Python-generated templates
4. **CLI replaces scripts**: Use `max-info` CLI instead of individual Python scripts

## Running Tests

```bash
cd /purpledata/RoyTemp/max_info_atlases/v2
pip install -e ".[dev]"
pytest tests/
```
