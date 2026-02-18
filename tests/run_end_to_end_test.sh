#!/bin/bash
set -e  # Exit on any error

# ==============================================================================
# End-to-End Test Pipeline for Max Info Atlases
# ==============================================================================
# This script runs the complete pipeline from feature extraction to percolation
# analysis on a test dataset.
#
# Prerequisites:
#   - Test dataset must already exist in $TEST_DIR
#   - Run create_test_dataset.py first if needed
#
# Usage:
#   ./tests/run_end_to_end_test.sh [--clean]
#
# Options:
#   --clean    Remove all existing results before running
# ==============================================================================

# Configuration
TEST_DIR=/u/home/r/rwollman/project-rwollman/max_info_cart_v2/data/test_data
RESULTS_DIR=/u/home/r/rwollman/project-rwollman/max_info_cart_v2/data/test_results

# Test parameters
N_FEATURES=50  # PCA components
K_NEIGHBORS=15
METRIC=correlation
RESOLUTIONS=(0 10 25 40 49)  # Leiden resolution indices

# ==============================================================================
# Parse arguments
# ==============================================================================
CLEAN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--clean]"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Clean up if requested
# ==============================================================================
if [ "$CLEAN" = true ]; then
    echo "=== Cleaning up existing results ==="
    rm -rf "$RESULTS_DIR"
    echo "✓ Cleaned up"
    echo ""
fi

# ==============================================================================
# Validate test dataset exists
# ==============================================================================
if [ ! -f "$TEST_DIR/test_data.h5ad" ]; then
    echo "ERROR: Test dataset not found at $TEST_DIR/test_data.h5ad"
    echo ""
    echo "Please create the test dataset first using:"
    echo "  python tests/create_test_dataset.py \\"
    echo "    --input /u/project/rwollman/data/external/mouse_brain/allen_merscope.h5ad \\"
    echo "    --output-dir $TEST_DIR \\"
    echo "    --n-sections 3 \\"
    echo "    --cells-per-section 1000"
    echo ""
    exit 1
fi

if [ ! -f "$TEST_DIR/Sections.npy" ]; then
    echo "ERROR: Sections.npy not found at $TEST_DIR/Sections.npy"
    echo "Please ensure the test dataset was created properly."
    exit 1
fi

echo "✓ Test dataset found: $TEST_DIR/test_data.h5ad"
echo ""

# ==============================================================================
# Step 1: Extract Cell Type Features (PCA)
# ==============================================================================
echo "=== Step 1: Extracting Cell Type Features ==="
echo "Method: PCA with $N_FEATURES components"
echo ""

max-info features celltype \
    --input "$TEST_DIR/test_data.h5ad" \
    --output "$RESULTS_DIR/features" \
    --type "pca$N_FEATURES"

echo ""
echo "✓ Features extracted"
echo ""

# ==============================================================================
# Step 2: Build k-NN Graph
# ==============================================================================
echo "=== Step 2: Building k-NN Graph ==="
echo "Parameters: k=$K_NEIGHBORS, metric=$METRIC"
echo ""

max-info features graph \
    --input "$RESULTS_DIR/features/features_pca$N_FEATURES.npy" \
    --output "$RESULTS_DIR/type_data/LeidenPCA${N_FEATURES}${METRIC^}/FEL.npy" \
    --k $K_NEIGHBORS \
    --metric $METRIC

echo ""
echo "✓ Graph built"
echo ""

# ==============================================================================
# Step 3: Run Leiden Clustering at Multiple Resolutions
# ==============================================================================
echo "=== Step 3: Running Leiden Clustering ==="
echo "Resolutions: ${RESOLUTIONS[@]}"
echo ""

for res_idx in "${RESOLUTIONS[@]}"; do
    echo "  Running resolution $res_idx..."
    max-info cluster leiden \
        --input "$RESULTS_DIR/type_data/LeidenPCA${N_FEATURES}${METRIC^}/FEL.npy" \
        --output "$RESULTS_DIR/type_data/LeidenPCA${N_FEATURES}${METRIC^}" \
        --resolution-idx $res_idx \
        --sections "$TEST_DIR/Sections.npy"
done

echo ""
echo "✓ Clustering completed for ${#RESOLUTIONS[@]} resolutions"
echo ""

# ==============================================================================
# Step 4: Run Percolation Analysis on All Resolutions
# ==============================================================================
echo "=== Step 4: Running Percolation Analysis ==="
echo "Processing ${#RESOLUTIONS[@]} resolutions: ${RESOLUTIONS[@]}"
echo ""

TOTAL_PROCESSED=0

for res_idx in "${RESOLUTIONS[@]}"; do
    echo "Processing resolution $res_idx..."
    
    # Create percolation output directory for this resolution
    PERCOLATION_DIR="$RESULTS_DIR/percolation_results/LeidenPCA${N_FEATURES}${METRIC^}/res_${res_idx}"
    mkdir -p "$PERCOLATION_DIR"
    
    # Run percolation and save .npz files
    python -c "
import numpy as np
from pathlib import Path
from max_info_atlases.percolation.graph_percolation import GraphPercolation, EdgeListManager

# Configuration
results_dir = Path('$RESULTS_DIR')
type_dir = results_dir / 'type_data/LeidenPCA${N_FEATURES}${METRIC^}/res_${res_idx}'
percolation_dir = Path('$PERCOLATION_DIR')

# Find all section files
section_files = sorted(type_dir.glob('*.npy'))
if not section_files:
    print(f'  No sections found in {type_dir}')
    exit(0)

print(f'  Found {len(section_files)} sections')

# Initialize edge list manager
elm = EdgeListManager(base_dir=str(results_dir / 'edge_lists'))

# Process each section and save results
for section_file in section_files:
    type_vec = np.load(section_file)
    
    # Load real XY coordinates from xy_data directory
    xy_file = Path('$TEST_DIR') / 'xy_data' / f'{section_file.stem}_XY.npy'
    if not xy_file.exists():
        print(f'  Skipping {section_file.stem}: No XY coordinates found at {xy_file}')
        continue
    
    XY = np.load(xy_file)
    
    # Verify dimensions match
    if len(XY) != len(type_vec):
        print(f'  Skipping {section_file.stem}: XY/type dimension mismatch')
        continue
    
    # Run percolation
    gp = GraphPercolation(XY, type_vec, maxK=100)
    gp.percolation(edge_list_manager=elm, xy_name=section_file.stem)
    
    # Save full results to .npz file
    output_file = percolation_dir / f'{section_file.stem}.npz'
    gp.save(str(output_file))
    
    # Save just the score to .score file (for efficient aggregation)
    score = gp.score()
    score_file = percolation_dir / f'{section_file.stem}.score'
    with open(score_file, 'w') as f:
        f.write(f'{score}\n')

print(f'  ✓ Processed {len(section_files)} sections for resolution ${res_idx}')
"
    
    NUM_SECTIONS=$(find "$PERCOLATION_DIR" -name "*.npz" -type f | wc -l)
    TOTAL_PROCESSED=$((TOTAL_PROCESSED + NUM_SECTIONS))
done

echo ""
echo "✓ Percolation analysis completed"
echo "  Total files processed: $TOTAL_PROCESSED"
echo ""

# ==============================================================================
# Step 5: Aggregate Scores into Final Table
# ==============================================================================
echo "=== Step 5: Aggregating Scores into Final Table ==="

PERCOLATION_BASE_DIR="$RESULTS_DIR/percolation_results"
FINAL_OUTPUT="$RESULTS_DIR/percolation_scores.csv"

# Use efficient .score file aggregation (no need to load .npz files!)
max-info results aggregate-scores \
    --base-dir "$PERCOLATION_BASE_DIR" \
    --output "$FINAL_OUTPUT"

echo ""
echo "✓ Final results saved to: $FINAL_OUTPUT"
echo ""

# ==============================================================================
# Final Summary
# ==============================================================================
echo "==================================================================="
echo "                    TEST PIPELINE COMPLETED                        "
echo "==================================================================="
echo ""
echo "Test data location:    $TEST_DIR"
echo "Results location:      $RESULTS_DIR"
echo ""
echo "Generated outputs:"
echo "  - Features:          $RESULTS_DIR/features/features_pca$N_FEATURES.npy"
echo "  - k-NN graph:        $RESULTS_DIR/type_data/LeidenPCA${N_FEATURES}${METRIC^}/FEL.npy"
echo "  - Clustering:        $RESULTS_DIR/type_data/LeidenPCA${N_FEATURES}${METRIC^}/res_*/*.npy"
echo "  - Edge lists:        $RESULTS_DIR/edge_lists/*.npz"
echo "  - Percolation data:  $RESULTS_DIR/percolation_results/**/*.npz (archival)"
echo "  - Percolation scores: $RESULTS_DIR/percolation_results/**/*.score (for aggregation)"
echo "  - Final scores CSV:  $RESULTS_DIR/percolation_scores.csv"
echo ""
echo "Results summary:"
echo "  - ${#RESOLUTIONS[@]} resolutions × 3 sections = 15 total scores"
echo "  - All combinations saved to CSV with metadata"
echo ""
echo "View results:"
echo "  head -20 $RESULTS_DIR/percolation_scores.csv"
echo "  # or load in pandas/R for analysis"
echo ""
echo "To clean results and re-run: $0 --clean"
echo "==================================================================="
