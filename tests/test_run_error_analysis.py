"""Tests for run error analysis."""

from pathlib import Path

import numpy as np

from max_info_atlases.run.error_analysis import _analyze_log_file, analyze_run_errors
from max_info_atlases.run.manifest import RunConfig


def _make_minimal_run_config(base_dir: Path, sections_file: Path) -> RunConfig:
    data = {
        "run_name": "test_run",
        "input": {
            "adata": str(base_dir / "input.h5ad"),
            "xy_dir": str(base_dir / "xy"),
            "sections_file": str(sections_file),
        },
        "output": {
            "base_dir": str(base_dir / "output"),
            "features_dir": "features",
            "graphs_dir": "graphs",
            "clustering_dir": "clustering",
            "percolation_dir": "percolation_results",
            "edge_lists_dir": "edge_lists",
            "jobs_dir": "jobs",
            "logs_dir": "logs",
            "scores_csv": "scores.csv",
        },
        "steps": {
            "features": {"data_types": ["raw"]},
            "graphs": {"k": 15, "distances": ["cosine"]},
            "clustering": {
                "leiden": {
                    "enabled": True,
                    "data_types": ["raw"],
                    "distances": ["cosine"],
                    "n_resolutions": 1,
                }
            },
            "percolation": {"max_k": 100},
            "aggregation": {"output_csv": "scores.csv"},
        },
    }
    return RunConfig(data)


def test_analyze_log_file_detects_abnormal_end(tmp_path):
    log_path = tmp_path / "run_clustering.123.7.log"
    log_path.write_text(
        "Processing chunk: chunk_0007_of_0010.txt\n"
        "Task ID: 7\n"
        "Start time: Mon Feb 9 00:00:00 PST 2026\n"
        "Processing 1 clustering jobs from chunk\n"
    )

    finding = _analyze_log_file("clustering", log_path)

    assert finding.abnormal_end is True
    assert finding.suspicious is True
    assert finding.task_id == 7
    assert finding.job_id == "123"


def test_analyze_run_errors_maps_warning_logs_to_missing_outputs(tmp_path, capsys):
    base_dir = tmp_path / "run"
    output_dir = base_dir / "output"
    base_dir.mkdir(parents=True, exist_ok=True)
    sections_file = base_dir / "Sections.npy"
    np.save(sections_file, np.array(["S1", "S2"], dtype=object))

    config = _make_minimal_run_config(base_dir, sections_file)

    # Prepared jobs/chunks for percolation step
    chunks_dir = output_dir / "jobs" / "percolation" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "jobs" / "percolation" / "jobs.txt").write_text(
        "LeidenRawCosine/res_0/S1.npy\t"
        f"{output_dir / 'percolation_results'}\t"
        f"{output_dir / 'edge_lists'}\t100\t"
        f"{base_dir / 'xy'}\t{output_dir / 'clustering'}\n"
        "LeidenRawCosine/res_0/S2.npy\t"
        f"{output_dir / 'percolation_results'}\t"
        f"{output_dir / 'edge_lists'}\t100\t"
        f"{base_dir / 'xy'}\t{output_dir / 'clustering'}\n"
    )
    (chunks_dir / "chunk_0001_of_0001.txt").write_text(
        "LeidenRawCosine/res_0/S1.npy\t"
        f"{output_dir / 'percolation_results'}\t"
        f"{output_dir / 'edge_lists'}\t100\t"
        f"{base_dir / 'xy'}\t{output_dir / 'clustering'}\n"
        "LeidenRawCosine/res_0/S2.npy\t"
        f"{output_dir / 'percolation_results'}\t"
        f"{output_dir / 'edge_lists'}\t100\t"
        f"{base_dir / 'xy'}\t{output_dir / 'clustering'}\n"
    )

    # Warning log for the same task
    log_dir = output_dir / "logs" / "percolation"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "test_run_percolation.111.1.log").write_text(
        "Processing chunk: chunk_0001_of_0001.txt\n"
        "Task ID: 1\n"
        "Start time: Mon Feb 9 00:00:00 PST 2026\n"
        "Processing 2 percolation jobs from chunk\n"
        "  Skipping missing file: /tmp/missing.npy\n"
        "End time: Mon Feb 9 00:00:05 PST 2026\n"
        "Done SGE_TASK_ID 1\n"
    )

    summary = analyze_run_errors(config, step="percolation", max_logs=10)
    captured = capsys.readouterr().out

    assert summary["suspicious_logs"] == 1
    assert summary["missing_outputs"] == 2
    assert summary["mapped_missing_outputs"] == 2
    assert "LeidenRawCosine/res_0/S1.score" in captured
    assert "LeidenRawCosine/res_0/S2.score" in captured
