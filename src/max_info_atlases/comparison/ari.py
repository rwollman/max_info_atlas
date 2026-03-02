"""
Pairwise Adjusted Rand Index (ARI) computation between clustering configurations.

This module is intentionally separate from the standard YAML-driven pipeline.
It shares the UGE job chunking / submission machinery from ``max_info_atlases.uge``
but does not use ``RunConfig`` or ``RunOrchestrator``.

How ARI is computed
-------------------
For each pair of configurations, label arrays are loaded for every available
section, then **concatenated into a single vector** before calling
``adjusted_rand_score``.  This gives one global ARI per pair that treats all
cells equally regardless of which section they come from.

The ``sections`` argument controls which section ``.npy`` files are loaded.
It is **not** used to split the ARI calculation — it is only the file-discovery
mechanism.

Output format
-------------
Each chunk result CSV has **one row per pair**, with pair-metadata columns
(suffixed ``_1`` / ``_2``), plus ``ari``, ``n_cells``, ``n_labels_1``,
``n_labels_2``, ``n_sections``.  Combining chunks therefore just concatenates
them — no secondary aggregation step needed.

CLI workflow
------------
1. Build chunk files::

       max-info ari prepare --config config/ari_cell_type_opt.yaml

2. Submit array job::

       max-info ari submit  --config config/ari_cell_type_opt.yaml

3. Combine chunk results::

       max-info ari combine --config config/ari_cell_type_opt.yaml

Notebook / interactive use
--------------------------
Use :func:`compute_pairwise_ari` for small cases that fit in memory.
"""

from __future__ import annotations

import glob
import itertools
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_subdir(row: pd.Series) -> Optional[str]:
    """
    Return the resolution/k subdirectory name for a clustering row.

    Priority: ``resolution`` (float) → ``resolution_idx`` (int) → ``clustering_k`` (int).
    Returns ``None`` when none are present/non-null (e.g. preexisting annotations).
    """
    from ..clustering.base import format_resolution_dirname

    if "resolution" in row.index and pd.notna(row["resolution"]):
        return format_resolution_dirname(float(row["resolution"]))

    if "resolution_idx" in row.index and pd.notna(row["resolution_idx"]):
        return f"res_{int(row['resolution_idx'])}"

    if "clustering_k" in row.index and pd.notna(row["clustering_k"]):
        return f"k_{int(row['clustering_k'])}"

    return None


def _algorithm_col(row: pd.Series) -> str:
    """Return the algorithm folder name from a row."""
    for col in ("algorithm_string", "algorithm"):
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    raise KeyError(
        "Row must contain 'algorithm' or 'algorithm_string'. "
        f"Available columns: {list(row.index)}"
    )


def load_labels_for_config(
    row: pd.Series,
    clustering_dir: Union[str, Path],
    sections: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Load clustering label arrays for all requested sections for one configuration.

    Parameters
    ----------
    row:
        A single row from the configurations DataFrame.  Must have ``algorithm``
        (or ``algorithm_string``) and at least one of ``resolution``,
        ``resolution_idx``, or ``clustering_k``.
    clustering_dir:
        Root directory that contains algorithm sub-folders.
    sections:
        Iterable of section names to load.

    Returns
    -------
    Dict mapping section → ``np.ndarray`` of integer labels.
    Missing section files are silently skipped.
    """
    clustering_dir = Path(clustering_dir)
    algo = _algorithm_col(row)
    subdir = _resolve_subdir(row)

    base = clustering_dir / algo / subdir if subdir is not None else clustering_dir / algo

    labels: Dict[str, np.ndarray] = {}
    for section in sections:
        npy_path = base / f"{section}.npy"
        if npy_path.exists():
            labels[section] = np.load(npy_path)
    return labels


def discover_sections(
    row: pd.Series,
    clustering_dir: Union[str, Path],
) -> List[str]:
    """
    Return all section names available on disk for a given configuration row.

    Useful when you don't have an explicit sections list.
    """
    clustering_dir = Path(clustering_dir)
    algo = _algorithm_col(row)
    subdir = _resolve_subdir(row)

    base = clustering_dir / algo / subdir if subdir is not None else clustering_dir / algo

    if not base.exists():
        return []

    return [p.stem for p in sorted(base.glob("*.npy"))]


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def build_comparison_pairs(
    df: pd.DataFrame,
    id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate all n-choose-2 pairs from a DataFrame of clustering configurations.

    Parameters
    ----------
    df:
        One row per clustering configuration.  The integer positional index
        (0 … N-1) is used as the stable identifier; call ``reset_index(drop=True)``
        before passing if the index is not already a clean 0-based range.
    id_cols:
        Columns to carry into the output (suffixed ``_1`` / ``_2``).
        Defaults to all columns.

    Returns
    -------
    DataFrame with ``idx_1``, ``idx_2`` plus ``{col}_1`` / ``{col}_2`` for
    every column in *id_cols*.  Row count equals ``n_pairs(len(df))``.
    """
    if id_cols is None:
        id_cols = list(df.columns)

    if len(df) < 2:
        raise ValueError(f"Need at least 2 configurations to form pairs, got {len(df)}.")

    indices = list(df.index)
    records = []
    for i, j in itertools.combinations(range(len(df)), 2):
        idx_i, idx_j = indices[i], indices[j]
        row_i, row_j = df.loc[idx_i], df.loc[idx_j]
        record: Dict = {"idx_1": idx_i, "idx_2": idx_j}
        for col in id_cols:
            record[f"{col}_1"] = row_i[col] if col in row_i.index else np.nan
            record[f"{col}_2"] = row_j[col] if col in row_j.index else np.nan
        records.append(record)

    return pd.DataFrame(records)


def pairs_to_job_list(pairs_df: pd.DataFrame) -> List[str]:
    """
    Convert a pairs DataFrame to a list of ``idx_1<TAB>idx_2`` strings.

    This format is compatible with :func:`~max_info_atlases.uge.job_generator.create_chunk_files`.
    """
    return [f"{int(row.idx_1)}\t{int(row.idx_2)}" for _, row in pairs_df.iterrows()]


# ---------------------------------------------------------------------------
# Core ARI computation  (global — sections are concatenated, not iterated)
# ---------------------------------------------------------------------------

def _compute_global_ari(
    labels_a: Dict[str, np.ndarray],
    labels_b: Dict[str, np.ndarray],
    min_labels: int = 1,
) -> Optional[Dict]:
    """
    Concatenate labels across all shared sections and compute one global ARI.

    Parameters
    ----------
    labels_a, labels_b:
        Dicts mapping section → label array (from :func:`load_labels_for_config`).
    min_labels:
        If either concatenated vector has fewer than this many unique labels
        the pair is skipped and ``None`` is returned.

    Returns
    -------
    Dict with keys ``ari``, ``n_cells``, ``n_labels_1``, ``n_labels_2``,
    ``n_sections``, or ``None`` if no valid shared data exists.
    """
    shared = sorted(set(labels_a) & set(labels_b))

    parts_a: List[np.ndarray] = []
    parts_b: List[np.ndarray] = []
    for section in shared:
        a, b = labels_a[section], labels_b[section]
        if len(a) != len(b):
            continue
        parts_a.append(a)
        parts_b.append(b)

    if not parts_a:
        return None

    all_a = np.concatenate(parts_a)
    all_b = np.concatenate(parts_b)

    n_a = len(np.unique(all_a))
    n_b = len(np.unique(all_b))

    if n_a < min_labels or n_b < min_labels:
        return None

    return {
        "ari": float(adjusted_rand_score(all_a, all_b)),
        "n_cells": len(all_a),
        "n_labels_1": n_a,
        "n_labels_2": n_b,
        "n_sections": len(parts_a),
    }


def compute_ari_for_chunk(
    chunk_file: Union[str, Path],
    config_df: pd.DataFrame,
    clustering_dir: Union[str, Path],
    sections: Sequence[str],
    output_csv: Optional[Union[str, Path]] = None,
    min_labels: int = 1,
) -> pd.DataFrame:
    """
    Compute global pairwise ARI for all pairs listed in one chunk file.

    Chunk files use the text format produced by
    :func:`~max_info_atlases.uge.job_generator.create_chunk_files`:
    one ``idx_1<TAB>idx_2`` line per pair, where the indices reference rows
    in *config_df*.

    For each pair, label arrays are loaded for all available sections, then
    concatenated into a single vector before calling ``adjusted_rand_score``.
    The result is **one row per pair**.

    Parameters
    ----------
    chunk_file:
        Path to a ``chunk_NNNN_of_MMMM.txt`` file.
    config_df:
        The full configurations DataFrame.  Must have a 0-based integer index.
    clustering_dir:
        Root directory containing clustering label ``.npy`` files.
    sections:
        Section names used to locate ``.npy`` files.  All sections present
        in both configurations are concatenated before computing ARI.
    output_csv:
        If provided, the result DataFrame is saved here as CSV.
    min_labels:
        Pairs where either concatenated label vector has fewer unique labels
        are excluded.

    Returns
    -------
    DataFrame with **one row per pair**: pair metadata columns suffixed
    ``_1`` / ``_2``, plus ``ari``, ``n_cells``, ``n_labels_1``,
    ``n_labels_2``, ``n_sections``.
    """
    from ..uge.job_generator import read_job_list

    jobs = read_job_list(chunk_file)

    # Cache to avoid reloading the same config's labels more than once
    label_cache: Dict[int, Dict[str, np.ndarray]] = {}

    def _get_labels(idx: int) -> Dict[str, np.ndarray]:
        if idx not in label_cache:
            label_cache[idx] = load_labels_for_config(
                config_df.loc[idx], clustering_dir, sections
            )
        return label_cache[idx]

    meta_cols = list(config_df.columns)

    records = []
    for job_line in jobs:
        parts = job_line.split("\t")
        if len(parts) < 2:
            continue
        idx_1, idx_2 = int(parts[0]), int(parts[1])

        ari_result = _compute_global_ari(
            _get_labels(idx_1), _get_labels(idx_2), min_labels=min_labels
        )
        if ari_result is None:
            continue

        row_1, row_2 = config_df.loc[idx_1], config_df.loc[idx_2]
        record: Dict = {"idx_1": idx_1, "idx_2": idx_2}
        for col in meta_cols:
            record[f"{col}_1"] = row_1[col] if col in row_1.index else np.nan
            record[f"{col}_2"] = row_2[col] if col in row_2.index else np.nan
        record.update(ari_result)
        records.append(record)

    if not records:
        result = pd.DataFrame()
    else:
        result = pd.DataFrame(records)
        front = (
            ["idx_1", "idx_2"]
            + [f"{c}_1" for c in meta_cols]
            + [f"{c}_2" for c in meta_cols]
            + ["ari", "n_cells", "n_labels_1", "n_labels_2", "n_sections"]
        )
        front = [c for c in front if c in result.columns]
        rest = [c for c in result.columns if c not in front]
        result = result[front + rest]

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)

    return result


# ---------------------------------------------------------------------------
# Combining chunk results
# ---------------------------------------------------------------------------

def combine_ari_chunks(
    output_dir: Union[str, Path],
    result_prefix: str = "ari_result",
    combined_filename: str = "ari_combined.csv",
) -> pd.DataFrame:
    """
    Concatenate all per-chunk ARI result CSV files into a single DataFrame.

    Since each chunk already contains one row per pair, no further aggregation
    is needed — this is a pure concatenation.

    Parameters
    ----------
    output_dir:
        Directory containing ``{result_prefix}_*.csv`` files.
    result_prefix:
        Filename prefix used when chunk results were saved.
    combined_filename:
        Name of the output file written into *output_dir*.

    Returns
    -------
    Combined DataFrame (one row per pair).
    """
    output_dir = Path(output_dir)
    pattern = str(output_dir / f"{result_prefix}_*.csv")
    result_files = sorted(glob.glob(pattern))

    if not result_files:
        raise FileNotFoundError(f"No files matching '{pattern}' found.")

    print(f"Combining {len(result_files)} chunk result files…")
    dfs = []
    for f in result_files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception as exc:
            print(f"  WARNING: could not read {f}: {exc}")

    if not dfs:
        raise RuntimeError("No chunk result files could be read.")

    combined = pd.concat(dfs, ignore_index=True)
    out_path = output_dir / combined_filename
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined)} rows → {out_path}")
    return combined


# ---------------------------------------------------------------------------
# Interactive convenience function
# ---------------------------------------------------------------------------

def compute_pairwise_ari(
    df: pd.DataFrame,
    clustering_dir: Union[str, Path],
    sections: Sequence[str],
    id_cols: Optional[List[str]] = None,
    min_labels: int = 1,
) -> pd.DataFrame:
    """
    Compute all n-choose-2 pairwise ARIs in a single call (small cases).

    For large numbers of configurations use the CLI workflow instead
    (``max-info ari prepare`` → ``max-info ari submit`` → ``max-info ari combine``).

    Each pair produces one row: labels from all sections are concatenated
    before computing the global ARI.

    Parameters
    ----------
    df:
        One row per clustering configuration.  Each row needs ``algorithm``
        (or ``algorithm_string``) and at least one of ``resolution``,
        ``resolution_idx``, or ``clustering_k``.
    clustering_dir:
        Root directory containing clustering label ``.npy`` files.
    sections:
        Section names used to locate ``.npy`` files.
    id_cols:
        Subset of *df* columns to carry into the output.  Defaults to all.
    min_labels:
        Pairs where either label vector has fewer unique labels are excluded.

    Returns
    -------
    DataFrame with one row per pair.
    """
    import tempfile

    df = df.reset_index(drop=True)
    pairs_df = build_comparison_pairs(df, id_cols=id_cols)
    job_lines = pairs_to_job_list(pairs_df)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write("\n".join(job_lines))
        tmp_path = tmp.name

    try:
        result = compute_ari_for_chunk(
            chunk_file=tmp_path,
            config_df=df,
            clustering_dir=clustering_dir,
            sections=sections,
            min_labels=min_labels,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def n_pairs(n_configs: int) -> int:
    """Number of n-choose-2 pairs for *n_configs* configurations."""
    return n_configs * (n_configs - 1) // 2


def n_chunks(n_configs: int, chunk_size: int) -> int:
    """Number of chunks needed to cover all pairs."""
    return (n_pairs(n_configs) + chunk_size - 1) // chunk_size
