"""
ARI analysis configuration loaded from a YAML file.

This is intentionally simpler than ``RunConfig`` — it has no pipeline steps,
no combinatorial method expansion, and no manifest tracking.  It just holds
the paths and settings needed for a pairwise ARI run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class AriConfig:
    """
    Loads and validates an ARI analysis configuration YAML.

    Expected structure (see ``config/ari_cell_type_opt.yaml`` for a full example)::

        run_name: "ari_cell_type_opt"
        input:
          configs_csv:   /path/to/cell_type_percolation_scores_reduced.csv
          clustering_dir: /path/to/clustering/
          sections:      /path/to/xy_coordinates/   # dir with *_XY.npy  OR  text file
        output:
          base_dir:    /path/to/ari_results/
          chunks_dir:  chunks        # relative to base_dir
          results_dir: results
          logs_dir:    logs
        analysis:
          id_cols: null              # null = all columns; or "algorithm,resolution"
          min_labels: 1
          aggregate: true
        hpc:
          conda_env:  max_info_atlases
          chunk_size: 500
          memory:     "8G"
          runtime:    "2:00:00"

    All path properties return absolute ``Path`` objects.
    """

    def __init__(self, data: Dict[str, Any], source_path: Optional[Path] = None):
        self._data = data
        self._source = source_path
        self._validate()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "AriConfig":
        """Load an ARI config from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ARI config not found: {p}")
        with open(p) as fh:
            data = yaml.safe_load(fh)
        return cls(data, source_path=p)

    def _validate(self) -> None:
        for section in ("run_name", "input", "output"):
            if section not in self._data:
                raise ValueError(f"Missing required config section: '{section}'")
        for key in ("configs_csv", "clustering_dir", "sections"):
            if key not in self._data["input"]:
                raise ValueError(f"Missing required input key: 'input.{key}'")
        if "base_dir" not in self._data["output"]:
            raise ValueError("Missing required output key: 'output.base_dir'")

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    @property
    def run_name(self) -> str:
        return self._data["run_name"]

    @property
    def description(self) -> str:
        return self._data.get("description", "")

    # ------------------------------------------------------------------
    # Input paths
    # ------------------------------------------------------------------

    @property
    def configs_csv(self) -> Path:
        return Path(self._data["input"]["configs_csv"])

    @property
    def clustering_dir(self) -> Path:
        return Path(self._data["input"]["clustering_dir"])

    @property
    def sections(self) -> Path:
        """Path to a sections text file or a directory containing *_XY.npy files."""
        return Path(self._data["input"]["sections"])

    # ------------------------------------------------------------------
    # Output paths  (all resolved as absolute paths under base_dir)
    # ------------------------------------------------------------------

    @property
    def base_dir(self) -> Path:
        return Path(self._data["output"]["base_dir"])

    @property
    def chunks_dir(self) -> Path:
        rel = self._data["output"].get("chunks_dir", "chunks")
        return self.base_dir / rel

    @property
    def results_dir(self) -> Path:
        rel = self._data["output"].get("results_dir", "results")
        return self.base_dir / rel

    @property
    def logs_dir(self) -> Path:
        rel = self._data["output"].get("logs_dir", "logs")
        return self.base_dir / rel

    @property
    def config_snapshot_path(self) -> Path:
        """Canonical location for the config snapshot written by ``ari prepare``."""
        return self.base_dir / "config_snapshot.csv"

    # ------------------------------------------------------------------
    # Analysis settings
    # ------------------------------------------------------------------

    @property
    def id_cols(self) -> Optional[List[str]]:
        """
        Columns from configs_csv to embed in pair metadata.
        ``None`` means all columns (default).
        """
        raw = self._data.get("analysis", {}).get("id_cols", None)
        if raw is None:
            return None
        if isinstance(raw, list):
            return [str(c) for c in raw]
        # Comma-separated string
        return [c.strip() for c in str(raw).split(",") if c.strip()]

    @property
    def min_labels(self) -> int:
        return int(self._data.get("analysis", {}).get("min_labels", 1))

    # ------------------------------------------------------------------
    # HPC settings
    # ------------------------------------------------------------------

    @property
    def conda_env(self) -> str:
        return self._data.get("hpc", {}).get("conda_env", "max_info_atlases")

    @property
    def chunk_size(self) -> int:
        return int(self._data.get("hpc", {}).get("chunk_size", 500))

    @property
    def memory(self) -> str:
        return self._data.get("hpc", {}).get("memory", "8G")

    @property
    def runtime(self) -> str:
        return self._data.get("hpc", {}).get("runtime", "2:00:00")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def worker_command(self, chunk_file_placeholder: str = "$CHUNK_FILE") -> str:
        """
        Return the ``--worker-command`` string for ``max-info jobs submit``.

        The placeholder is substituted at runtime by the UGE submission script.
        """
        return (
            f"max-info run-ari "
            f"--config {self._source or 'ari_config.yaml'} "
            f"--chunk-file {chunk_file_placeholder}"
        )

    def __repr__(self) -> str:
        return (
            f"AriConfig(run_name={self.run_name!r}, "
            f"configs_csv={self.configs_csv}, "
            f"base_dir={self.base_dir})"
        )
