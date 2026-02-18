"""
Run configuration and manifest generation.

The RunConfig loads a YAML run definition. The RunManifest computes
the full set of expected outputs and checks progress against actual files.
"""

import os
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..features.graphs import build_folder_name


@dataclass
class MethodSpec:
    """Specification for a single clustering method."""
    name: str           # 'leiden', 'phenograph', 'preexisting'
    enabled: bool
    data_types: List[str] = field(default_factory=list)
    distances: List[str] = field(default_factory=list)
    n_resolutions: int = 50
    levels: List[str] = field(default_factory=list)  # Preexisting only


class RunConfig:
    """
    Loads and validates a run configuration YAML.
    
    The config defines the full combinatorial space for a pipeline run.
    
    Example:
        config = RunConfig.from_yaml('config/run_cell_type_opt.yaml')
        print(config.run_name)
        print(config.methods)
    """
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self._validate()
    
    @classmethod
    def from_yaml(cls, path: str) -> 'RunConfig':
        """Load run config from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Run config not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(data)
    
    def _validate(self):
        """Validate required config sections."""
        required_sections = ['run_name', 'input', 'output', 'steps']
        for section in required_sections:
            if section not in self._data:
                raise ValueError(f"Missing required config section: '{section}'")
    
    # --- Properties ---
    
    @property
    def run_name(self) -> str:
        return self._data['run_name']
    
    @property
    def description(self) -> str:
        return self._data.get('description', '')
    
    # --- Input paths ---
    
    @property
    def adata_path(self) -> str:
        return self._data['input']['adata']
    
    @property
    def xy_dir(self) -> str:
        return self._data['input']['xy_dir']
    
    @property
    def sections_file(self) -> str:
        return self._data['input']['sections_file']
    
    @property
    def section_column(self) -> Optional[str]:
        """Name of the section column in adata.obs (if it exists there)."""
        return self._data['input'].get('section_column')
    
    @property
    def preexisting_obs_columns(self) -> Optional[dict]:
        """Mapping from preexisting annotation level name to adata.obs column name."""
        return self._data['input'].get('preexisting_obs_columns')
    
    @property
    def preexisting_dir(self) -> Optional[Path]:
        """Directory where extracted preexisting annotations are saved (inside clustering_dir)."""
        # Preexisting annotations are extracted from adata.obs into clustering_dir
        # as Preexisting_{level}/{section}.npy files
        return self.clustering_dir
    
    # --- Output paths ---
    
    @property
    def base_dir(self) -> Path:
        return Path(self._data['output']['base_dir'])
    
    @property
    def features_dir(self) -> Path:
        return self.base_dir / self._data['output'].get('features_dir', 'features')
    
    @property
    def graphs_dir(self) -> Path:
        return self.base_dir / self._data['output'].get('graphs_dir', 'graphs')
    
    @property
    def clustering_dir(self) -> Path:
        return self.base_dir / self._data['output'].get('clustering_dir', 'clustering')
    
    @property
    def percolation_dir(self) -> Path:
        return self.base_dir / self._data['output'].get('percolation_dir', 'percolation_results')
    
    @property
    def edge_lists_dir(self) -> Path:
        return self.base_dir / self._data['output'].get('edge_lists_dir', 'edge_lists')
    
    @property
    def jobs_dir(self) -> Path:
        return self.base_dir / self._data['output'].get('jobs_dir', 'jobs')
    
    @property
    def logs_dir(self) -> Path:
        return self.base_dir / self._data['output'].get('logs_dir', 'logs')
    
    @property
    def scores_csv(self) -> Path:
        return self.base_dir / self._data['output'].get('scores_csv', 'percolation_scores.csv')
    
    @property
    def reduced_scores_csv(self) -> Path:
        return self.base_dir / self._data['output'].get('reduced_scores_csv', 'percolation_scores_reduced.csv')
    
    # --- Steps ---
    
    @property
    def local_frequency_config(self) -> Optional[Dict]:
        """Get local frequency feature configuration."""
        return self._data['steps'].get('features', {}).get('local_frequency')
    
    @property
    def derived_features_config(self) -> List[Dict]:
        """Get derived features configuration."""
        return self._data['steps'].get('features', {}).get('derived', [])
    
    @property
    def feature_data_types(self) -> List[str]:
        """
        Get all feature data types (expression + local frequency + derived).
        
        Auto-generates from:
        1. Explicit data_types list (expression features)
        2. local_frequency config (generates localfreq_k{k} or localfreq_w_k{k})
        3. derived config (generates derived feature names)
        """
        all_types = []
        
        # 1. Explicit data_types (expression features)
        explicit_types = self._data['steps'].get('features', {}).get('data_types', [])
        all_types.extend(explicit_types)
        
        # 2. Local frequency features
        lf_config = self.local_frequency_config
        if lf_config:
            k_values = lf_config.get('k_values', [])
            weighted = lf_config.get('weighted', False)
            
            # Handle weighted as bool or list
            if isinstance(weighted, bool):
                weighted_list = [weighted]
            else:
                weighted_list = weighted
            
            # Generate names
            for k in k_values:
                for w in weighted_list:
                    if w:
                        all_types.append(f'localfreq_w_k{k}')
                    else:
                        all_types.append(f'localfreq_k{k}')
        
        # 3. Derived features
        derived_config = self.derived_features_config
        if derived_config:
            from ..features.derived import parse_derived_config, build_derived_feature_name
            
            # Parse derived specs to expand parameters
            parsed_specs = parse_derived_config(derived_config)
            
            for spec in parsed_specs:
                source = spec.get('source')
                operation = spec.get('operation')
                
                # Determine source feature names
                if source == 'local_frequency':
                    # Apply to all local frequency features
                    source_names = [t for t in all_types if t.startswith('localfreq_')]
                else:
                    # Specific source feature
                    source_names = [source] if source in all_types else []
                
                # Generate derived feature names
                for source_name in source_names:
                    derived_name = build_derived_feature_name(source_name, operation, spec)
                    if derived_name not in all_types:
                        all_types.append(derived_name)
        
        return all_types
    
    @property
    def feature_dependencies(self) -> Dict[str, str]:
        """
        Get mapping of derived feature names to their source feature names.
        
        Returns:
            Dict mapping derived_feature_name -> source_feature_name
        """
        dependencies = {}
        
        derived_config = self.derived_features_config
        if not derived_config:
            return dependencies
        
        from ..features.derived import parse_derived_config, build_derived_feature_name
        
        # Get all feature types for source resolution
        all_types = self._data['steps'].get('features', {}).get('data_types', [])
        
        # Add local frequency features
        lf_config = self.local_frequency_config
        if lf_config:
            k_values = lf_config.get('k_values', [])
            weighted = lf_config.get('weighted', False)
            
            if isinstance(weighted, bool):
                weighted_list = [weighted]
            else:
                weighted_list = weighted
            
            for k in k_values:
                for w in weighted_list:
                    if w:
                        all_types.append(f'localfreq_w_k{k}')
                    else:
                        all_types.append(f'localfreq_k{k}')
        
        # Parse derived specs
        parsed_specs = parse_derived_config(derived_config)
        
        for spec in parsed_specs:
            source = spec.get('source')
            operation = spec.get('operation')
            
            # Determine source feature names
            if source == 'local_frequency':
                # Apply to all local frequency features
                source_names = [t for t in all_types if t.startswith('localfreq_')]
            else:
                # Specific source feature
                source_names = [source] if source in all_types else []
            
            # Map derived -> source
            for source_name in source_names:
                derived_name = build_derived_feature_name(source_name, operation, spec)
                dependencies[derived_name] = source_name
        
        return dependencies
    
    @property
    def graph_k(self) -> int:
        return self._data['steps']['graphs']['k']
    
    @property
    def graph_distances(self) -> List[str]:
        return self._data['steps']['graphs']['distances']
    
    @property
    def percolation_max_k(self) -> int:
        return self._data['steps']['percolation']['max_k']
    
    @property
    def methods(self) -> List[MethodSpec]:
        """
        Parse clustering method specs from config.
        
        Supports:
        - Explicit list: data_types: [localfreq_k50, localfreq_k100]
        - All features: data_types: all
        - Wildcards: data_types: ["localfreq_k*", "localfreq_pca15_*"]
        """
        import fnmatch
        
        methods = []
        clustering = self._data['steps']['clustering']
        
        # Get all available feature types for resolution
        all_feature_types = self.feature_data_types
        
        for method_name, method_cfg in clustering.items():
            if not isinstance(method_cfg, dict):
                continue
            
            if not method_cfg.get('enabled', True):
                continue
            
            # Resolve data_types (support 'all' and wildcards)
            dt_config = method_cfg.get('data_types', [])
            
            if dt_config == 'all' or dt_config == '*':
                # Use all generated features
                resolved_data_types = all_feature_types
            elif isinstance(dt_config, list):
                # Resolve wildcards in list
                resolved_data_types = []
                for pattern in dt_config:
                    if '*' in pattern or '?' in pattern:
                        # Wildcard pattern - match against all features
                        matches = [ft for ft in all_feature_types 
                                   if fnmatch.fnmatch(ft, pattern)]
                        resolved_data_types.extend(matches)
                    else:
                        # Literal feature name
                        if pattern in all_feature_types:
                            resolved_data_types.append(pattern)
                
                # Remove duplicates while preserving order
                seen = set()
                resolved_data_types = [x for x in resolved_data_types 
                                       if not (x in seen or seen.add(x))]
            else:
                # Backward compatibility: treat as list
                resolved_data_types = dt_config if isinstance(dt_config, list) else []
            
            spec = MethodSpec(
                name=method_name,
                enabled=True,
                data_types=resolved_data_types,
                distances=method_cfg.get('distances', []),
                n_resolutions=method_cfg.get('n_resolutions', 50),
                levels=method_cfg.get('levels', []),
            )
            methods.append(spec)
        
        return methods
    
    # --- HPC settings ---
    
    @property
    def hpc_conda_env(self) -> str:
        return self._data.get('hpc', {}).get('conda_env', 'max_info_atlases')
    
    def hpc_resources(self, step: str) -> Dict[str, Any]:
        """Get HPC resources for a step."""
        defaults = {'memory': '16G', 'runtime': '4:00:00', 'chunk_size': 6000}
        resources = self._data.get('hpc', {}).get('resources', {}).get(step, {})
        return {**defaults, **resources}


class RunManifest:
    """
    Computes the full set of expected outputs from a RunConfig.
    
    This is the core of the monitoring system: it knows exactly what files
    the run should produce, so you can check progress at any time.
    
    Example:
        config = RunConfig.from_yaml('config/run_cell_type_opt.yaml')
        manifest = RunManifest(config)
        
        # See what we expect
        summary = manifest.summary()
        print(summary)
        
        # Check progress
        progress = manifest.check_progress()
        for step, info in progress.items():
            print(f"{step}: {info['completed']}/{info['expected']} ({info['pct']:.1f}%)")
    """
    
    def __init__(self, config: RunConfig, sections: Optional[List[str]] = None):
        """
        Initialize manifest.
        
        Args:
            config: Run configuration
            sections: List of section names. If None, loaded from config.sections_file.
        """
        self.config = config
        self._sections = sections
        self._expected: Optional[Dict[str, List[str]]] = None
    
    @property
    def sections(self) -> List[str]:
        """Get section names (loads from file if not provided)."""
        if self._sections is None:
            self._sections = self._load_sections()
        return self._sections
    
    def _load_sections(self) -> List[str]:
        """Load section names from Sections.npy or scan XY files."""
        sections_file = Path(self.config.sections_file)
        
        if sections_file.exists():
            sections_arr = np.load(sections_file, allow_pickle=True)
            # Get unique section names
            return sorted(set(str(s) for s in sections_arr))
        
        # Fallback: scan XY directory for {section}_XY.npy files
        xy_dir = Path(self.config.xy_dir)
        if xy_dir.exists():
            sections = []
            for f in xy_dir.glob("*_XY.npy"):
                section = f.stem.replace('_XY', '')
                sections.append(section)
            return sorted(sections)
        
        raise FileNotFoundError(
            f"Cannot determine sections. Neither {sections_file} nor {self.config.xy_dir} found."
        )
    
    def _method_folder_name(self, method_name: str, data_type: str, distance: str) -> str:
        """Build the folder name for a method/data_type/distance combination."""
        # Capitalize method name for folder
        if method_name == 'leiden':
            method_str = 'Leiden'
        elif method_name == 'phenograph':
            method_str = 'PhenoGraph'
        else:
            method_str = method_name.capitalize()
        
        return build_folder_name(method_str, data_type, distance)
    
    def compute_expected(self) -> Dict[str, List[str]]:
        """
        Compute all expected output files for every pipeline step.
        
        Returns:
            Dict mapping step name to list of expected relative file paths.
        """
        if self._expected is not None:
            return self._expected
        
        sections = self.sections
        config = self.config
        
        expected = {
            'features': [],
            'graphs': [],
            'clustering': [],
            'percolation': [],
            'aggregation': [],
        }
        
        # --- Features ---
        for dt in config.feature_data_types:
            expected['features'].append(f"features_{dt}.npy")
        
        # --- Graphs ---
        for dt in config.feature_data_types:
            for dist in config.graph_distances:
                expected['graphs'].append(f"FEL_{dt}_{dist}.npy")
        
        # --- Clustering + Percolation ---
        for method in config.methods:
            if method.name in ('leiden', 'phenograph'):
                for dt in method.data_types:
                    for dist in method.distances:
                        folder = self._method_folder_name(method.name, dt, dist)
                        
                        # Import here to avoid circular dependency
                        from ..clustering.base import get_resolution_values, format_resolution_dirname
                        resolution_values = get_resolution_values(method.n_resolutions)
                        
                        for res_idx in range(method.n_resolutions):
                            resolution = resolution_values[res_idx]
                            res_dirname = format_resolution_dirname(resolution)
                            
                            for section in sections:
                                # Clustering output
                                expected['clustering'].append(
                                    f"{folder}/{res_dirname}/{section}.npy"
                                )
                                # Percolation output
                                expected['percolation'].append(
                                    f"{folder}/{res_dirname}/{section}.score"
                                )
            
            elif method.name == 'preexisting':
                for level in method.levels:
                    for section in sections:
                        # Preexisting: no clustering step, just percolation
                        expected['percolation'].append(
                            f"Preexisting_{level}/{section}.score"
                        )
        
        # --- Aggregation ---
        expected['aggregation'].append(
            config._data['output'].get('scores_csv', 'percolation_scores.csv')
        )
        # Reduced scores (weighted average across sections)
        expected['aggregation'].append(
            config._data['output'].get('reduced_scores_csv', 'percolation_scores_reduced.csv')
        )
        
        self._expected = expected
        return expected
    
    def summary(self) -> Dict[str, int]:
        """
        Summary counts of expected outputs per step.
        
        Returns:
            Dict mapping step name to count of expected files.
        """
        expected = self.compute_expected()
        return {step: len(files) for step, files in expected.items()}

    def hpc_job_counts(self) -> Dict[str, int]:
        """
        Number of actual HPC jobs per step (distinct from output file counts).

        For clustering each job covers all sections and writes one file per
        section, so HPC jobs = output files / n_sections.  For every other
        step the file count equals the job count.

        Returns:
            Dict mapping step name to number of HPC jobs.
        """
        file_counts = self.summary()
        n_sections = max(len(self.sections), 1)
        job_counts = {}
        for step, count in file_counts.items():
            if step == 'clustering':
                job_counts[step] = count // n_sections
            else:
                job_counts[step] = count
        return job_counts

    def print_summary(self):
        """Print a human-readable summary of the run."""
        config = self.config
        sections = self.sections
        counts = self.summary()
        jobs = self.hpc_job_counts()
        
        print(f"Run: {config.run_name}")
        if config.description:
            print(f"Description: {config.description}")
        print(f"Sections: {len(sections)}")
        print()
        
        # Methods breakdown
        print("Methods:")
        for method in config.methods:
            if method.name in ('leiden', 'phenograph'):
                combos = len(method.data_types) * len(method.distances) * method.n_resolutions
                print(f"  {method.name}: {len(method.data_types)} data_types x "
                      f"{len(method.distances)} distances x "
                      f"{method.n_resolutions} resolutions = {combos} configurations")
            elif method.name == 'preexisting':
                print(f"  preexisting: {len(method.levels)} annotation levels")
        print()
        
        # Expected outputs
        print("Expected outputs:")
        print(f"  {'Step':<20} {'Output Files':>14} {'HPC Jobs':>10}")
        print(f"  {'-' * 20} {'-' * 14} {'-' * 10}")
        total_files = 0
        total_jobs = 0
        for step, count in counts.items():
            print(f"  {step:<20} {count:>14,} {jobs[step]:>10,}")
            total_files += count
            total_jobs += jobs[step]
        print(f"  {'-' * 20} {'-' * 14} {'-' * 10}")
        print(f"  {'TOTAL':<20} {total_files:>14,} {total_jobs:>10,}")
    
    def check_progress(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare expected outputs against actual files on disk.
        
        Returns:
            Dict mapping step name to progress info:
                expected: total expected count
                completed: number of existing files
                pct: completion percentage
                missing_count: number of missing files
        """
        expected = self.compute_expected()
        config = self.config
        
        # Map step to base directory
        step_dirs = {
            'features': config.features_dir,
            'graphs': config.graphs_dir,
            'clustering': config.clustering_dir,
            'percolation': config.percolation_dir,
            'aggregation': config.base_dir,
        }
        
        progress = {}
        
        for step, expected_files in expected.items():
            step_dir = step_dirs[step]
            
            completed = 0
            missing = []
            
            for rel_path in expected_files:
                full_path = step_dir / rel_path
                if full_path.exists():
                    completed += 1
                else:
                    missing.append(rel_path)
            
            total = len(expected_files)
            pct = (completed / total * 100) if total > 0 else 100.0
            
            progress[step] = {
                'expected': total,
                'completed': completed,
                'pct': pct,
                'missing_count': len(missing),
                'missing': missing,  # Full list (can be large)
            }
        
        return progress
    
    def print_progress(self, show_missing: int = 5):
        """
        Print human-readable progress report.
        
        Args:
            show_missing: Max number of missing files to show per step (0=none)
        """
        config = self.config
        progress = self.check_progress()
        jobs = self.hpc_job_counts()
        
        print(f"Run: {config.run_name}")
        print(f"Checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 75}")
        print(f"  {'Step':<20} {'HPC Jobs':>10} {'Output Files':>14} {'Completed':>10} {'Progress':>10}")
        print(f"  {'-' * 20} {'-' * 10} {'-' * 14} {'-' * 10} {'-' * 10}")
        
        all_done = True
        for step, info in progress.items():
            status = f"{info['pct']:.1f}%"
            if info['pct'] == 100.0:
                status += "  ✓"
            else:
                all_done = False
            
            print(f"  {step:<20} {jobs[step]:>10,} {info['expected']:>14,} {info['completed']:>10,} {status:>10}")
        
        print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}")
        
        if all_done:
            print("\n  All steps complete!")
        else:
            # Show missing files for incomplete steps
            for step, info in progress.items():
                if info['missing_count'] > 0 and show_missing > 0:
                    print(f"\n  Missing {step} outputs ({info['missing_count']} total):")
                    for path in info['missing'][:show_missing]:
                        print(f"    - {path}")
                    if info['missing_count'] > show_missing:
                        print(f"    ... and {info['missing_count'] - show_missing} more")
    
    def save_manifest(self, output_path: Optional[str] = None):
        """
        Save manifest to JSON file for later reference.
        
        Args:
            output_path: Path to save manifest. Defaults to {jobs_dir}/manifest.json.
        """
        import json
        
        if output_path is None:
            output_path = self.config.jobs_dir / "manifest.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        expected = self.compute_expected()
        
        manifest_data = {
            'run_name': self.config.run_name,
            'description': self.config.description,
            'created': datetime.now().isoformat(),
            'sections': self.sections,
            'summary': self.summary(),
            'expected_outputs': expected,
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        print(f"Saved manifest to {output_path}")
        return output_path
