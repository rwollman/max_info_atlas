"""
Run error analysis utilities.

Maps suspicious log files to the exact run parameters (job lines) and missing outputs.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple

from .manifest import RunConfig, RunManifest
from ..uge.job_generator import read_job_list


ERROR_PATTERNS = [
    re.compile(r"\bERROR\b", re.IGNORECASE),
    re.compile(r"Traceback", re.IGNORECASE),
    re.compile(r"\bException\b", re.IGNORECASE),
    re.compile(r"MemoryError", re.IGNORECASE),
    re.compile(r"Segmentation fault", re.IGNORECASE),
    re.compile(r"\bkilled\b", re.IGNORECASE),
    re.compile(r"No chunk file found", re.IGNORECASE),
]

WARNING_PATTERNS = [
    re.compile(r"Skipping missing file", re.IGNORECASE),
    re.compile(r"Skipping \(no XY file\)", re.IGNORECASE),
    re.compile(r"Skipping \(XY/type mismatch\)", re.IGNORECASE),
    re.compile(r"Skipping malformed job", re.IGNORECASE),
]

DONE_PATTERN = re.compile(r"Done SGE_TASK_ID", re.IGNORECASE)
START_PATTERN = re.compile(r"Start time:", re.IGNORECASE)
LOG_NAME_PATTERN = re.compile(r"\.(?P<job_id>\d+)\.(?P<task_id>\d+)\.log$")
CHUNK_NAME_PATTERN = re.compile(r"chunk_(?P<task_id>\d+)_of_\d+\.txt$")


@dataclass
class LogFinding:
    step: str
    path: Path
    task_id: Optional[int]
    job_id: Optional[str]
    has_error: bool
    has_warning: bool
    abnormal_end: bool
    matched_lines: List[str]

    @property
    def suspicious(self) -> bool:
        return self.has_error or self.has_warning or self.abnormal_end


@dataclass
class JobRecord:
    step: str
    task_id: Optional[int]
    job_line: str
    output_rel_paths: List[str]


def _normalize_relative(path_str: str, base_dir: Path) -> str:
    path = Path(path_str)
    try:
        if path.is_absolute():
            return path.relative_to(base_dir).as_posix()
    except ValueError:
        pass
    return path.as_posix()


def _parse_task_id_from_chunk(chunk_path: Path) -> Optional[int]:
    match = CHUNK_NAME_PATTERN.search(chunk_path.name)
    if not match:
        return None
    return int(match.group("task_id"))


def _parse_log_name(log_path: Path) -> Tuple[Optional[str], Optional[int]]:
    match = LOG_NAME_PATTERN.search(log_path.name)
    if not match:
        return None, None
    return match.group("job_id"), int(match.group("task_id"))


def _collect_matching_lines(
    text: str,
    patterns: List[re.Pattern],
    max_lines: int = 5,
) -> List[str]:
    matches: List[str] = []
    for line in text.splitlines():
        if any(p.search(line) for p in patterns):
            matches.append(line.strip())
            if len(matches) >= max_lines:
                break
    return matches


def _analyze_log_file(step: str, log_path: Path) -> LogFinding:
    text = log_path.read_text(errors="replace")
    job_id, task_id = _parse_log_name(log_path)

    has_error = any(p.search(text) for p in ERROR_PATTERNS)
    has_warning = any(p.search(text) for p in WARNING_PATTERNS)
    completed = bool(DONE_PATTERN.search(text))
    started = bool(START_PATTERN.search(text))
    abnormal_end = started and not completed

    matched_lines = _collect_matching_lines(text, ERROR_PATTERNS + WARNING_PATTERNS)
    if abnormal_end and not matched_lines:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            matched_lines = [f"Abnormal termination (last line): {lines[-1]}"]
        else:
            matched_lines = ["Abnormal termination (empty log file)"]

    return LogFinding(
        step=step,
        path=log_path,
        task_id=task_id,
        job_id=job_id,
        has_error=has_error,
        has_warning=has_warning,
        abnormal_end=abnormal_end,
        matched_lines=matched_lines,
    )


def _outputs_for_job(step: str, job_line: str, config: RunConfig, sections: List[str]) -> List[str]:
    parts = job_line.split("\t")

    if step == "features" and len(parts) >= 3:
        data_type = parts[2]
        return [f"features_{data_type}.npy"]

    if step == "graphs" and len(parts) >= 2:
        return [_normalize_relative(parts[1], config.graphs_dir)]

    if step == "clustering" and len(parts) >= 4:
        from ..clustering.base import get_resolution_values, format_resolution_dirname
        output_dir = _normalize_relative(parts[2], config.clustering_dir)
        res_idx = int(parts[3])
        # Get resolution value
        for method in config.methods:
            if method.name in ('leiden', 'phenograph'):
                resolution_values = get_resolution_values(method.n_resolutions)
                resolution = resolution_values[res_idx]
                res_dirname = format_resolution_dirname(resolution)
                return [f"{output_dir}/{res_dirname}/{section}.npy" for section in sections]
        # Fallback if no method found
        res_dirname = f"res_{res_idx}"
        return [f"{output_dir}/{res_dirname}/{section}.npy" for section in sections]

    if step == "percolation" and len(parts) >= 1:
        rel_path = parts[0]
        return [rel_path.replace(".npy", ".score")]

    if step == "aggregation" and len(parts) >= 2:
        outputs = [_normalize_relative(parts[1], config.base_dir)]
        # Include reduced scores if present in job params
        if len(parts) >= 5:
            outputs.append(_normalize_relative(parts[4], config.base_dir))
        return outputs

    return []


def _build_job_records(step: str, config: RunConfig, manifest: RunManifest) -> List[JobRecord]:
    step_dir = config.jobs_dir / step
    chunks_dir = step_dir / "chunks"
    jobs_file = step_dir / "jobs.txt"
    sections = manifest.sections

    records: List[JobRecord] = []

    if chunks_dir.exists():
        chunk_files = sorted(chunks_dir.glob("chunk_*_of_*.txt"))
        for chunk_file in chunk_files:
            task_id = _parse_task_id_from_chunk(chunk_file)
            jobs = read_job_list(chunk_file)
            for job in jobs:
                records.append(
                    JobRecord(
                        step=step,
                        task_id=task_id,
                        job_line=job,
                        output_rel_paths=_outputs_for_job(step, job, config, sections),
                    )
                )
        return records

    if jobs_file.exists():
        for job in read_job_list(jobs_file):
            records.append(
                JobRecord(
                    step=step,
                    task_id=None,
                    job_line=job,
                    output_rel_paths=_outputs_for_job(step, job, config, sections),
                )
            )

    return records


def analyze_run_errors(
    config: RunConfig,
    step: Optional[str] = None,
    max_logs: int = 20,
    max_jobs_per_log: int = 5,
    max_missing_per_log: int = 5,
) -> Dict[str, int]:
    """
    Analyze suspicious logs and map them to exact job parameters.

    Returns summary counts for CLI/reporting:
        {
            'suspicious_logs': int,
            'missing_outputs': int,
            'mapped_missing_outputs': int,
        }
    """
    manifest = RunManifest(config)
    progress = manifest.check_progress()

    all_steps = ["features", "graphs", "clustering", "percolation", "aggregation"]
    steps = [step] if step else all_steps

    missing_by_step: Dict[str, Set[str]] = {
        s: set(progress[s]["missing"]) for s in steps if s in progress
    }
    total_missing = sum(len(v) for v in missing_by_step.values())

    job_records_by_step: Dict[str, List[JobRecord]] = {
        s: _build_job_records(s, config, manifest) for s in steps
    }

    outputs_to_jobs: Dict[Tuple[str, str], List[JobRecord]] = {}
    jobs_by_task: Dict[Tuple[str, int], List[JobRecord]] = {}
    for step_name, records in job_records_by_step.items():
        for rec in records:
            for output_rel in rec.output_rel_paths:
                outputs_to_jobs.setdefault((step_name, output_rel), []).append(rec)
            if rec.task_id is not None:
                jobs_by_task.setdefault((step_name, rec.task_id), []).append(rec)

    findings: List[LogFinding] = []
    for step_name in steps:
        log_dir = config.logs_dir / step_name
        if not log_dir.exists():
            continue
        for log_path in sorted(log_dir.glob("*.log")):
            finding = _analyze_log_file(step_name, log_path)
            if finding.suspicious:
                findings.append(finding)

    findings = findings[:max_logs]
    mapped_missing_total = 0

    print(f"Run: {config.run_name}")
    print("Analyzing suspicious logs and mapping to run parameters")
    print(f"Logs directory: {config.logs_dir}")
    print(f"Total missing outputs (selected steps): {total_missing}")
    print(f"Suspicious logs found: {len(findings)}")
    print()

    if not findings:
        print("No suspicious logs found.")
        return {
            "suspicious_logs": 0,
            "missing_outputs": total_missing,
            "mapped_missing_outputs": 0,
        }

    for idx, finding in enumerate(findings, start=1):
        print(f"[{idx}] {finding.path}")
        print(f"  Step: {finding.step}")
        print(f"  Job ID: {finding.job_id or 'unknown'}")
        print(f"  Task ID: {finding.task_id if finding.task_id is not None else 'unknown'}")
        labels = []
        if finding.has_error:
            labels.append("error")
        if finding.has_warning:
            labels.append("warning")
        if finding.abnormal_end:
            labels.append("abnormal_end")
        print(f"  Flags: {', '.join(labels) if labels else 'none'}")

        if finding.matched_lines:
            print("  Matched log lines:")
            for line in finding.matched_lines:
                print(f"    - {line}")

        task_jobs = (
            jobs_by_task.get((finding.step, finding.task_id), [])
            if finding.task_id is not None
            else []
        )

        mapped_missing: List[str] = []
        for rec in task_jobs:
            for out_rel in rec.output_rel_paths:
                if out_rel in missing_by_step.get(finding.step, set()):
                    mapped_missing.append(out_rel)

        if mapped_missing:
            unique_missing = sorted(set(mapped_missing))
            mapped_missing_total += len(unique_missing)
            print(f"  Missing outputs linked to this task: {len(unique_missing)}")
            for rel_path in unique_missing[:max_missing_per_log]:
                print(f"    - {rel_path}")
            remaining = len(unique_missing) - max_missing_per_log
            if remaining > 0:
                print(f"    ... and {remaining} more")
        else:
            print("  Missing outputs linked to this task: 0")

        if task_jobs:
            print(f"  Job parameters in chunk (showing up to {max_jobs_per_log}):")
            for rec in task_jobs[:max_jobs_per_log]:
                print(f"    - {rec.job_line}")
            remaining = len(task_jobs) - max_jobs_per_log
            if remaining > 0:
                print(f"    ... and {remaining} more jobs in chunk")
        else:
            print("  Job parameters in chunk: unavailable")

        print()

    # Missing outputs with no suspicious log mapping (still useful to highlight)
    unmapped = 0
    for step_name in steps:
        missing = missing_by_step.get(step_name, set())
        for rel_path in missing:
            if (step_name, rel_path) not in outputs_to_jobs:
                unmapped += 1

    if unmapped > 0:
        print(f"Missing outputs without job mapping: {unmapped}")
        print("This usually means jobs/chunks were cleaned or not prepared yet.")
        print()

    return {
        "suspicious_logs": len(findings),
        "missing_outputs": total_missing,
        "mapped_missing_outputs": mapped_missing_total,
    }
