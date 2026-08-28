#!/usr/bin/env python3
"""Summarize JamieTest rank logs in place on Frontier.

Example:
    python analyze_jamie_test.py jobOutputs/JamieTest-123456 --archive
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re
import shlex
import tarfile


STATUS_RE = re.compile(r"status-rank-(\d+)-(.+)\.txt$")
TRACE_RE = re.compile(r"trace-rank-(\d+)-(.+)\.log$")
RCCL_RE = re.compile(r"rccl-(.+)-(\d+)\.log$")
STAGE_RE = re.compile(r"\bstage=([^\s]+)")
PID_RE = re.compile(r"\bpid=(\d+)")

ERROR_PATTERNS = {
    "SIGBUS": re.compile(r"SIGBUS|Bus error|signal(?:\s+|=)7\b", re.IGNORECASE),
    "SIGABRT": re.compile(
        r"SIGABRT|Fatal Python error:\s*Aborted|signal(?:\s+|=)6\b|"
        r"Exited with exit code 134\b",
        re.IGNORECASE,
    ),
    "illegal GPU instruction": re.compile(
        r"HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION|illegal shader instruction",
        re.IGNORECASE,
    ),
    "timeout": re.compile(r"timed?\s*out|timeout|watchdog", re.IGNORECASE),
    "collective error": re.compile(
        r"(?:NCCL|RCCL).*(?:\bWARN\b|\bERROR\b|unhandled|abort(?:ed)?|"
        r"system error|internal error)|collective.*(?:error|fail)",
        re.IGNORECASE,
    ),
    "traceback": re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    "segfault": re.compile(r"SIGSEGV|segmentation fault", re.IGNORECASE),
    "out of memory": re.compile(r"out of memory|OOM", re.IGNORECASE),
}

BENIGN_PLUGIN_PATTERNS = (
    re.compile(
        r"NCCL INFO (?:NET|TUNER)/Plugin: (?:Failed to find|Could not find)",
        re.IGNORECASE,
    ),
    re.compile(
        r"NCCL INFO PROFILER/Plugin: Could not find",
        re.IGNORECASE,
    ),
)


@dataclass
class RankRecord:
    rank: int
    host: str = "unknown"
    state: str = "missing"
    status: dict[str, str] = field(default_factory=dict)
    status_file: Path | None = None
    trace_file: Path | None = None
    last_stage: str = "no_trace"
    pid: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="jobOutputs/JamieTest-<job-id>")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="create a compact .tar.gz with summaries and suspect-rank logs",
    )
    parser.add_argument(
        "--max-error-lines",
        type=int,
        default=80,
        help="maximum matching error lines included in the summary",
    )
    parser.add_argument(
        "--max-suspect-ranks",
        type=int,
        default=32,
        help="maximum detailed suspect-rank records included in the summary",
    )
    return parser.parse_args()


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    try:
        tokens = shlex.split(line)
    except ValueError:
        tokens = line.split()
    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def compact_ranges(values: list[int]) -> str:
    if not values:
        return "none"
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def load_rank_records(log_dir: Path) -> tuple[dict[int, RankRecord], dict[tuple[str, str], int]]:
    records: dict[int, RankRecord] = {}
    pid_to_rank: dict[tuple[str, str], int] = {}

    for path in sorted(log_dir.glob("status-rank-*.txt")):
        match = STATUS_RE.match(path.name)
        if match is None:
            continue
        rank, host = int(match.group(1)), match.group(2)
        fields = parse_fields(path.read_text(errors="replace").strip())
        record = records.setdefault(rank, RankRecord(rank=rank))
        record.host = fields.get("host", host)
        record.state = fields.get("state", "unknown")
        record.status = fields
        record.status_file = path

    for path in sorted(log_dir.glob("trace-rank-*.log")):
        match = TRACE_RE.match(path.name)
        if match is None:
            continue
        rank, host = int(match.group(1)), match.group(2)
        record = records.setdefault(rank, RankRecord(rank=rank, host=host))
        record.trace_file = path
        for line in path.read_text(errors="replace").splitlines():
            stage_match = STAGE_RE.search(line)
            if stage_match:
                record.last_stage = stage_match.group(1)
            pid_match = PID_RE.search(line)
            if pid_match:
                record.pid = pid_match.group(1)
        if record.pid is not None:
            pid_to_rank[(record.host, record.pid)] = rank

    return records, pid_to_rank


def scan_errors(
    paths: list[Path],
    records: dict[int, RankRecord],
    pid_to_rank: dict[tuple[str, str], int],
    limit: int,
) -> tuple[Counter[str], list[str], set[Path], set[int]]:
    counts: Counter[str] = Counter()
    samples: list[str] = []
    error_files: set[Path] = set()
    error_ranks: set[int] = set()

    for path in paths:
        rank: int | None = None
        trace_match = TRACE_RE.match(path.name)
        status_match = STATUS_RE.match(path.name)
        rccl_match = RCCL_RE.match(path.name)
        slurm_match = re.search(r"slurm-rank-(\d+)-", path.name)
        if trace_match or status_match:
            rank = int((trace_match or status_match).group(1))
        elif slurm_match:
            rank = int(slurm_match.group(1))
        elif rccl_match:
            rank = pid_to_rank.get((rccl_match.group(1), rccl_match.group(2)))

        try:
            with path.open(errors="replace") as stream:
                for line_number, line in enumerate(stream, 1):
                    if any(pattern.search(line) for pattern in BENIGN_PLUGIN_PATTERNS):
                        continue
                    labels = [label for label, pattern in ERROR_PATTERNS.items() if pattern.search(line)]
                    if not labels:
                        continue
                    counts.update(labels)
                    error_files.add(path)
                    if rank is not None:
                        error_ranks.add(rank)
                    if len(samples) < limit:
                        rank_label = "?" if rank is None else str(rank)
                        samples.append(
                            f"rank={rank_label} {path.name}:{line_number}: {line.rstrip()[:500]}"
                        )
        except OSError as exc:
            samples.append(f"could not read {path}: {exc}")

    return counts, samples, error_files, error_ranks


def build_report(
    log_dir: Path,
    max_error_lines: int,
    max_suspect_ranks: int = 32,
):
    records, pid_to_rank = load_rank_records(log_dir)
    log_paths = sorted(
        path
        for path in log_dir.iterdir()
        if path.is_file() and path.name != "analysis-summary.txt"
    )
    master_log = log_dir.parent.parent / f"{log_dir.name}.out"
    if master_log.is_file():
        log_paths.append(master_log)
    error_counts, error_samples, error_files, error_ranks = scan_errors(
        log_paths, records, pid_to_rank, max_error_lines
    )

    ranks = sorted(records)
    expected = list(range(max(ranks) + 1)) if ranks else []
    missing = sorted(set(expected) - set(ranks))
    states = Counter(record.state for record in records.values())
    stages = Counter(record.last_stage for record in records.values())
    failed = sorted(rank for rank, record in records.items() if record.state == "failed")
    incomplete = sorted(rank for rank, record in records.items() if record.state != "completed")

    sizes: dict[str, int] = defaultdict(int)
    for path in log_paths:
        prefix = path.name.split("-", 1)[0]
        sizes[prefix] += path.stat().st_size
    total_size = sum(sizes.values())

    lines = [
        f"JamieTest analysis: {log_dir}",
        f"files={len(log_paths)} total_size={human_size(total_size)} observed_ranks={len(records)}",
        f"rank_states={dict(sorted(states.items()))}",
        f"failed_ranks={compact_ranges(failed)}",
        f"incomplete_ranks={compact_ranges(incomplete)}",
        f"missing_rank_records={compact_ranges(missing)}",
        "",
        "Last trace stage by rank count:",
    ]
    lines.extend(f"  {stage}: {count}" for stage, count in stages.most_common())

    lines.extend(["", "Trace-stage placement:"])
    records_by_host: dict[str, list[RankRecord]] = defaultdict(list)
    for record in records.values():
        records_by_host[record.host].append(record)
    for stage, count in stages.most_common():
        stage_records = sorted(
            (record for record in records.values() if record.last_stage == stage),
            key=lambda record: record.rank,
        )
        local_ranks = Counter(
            record.status.get("local_rank", "?") for record in stage_records
        )
        stage_hosts = sorted({record.host for record in stage_records})
        uniform_hosts = sorted(
            host
            for host in stage_hosts
            if all(record.last_stage == stage for record in records_by_host[host])
        )
        lines.append(
            f"  {stage}: count={count} hosts={len(stage_hosts)} "
            f"uniform_hosts={len(uniform_hosts)} "
            f"local_ranks={dict(sorted(local_ranks.items()))}"
        )
        lines.append(
            f"    ranks={compact_ranges([record.rank for record in stage_records])}"
        )
        if uniform_hosts:
            lines.append(f"    uniform_host_names={','.join(uniform_hosts)}")

    lines.extend(["", "Log sizes:"])
    lines.extend(f"  {kind}: {human_size(size)}" for kind, size in sorted(sizes.items()))
    lines.extend(["", f"Error signatures: {dict(error_counts)}"])

    suspect_ranks = sorted(set(failed) | error_ranks)
    if suspect_ranks:
        lines.extend(["", "Suspect ranks:"])
        shown_suspect_ranks = suspect_ranks[: max(0, max_suspect_ranks)]
        for rank in shown_suspect_ranks:
            record = records.get(rank, RankRecord(rank=rank))
            status_details = " ".join(
                f"{key}={value}"
                for key, value in record.status.items()
                if key in {"state", "host", "rc", "signal"}
            )
            lines.append(
                f"  rank={rank} last_stage={record.last_stage} {status_details}".rstrip()
            )
        omitted = len(suspect_ranks) - len(shown_suspect_ranks)
        if omitted:
            lines.append(
                f"  ... {omitted} additional suspect ranks omitted "
                f"(use --max-suspect-ranks to show more)"
            )

    if error_samples:
        lines.extend(["", "Matched error lines:"])
        lines.extend(f"  {sample}" for sample in error_samples)

    if total_size < 250 * 1024**2:
        lines.extend(["", "Archive note: the full directory is small enough to compress and copy."])
    else:
        lines.extend(
            [
                "",
                "Archive note: logs are fairly large; use --archive for a compact diagnostic bundle.",
            ]
        )

    return "\n".join(lines) + "\n", records, error_files, error_ranks


def create_archive(
    log_dir: Path,
    summary_path: Path,
    records: dict[int, RankRecord],
    error_files: set[Path],
    error_ranks: set[int],
) -> Path:
    archive_path = log_dir.with_name(f"{log_dir.name}-diagnostics.tar.gz")
    selected = {summary_path, *error_files}
    master_log = log_dir.parent.parent / f"{log_dir.name}.out"
    if master_log.is_file():
        selected.add(master_log)
    for record in records.values():
        if record.status_file is not None:
            selected.add(record.status_file)
        if record.trace_file is not None:
            selected.add(record.trace_file)
        if record.rank in error_ranks:
            selected.update(log_dir.glob(f"slurm-rank-{record.rank}-*.out"))
            if record.pid is not None:
                selected.update(log_dir.glob(f"rccl-{record.host}-{record.pid}.log"))

    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(selected):
            if path.is_file():
                archive.add(path, arcname=f"{log_dir.name}/{path.name}")
    return archive_path


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"Log directory does not exist: {log_dir}")

    report, records, error_files, error_ranks = build_report(
        log_dir, args.max_error_lines, args.max_suspect_ranks
    )
    summary_path = log_dir / "analysis-summary.txt"
    summary_path.write_text(report)
    print(report, end="")
    print(f"Summary written to {summary_path}")

    if args.archive:
        archive_path = create_archive(
            log_dir, summary_path, records, error_files, error_ranks
        )
        print(f"Compact archive written to {archive_path} ({human_size(archive_path.stat().st_size)})")


if __name__ == "__main__":
    main()
