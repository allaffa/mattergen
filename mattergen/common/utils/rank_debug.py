"""Low-overhead, rank-local startup breadcrumbs for distributed debugging."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import socket
from pathlib import Path


_TRACE_FILE_ENV = "MATTERGEN_RANK_TRACE_FILE"


def _rank() -> str:
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
        value = os.getenv(name)
        if value is not None:
            return value
    return "0"


def trace_rank(stage: str, **details: object) -> None:
    """Print and flush one timestamped breadcrumb to this rank's trace file.

    Tracing is disabled unless ``MATTERGEN_RANK_TRACE_FILE`` is set. Opening
    the file for each breadcrumb makes the last completed stage durable even
    when a native library terminates the process with SIGBUS.
    """

    trace_file = os.getenv(_TRACE_FILE_ENV)
    if not trace_file:
        return

    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    fields = " ".join(f"{key}={value!r}" for key, value in sorted(details.items()))
    message = (
        f"{timestamp} rank={_rank()} pid={os.getpid()} host={socket.gethostname()} "
        f"stage={stage}"
    )
    if fields:
        message = f"{message} {fields}"

    path = Path(trace_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as stream:
        print(message, file=stream, flush=True)


__all__ = ["trace_rank"]
