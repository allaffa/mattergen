# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MPI helpers for the cached-dataset backends.

Mirrors the policy used by HydraGNN (see
``hydragnn/utils/distributed/distributed.py``):

* ``mpi4py`` is an *optional* dependency; if it is missing or the process is
  not part of an MPI world, the helpers transparently fall back to a
  single-rank ``_SingleRankComm``.
* The active communicator is detected from common HPC environment variables
  (Open MPI, SLURM) before falling back to ``MPI.COMM_WORLD``.
* :func:`nsplit` mirrors HydraGNN's even-as-possible split used to assign
  contiguous slices of work to ranks.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Sequence


class _SingleRankComm:
    """Minimal stand-in for ``MPI.Intracomm`` when MPI is unavailable."""

    rank = 0
    size = 1

    def Get_rank(self) -> int:  # noqa: N802 - mimic MPI naming
        return 0

    def Get_size(self) -> int:  # noqa: N802 - mimic MPI naming
        return 1

    def Barrier(self) -> None:  # noqa: N802 - mimic MPI naming
        return None

    def barrier(self) -> None:
        return None

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj

    def allgather(self, obj: Any) -> list[Any]:
        return [obj]


SingleRankComm = _SingleRankComm  # public alias


def _detect_mpi_env() -> bool:
    """Return True if the process looks like it is launched under MPI/SLURM."""
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0")) > 1:
        return True
    if int(os.environ.get("PMI_SIZE", "0")) > 1:
        return True
    if int(os.environ.get("SLURM_NTASKS", "0")) > 1:
        return True
    return False


def auto_comm(force: bool | None = None):
    """Return the active communicator.

    Parameters
    ----------
    force:
        ``True`` always tries to import ``mpi4py`` and returns
        ``MPI.COMM_WORLD``; raises if mpi4py is missing.
        ``False`` always returns a single-rank stand-in.
        ``None`` (default) returns ``MPI.COMM_WORLD`` when mpi4py is
        importable *and* the launch environment looks like MPI/SLURM, else
        the single-rank stand-in.
    """
    if force is False:
        return _SingleRankComm()

    use_mpi = force is True or _detect_mpi_env()
    if not use_mpi:
        return _SingleRankComm()

    try:
        from mpi4py import MPI  # type: ignore
    except ImportError as e:
        if force is True:
            raise ImportError(
                "MPI was requested but `mpi4py` is not installed. "
                "Install with `pip install mattergen[mpi]`."
            ) from e
        return _SingleRankComm()
    return MPI.COMM_WORLD


def is_mpi_comm(comm: Any) -> bool:
    """Return True if ``comm`` is a real ``mpi4py`` communicator (size > 1)."""
    if comm is None:
        return False
    if isinstance(comm, _SingleRankComm):
        return False
    try:
        return int(comm.Get_size()) > 1
    except Exception:
        return False


def nsplit(total: int, n: int) -> list[int]:
    """Split ``total`` items as evenly as possible into ``n`` chunks.

    Mirrors :func:`hydragnn.utils.distributed.distributed.nsplit` so that
    rank ``r`` owns ``sum(splits[:r]) : sum(splits[:r+1])``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def shard_range(total: int, rank: int, size: int) -> tuple[int, int]:
    """Return ``[start, end)`` for ``rank`` of ``size`` over ``total`` items."""
    splits = nsplit(total, size)
    start = sum(splits[:rank])
    return start, start + splits[rank]


def gather_lengths(local: Sequence[int], comm: Any) -> list[Sequence[int]]:
    """Allgather a per-rank sequence of integers."""
    return comm.allgather(list(local))


__all__ = [
    "auto_comm",
    "gather_lengths",
    "is_mpi_comm",
    "nsplit",
    "shard_range",
    "SingleRankComm",
]
