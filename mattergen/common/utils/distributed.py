"""Hardware-agnostic DDP setup utilities.

This module mirrors HydraGNN's ``hydragnn/utils/distributed/distributed.py``
policy so MatterGen training scales seamlessly across:

* a single-process laptop (CPU or single GPU),
* a multi-GPU node launched with ``torchrun``,
* an HPC node launched with ``mpirun`` / ``srun`` (no torchrun in front),
* a multi-node Slurm/LSF/PBS job that does *not* set the PyTorch-standard
  ``RANK`` / ``LOCAL_RANK`` / ``MASTER_ADDR`` / ``MASTER_PORT`` env vars.

Design
------
* **Bootstrap policy** — Detect ``world_size``/``world_rank`` from scheduler
  env vars (OpenMPI, SLURM, PMI/Aurora) before falling back to mpi4py and
  finally a single-rank "no DDP" mode.
* **Backend auto-pick** — NCCL when CUDA is present, XCCL on Intel XPU,
  Gloo otherwise; overridable via ``MATTERGEN_BACKEND`` or the
  ``distributed_backend`` config field.
* **Master discovery** — Parse ``SLURM_NODELIST`` / ``SLURM_STEP_NODELIST``
  / ``LSB_HOSTS`` / ``PBS_O_HOST`` for the master address; derive a stable
  per-job port from ``SLURM_JOB_ID`` / ``LSB_JOBID`` / ``PBS_JOBID`` with a
  small retry loop on ``EADDRINUSE``.
* **Optional MPI** — ``mpi4py`` is imported lazily and is *not* required
  for any single-node path.

The helpers are intentionally kept dependency-free so they can be reused by
sampling/eval scripts without pulling in the full training stack.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import time
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


_BACKEND_ENV = "MATTERGEN_BACKEND"
_MASTER_ADDR_ENV = "MATTERGEN_MASTER_ADDR"
_MASTER_PORT_ENV = "MATTERGEN_MASTER_PORT"
_MASTER_PORT_RETRIES_ENV = "MATTERGEN_MASTER_PORT_RETRIES"


# ----------------------------------------------------------------------------
# Rank / world-size detection
# ----------------------------------------------------------------------------


def init_comm_size_and_rank() -> tuple[int, int]:
    """Detect ``(world_size, world_rank)`` from the launcher's environment.

    Resolution order (matching HydraGNN):

    1. OpenMPI: ``OMPI_COMM_WORLD_SIZE`` + ``OMPI_COMM_WORLD_RANK``.
    2. Slurm: ``SLURM_NPROCS`` + ``SLURM_PROCID``.
    3. PyTorch / torchrun: ``WORLD_SIZE`` + ``RANK``.
    4. Already-initialised torch.distributed group.
    5. ``mpi4py`` if importable.
    6. Single-process fallback (1, 0).
    """
    if os.getenv("OMPI_COMM_WORLD_SIZE") and os.getenv("OMPI_COMM_WORLD_RANK"):
        return (
            int(os.environ["OMPI_COMM_WORLD_SIZE"]),
            int(os.environ["OMPI_COMM_WORLD_RANK"]),
        )

    if os.getenv("SLURM_NPROCS") and os.getenv("SLURM_PROCID"):
        return int(os.environ["SLURM_NPROCS"]), int(os.environ["SLURM_PROCID"])

    if os.getenv("WORLD_SIZE") and os.getenv("RANK"):
        return int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    try:
        from mpi4py import MPI  # type: ignore

        return MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 1, 0


def get_local_rank() -> int:
    """Detect the per-node local rank used to bind the process to a GPU.

    Resolution order:

    1. ``LOCAL_RANK`` (torchrun, deepspeed, accelerate).
    2. ``OMPI_COMM_WORLD_LOCAL_RANK`` (OpenMPI).
    3. ``SLURM_LOCALID`` (Slurm step).
    4. ``PALS_LOCAL_RANKID`` (Aurora / Intel PALS).
    5. ``MPI_LOCALRANKID`` / ``MV2_COMM_WORLD_LOCAL_RANK`` (MPICH/MVAPICH).
    6. Fall back to ``world_rank % torch.cuda.device_count()`` when CUDA
       is available, else 0.
    """
    for env_var in (
        "LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "PALS_LOCAL_RANKID",
        "MPI_LOCALRANKID",
        "MV2_COMM_WORLD_LOCAL_RANK",
    ):
        v = os.getenv(env_var)
        if v is not None and v != "":
            return int(v)

    _, world_rank = init_comm_size_and_rank()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return world_rank % torch.cuda.device_count()
    return 0


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_initialized() else 1


def is_master() -> bool:
    return get_rank() == 0


def synchronize() -> None:
    """``dist.barrier()`` no-op when running single-process."""
    if get_world_size() > 1:
        dist.barrier()


# ----------------------------------------------------------------------------
# Backend / device selection
# ----------------------------------------------------------------------------


def _select_backend(requested: str | None = None) -> str:
    """Pick a torch.distributed backend that's available on this host."""
    if requested is not None and requested != "auto":
        # Honour the explicit choice but downgrade nccl→gloo on CPU-only hosts
        # so single-CPU smoke tests don't break.
        if requested == "nccl" and not torch.cuda.is_available():
            return "gloo"
        return requested

    env_choice = os.getenv(_BACKEND_ENV)
    if env_choice:
        return env_choice

    if dist.is_nccl_available() and torch.cuda.is_available():
        return "nccl"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        # PyTorch 2.4+ ships a "xccl" backend for Intel XPU.
        return "xccl"
    if dist.is_gloo_available():
        return "gloo"
    raise RuntimeError("No torch.distributed backend available on this host.")


def resolve_device(local_rank: int | None = None) -> torch.device:
    """Return the per-rank compute device and bind it for collectives."""
    if torch.cuda.is_available():
        idx = local_rank if local_rank is not None else get_local_rank()
        if idx >= torch.cuda.device_count():
            logger.warning(
                "local_rank=%d exceeds visible CUDA device count=%d; "
                "wrapping modulo. Set CUDA_VISIBLE_DEVICES correctly per rank.",
                idx,
                torch.cuda.device_count(),
            )
            idx = idx % torch.cuda.device_count()
        torch.cuda.set_device(idx)
        return torch.device("cuda", idx)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        idx = local_rank if local_rank is not None else get_local_rank()
        if idx >= torch.xpu.device_count():
            idx = idx % torch.xpu.device_count()
        torch.xpu.set_device(idx)
        return torch.device("xpu", idx)

    return torch.device("cpu")


# ----------------------------------------------------------------------------
# Master address / port discovery
# ----------------------------------------------------------------------------


_SLURM_NODELIST_RE = re.compile(r"^([A-Za-z0-9_\-\.]+?)(?:\[(.+)\])?(?:,|$)")


def _parse_slurm_nodelist(nodelist: str) -> list[str]:
    """Return individual hostnames from a Slurm-style ``nodelist`` string.

    Supports ``node01``, ``node[01-03]``, ``node[01,03,05-07]``,
    ``a01,b[02-03]``. Only the first hostname is required by ``setup_ddp``;
    we still expand the full list to keep this helper general.
    """
    hosts: list[str] = []
    cursor = 0
    while cursor < len(nodelist):
        m = _SLURM_NODELIST_RE.match(nodelist[cursor:])
        if not m:
            # Fall back: treat the rest as a single node.
            hosts.append(nodelist[cursor:].strip(","))
            break
        prefix, ranges = m.group(1), m.group(2)
        if ranges is None:
            hosts.append(prefix)
        else:
            for chunk in ranges.split(","):
                if "-" in chunk:
                    lo, hi = chunk.split("-", 1)
                    width = len(lo)
                    for i in range(int(lo), int(hi) + 1):
                        hosts.append(f"{prefix}{i:0{width}d}")
                else:
                    width = len(chunk)
                    hosts.append(f"{prefix}{int(chunk):0{width}d}")
        cursor += m.end()
    return hosts or [nodelist]


def _derive_master_port(default_port: int = 29500) -> int:
    """Return a per-job-stable TCP port for the rendezvous master."""
    explicit = os.getenv(_MASTER_PORT_ENV) or os.getenv("MASTER_PORT")
    if explicit:
        return int(explicit)
    for var in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID"):
        value = os.getenv(var)
        if not value:
            continue
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            return 10000 + (int(digits) % 50000)
    return default_port


def _derive_master_addr() -> str:
    """Return the IP/hostname of global rank 0."""
    explicit = os.getenv(_MASTER_ADDR_ENV) or os.getenv("MASTER_ADDR")
    if explicit:
        return explicit

    if os.getenv("LSB_HOSTS"):
        # Summit (LSF): "<batch_host> <rank0_host> <rank1_host> ..."
        parts = os.environ["LSB_HOSTS"].split()
        return parts[1] if len(parts) > 1 else parts[0]

    for var in ("SLURM_STEP_NODELIST", "SLURM_NODELIST", "SLURM_JOB_NODELIST"):
        nodelist = os.getenv(var)
        if nodelist:
            return _parse_slurm_nodelist(nodelist)[0]

    pbs_host = os.getenv("PBS_O_HOST")
    if pbs_host:
        return pbs_host

    # Single-host fallback: every rank resolves to the same loopback addr.
    return "127.0.0.1"


# ----------------------------------------------------------------------------
# Process group bring-up
# ----------------------------------------------------------------------------


def setup_ddp(
    backend: str | None = None,
    timeout_seconds: int = 1800,
) -> tuple[int, int]:
    """Initialise ``torch.distributed`` with scheduler-aware env discovery.

    Idempotent: if a process group is already initialised this function only
    returns the current ``(world_size, world_rank)``.

    Returns
    -------
    world_size, world_rank
        Always returns sensible values, including ``(1, 0)`` when running
        single-process (in which case no process group is created).
    """
    if is_initialized():
        return get_world_size(), get_rank()

    world_size, world_rank = init_comm_size_and_rank()

    if world_size <= 1:
        if is_master():
            logger.info("Running single-process: skipping torch.distributed init.")
        return 1, 0

    chosen_backend = _select_backend(backend)
    master_addr = _derive_master_addr()
    base_port = _derive_master_port()
    explicit_port = os.getenv(_MASTER_PORT_ENV) is not None or os.getenv("MASTER_PORT") is not None
    port_retries = int(os.getenv(_MASTER_PORT_RETRIES_ENV, "8"))

    last_exc: Exception | None = None
    for attempt in range(port_retries + 1):
        port = base_port + attempt
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(world_rank)
        os.environ.setdefault("LOCAL_RANK", str(get_local_rank()))

        if world_rank == 0:
            logger.info(
                "torch.distributed init: backend=%s master=%s:%d world_size=%d",
                chosen_backend,
                master_addr,
                port,
                world_size,
            )

        try:
            dist.init_process_group(
                backend=chosen_backend,
                init_method="env://",
                timeout=timedelta(seconds=timeout_seconds),
            )
            return world_size, world_rank
        except Exception as exc:  # noqa: BLE001 - retry on port collisions only
            err = str(exc).lower()
            collision = "eaddrinuse" in err or "address already in use" in err
            if collision and not explicit_port and attempt < port_retries:
                if world_rank == 0:
                    logger.warning(
                        "MASTER_PORT %d in use, retrying with %d", port, port + 1
                    )
                time.sleep(0.5)
                last_exc = exc
                continue
            raise

    assert last_exc is not None  # pragma: no cover
    raise last_exc


def wrap_ddp(
    model: torch.nn.Module,
    device: torch.device,
    *,
    find_unused_parameters: bool = False,
    sync_batch_norm: bool = False,
    static_graph: bool = False,
    gradient_as_bucket_view: bool = True,
) -> torch.nn.Module:
    """Wrap ``model`` in :class:`DistributedDataParallel` if a group exists.

    Returns the model unchanged when running single-process so callers can
    use a single code path.
    """
    if not is_initialized() or get_world_size() == 1:
        return model

    if sync_batch_norm and device.type == "cuda":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ddp_kwargs: dict[str, Any] = {
        "find_unused_parameters": find_unused_parameters,
        "gradient_as_bucket_view": gradient_as_bucket_view,
        "static_graph": static_graph,
    }
    if device.type in ("cuda", "xpu"):
        ddp_kwargs["device_ids"] = [device.index]
        ddp_kwargs["output_device"] = device.index

    return DistributedDataParallel(model, **ddp_kwargs)


def cleanup() -> None:
    """Tear down the process group (safe to call when uninitialised)."""
    if is_initialized():
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001 - barrier may fail mid-shutdown
            pass
        dist.destroy_process_group()


def hostname_port_summary() -> str:
    """Compact debug string useful for logging at startup."""
    return (
        f"host={socket.gethostname()} pid={os.getpid()} "
        f"rank={get_rank()}/{get_world_size()} local_rank={get_local_rank()} "
        f"cuda={torch.cuda.is_available()} "
        f"devices={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
    )
