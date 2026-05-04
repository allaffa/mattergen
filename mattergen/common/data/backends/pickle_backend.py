# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Single-file pickle backend for cached datasets.

Inspired by HydraGNN's ``SimplePickleDataset`` but specialised to MatterGen's
``CacheBundle`` layout. The whole split (core packed arrays + properties) is
stored in a single ``dataset.pkl`` per split. This is fast for the dataset
sizes MatterGen uses (mp_20 trains in ~20MB) and avoids the per-structure
file explosion that HydraGNN's chunked pickle layout requires.

MPI policy
----------
When an MPI communicator is supplied to :func:`write` or :func:`read`, the
backend mirrors HydraGNN's collective IO contract:

* **Write**: every rank passes its *local* slab; rank 0 gathers the slabs
  via :meth:`Comm.allgather` and writes a single, rank-stable file. This
  keeps the on-disk layout backend-format-stable while making preprocessing
  embarrassingly parallel.
* **Read**: rank 0 deserialises from disk and broadcasts the bundle to all
  ranks. This avoids contention on parallel filesystems while preserving
  MatterGen's ``DistributedSampler``-based dataloader contract (every rank
  holds the full dataset and the sampler shards batches).
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Iterable

import numpy as np

from mattergen.common.data.backends._bundle import (
    PICKLE_FILENAME,
    CacheBundle,
    concat_bundles,
)
from mattergen.common.data.types import PropertySourceId

_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


def _path(cache_path: str) -> str:
    return os.path.join(cache_path, PICKLE_FILENAME)


def _comm_size(comm: Any) -> int:
    if comm is None:
        return 1
    return int(comm.Get_size())


def _comm_rank(comm: Any) -> int:
    if comm is None:
        return 0
    return int(comm.Get_rank())


def _serialise_payload(bundle: CacheBundle) -> dict:
    return {
        "format_version": 1,
        "pos": np.ascontiguousarray(bundle.pos),
        "cell": np.ascontiguousarray(bundle.cell),
        "atomic_numbers": np.ascontiguousarray(bundle.atomic_numbers),
        "num_atoms": np.ascontiguousarray(bundle.num_atoms),
        "structure_id": np.ascontiguousarray(bundle.structure_id),
        "properties": {k: np.ascontiguousarray(v) for k, v in bundle.properties.items()},
    }


def write(cache_path: str, bundle: CacheBundle, comm: Any = None) -> None:
    size = _comm_size(comm)
    rank = _comm_rank(comm)

    if size > 1:
        # Each rank contributes its local slab; rank 0 concatenates and writes.
        gathered = comm.allgather(bundle)
        if rank == 0:
            global_bundle = concat_bundles(gathered)
            _write_local(cache_path, global_bundle)
        comm.Barrier()
        return

    _write_local(cache_path, bundle)


def _write_local(cache_path: str, bundle: CacheBundle) -> None:
    payload = _serialise_payload(bundle)
    tmp = _path(cache_path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=_PICKLE_PROTOCOL)
    os.replace(tmp, _path(cache_path))


def _select_properties(
    available: dict[str, np.ndarray],
    properties: Iterable[PropertySourceId] | None,
    cache_path: str,
) -> dict[str, np.ndarray]:
    if properties is None:
        return dict(available)
    wanted = list(properties)
    missing = [p for p in wanted if p not in available]
    if missing:
        raise FileNotFoundError(
            f"Properties {missing} not found in {cache_path!r}. "
            f"Available: {sorted(available)}"
        )
    return {p: available[p] for p in wanted}


def _read_local(
    cache_path: str, properties: Iterable[PropertySourceId] | None
) -> CacheBundle:
    with open(_path(cache_path), "rb") as f:
        payload = pickle.load(f)
    selected = _select_properties(payload.get("properties", {}), properties, cache_path)
    return CacheBundle(
        pos=payload["pos"],
        cell=payload["cell"],
        atomic_numbers=payload["atomic_numbers"],
        num_atoms=payload["num_atoms"],
        structure_id=payload["structure_id"],
        properties=selected,
    )


def read(
    cache_path: str,
    properties: Iterable[PropertySourceId] | None = None,
    comm: Any = None,
) -> CacheBundle:
    size = _comm_size(comm)
    rank = _comm_rank(comm)

    if size > 1:
        bundle = _read_local(cache_path, properties) if rank == 0 else None
        bundle = comm.bcast(bundle, root=0)
        return bundle

    return _read_local(cache_path, properties)


def list_properties(cache_path: str) -> list[PropertySourceId]:
    with open(_path(cache_path), "rb") as f:
        payload = pickle.load(f)
    return sorted(payload.get("properties", {}).keys())
