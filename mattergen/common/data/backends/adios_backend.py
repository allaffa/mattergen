# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ADIOS2 backend for cached datasets.

The ADIOS2 backend mirrors the packed-array layout of
:class:`~mattergen.common.data.backends.CacheBundle` as named variables in a
single ``dataset.bp`` file. ADIOS2 is an optional dependency; importing
:mod:`adios2` is deferred to read/write time so MatterGen can run without it.
String columns (``structure_id`` and string-valued properties such as
``space_group``) are stored as length-prefixed UTF-8 byte arrays.

MPI policy
----------
Mirrors HydraGNN's ``AdiosWriter``/``AdiosDataset``:

* **Write**: when an MPI communicator is supplied, every rank passes its
  *local* slab. Per-variable global shapes and per-rank offsets are
  established via :meth:`Comm.allgather`; ADIOS2 writes the file
  collectively with ``Stream(path, 'w', comm)``. The result is a single
  rank-stable ``.bp`` file.
* **Read**: rank 0 deserialises the bundle and broadcasts it to all ranks.
  This preserves MatterGen's ``DistributedSampler`` contract (every rank
  holds the full dataset; the sampler shards batches at iteration time).
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import numpy as np

from mattergen.common.data.backends._bundle import (
    ADIOS_FILENAME,
    CacheBundle,
)
from mattergen.common.data.types import PropertySourceId

_PROP_PREFIX = "prop/"
_PROP_DTYPE_PREFIX = "prop_dtype/"
_CORE_VARS = ("pos", "cell", "atomic_numbers", "num_atoms")


def _import_adios2():
    try:
        import adios2  # type: ignore
    except ImportError as e:  # pragma: no cover - exercised only when extra missing
        raise ImportError(
            "The ADIOS2 backend requires the optional `adios2` package. "
            "Install with `pip install mattergen[adios]` or `pip install adios2`."
        ) from e
    return adios2


def _path(cache_path: str) -> str:
    return os.path.join(cache_path, ADIOS_FILENAME)


def _comm_size(comm: Any) -> int:
    if comm is None:
        return 1
    return int(comm.Get_size())


def _comm_rank(comm: Any) -> int:
    if comm is None:
        return 0
    return int(comm.Get_rank())


def _encode_strings(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode a 1-D array of strings as a flat uint8 buffer + per-item lengths."""
    encoded = [str(x).encode("utf-8") for x in arr]
    lengths = np.array([len(b) for b in encoded], dtype=np.int64)
    if encoded:
        flat = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
    else:
        flat = np.zeros((0,), dtype=np.uint8)
    return flat, lengths


def _decode_strings(flat: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    out = []
    offset = 0
    raw = flat.tobytes()
    for n in lengths:
        n = int(n)
        out.append(raw[offset : offset + n].decode("utf-8"))
        offset += n
    return np.array(out, dtype=object) if out else np.array([], dtype=object)


def _global_shape_and_offset(local_shape: tuple[int, ...], comm: Any) -> tuple[list[int], list[int]]:
    """Compute ADIOS2 (global_shape, start) along axis 0.

    Higher dimensions must be globally identical; we assert via allgather.
    """
    rank = _comm_rank(comm)
    shapes = comm.allgather(list(local_shape))
    if local_shape:
        for s in shapes:
            assert (
                list(s)[1:] == list(local_shape)[1:]
            ), f"Trailing dims must be uniform across ranks: {shapes}"
    n_local = local_shape[0] if local_shape else 0
    n_total = sum(s[0] for s in shapes if s)
    offset0 = sum(shapes[r][0] for r in range(rank) if shapes[r])
    global_shape = [n_total] + list(local_shape[1:])
    start = [offset0] + [0] * (len(local_shape) - 1) if local_shape else [0]
    return global_shape, start


def _write_array(stream, name: str, arr: np.ndarray, comm: Any) -> None:
    arr = np.ascontiguousarray(arr)
    if comm is None or _comm_size(comm) == 1:
        stream.write(name, arr, list(arr.shape), [0] * arr.ndim, list(arr.shape))
    else:
        global_shape, start = _global_shape_and_offset(arr.shape, comm)
        stream.write(name, arr, global_shape, start, list(arr.shape))


def write(cache_path: str, bundle: CacheBundle, comm: Any = None) -> None:
    adios2 = _import_adios2()

    rank = _comm_rank(comm)
    size = _comm_size(comm)

    # Only rank 0 cleans up an existing .bp; barrier so writers don't race.
    path = _path(cache_path)
    if rank == 0:
        if os.path.isdir(path):  # ADIOS2 stores .bp as a directory in v2.x
            import shutil

            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    if comm is not None and hasattr(comm, "Barrier"):
        comm.Barrier()

    # Property ordering must agree across ranks for collective definitions.
    if size > 1:
        per_rank_names = comm.allgather(sorted(bundle.properties.keys()))
        if any(n != per_rank_names[0] for n in per_rank_names):
            raise ValueError(
                "All ranks must contribute the same set of properties for a "
                f"collective ADIOS2 write. Got per-rank names: {per_rank_names}"
            )
        prop_names = per_rank_names[0]
    else:
        prop_names = sorted(bundle.properties.keys())

    open_kwargs: dict[str, Any] = {}
    if size > 1:
        open_kwargs["comm"] = comm

    with adios2.Stream(path, "w", **open_kwargs) as f:
        # Core packed arrays.
        for name in _CORE_VARS:
            arr = np.ascontiguousarray(getattr(bundle, name))
            _write_array(f, name, arr, comm)
            f.write_attribute(f"{name}/dtype", str(arr.dtype))

        # structure_id as utf-8 byte buffer + lengths.
        sid_flat, sid_lengths = _encode_strings(np.asarray(bundle.structure_id))
        _write_array(f, "structure_id/data", sid_flat, comm)
        _write_array(f, "structure_id/lengths", sid_lengths, comm)

        f.write_attribute("properties/names", prop_names)

        for name in prop_names:
            arr = np.asarray(bundle.properties[name])
            if arr.dtype.kind == "U":
                flat, lengths = _encode_strings(arr)
                _write_array(f, f"{_PROP_PREFIX}{name}/data", flat, comm)
                _write_array(f, f"{_PROP_PREFIX}{name}/lengths", lengths, comm)
                f.write_attribute(f"{_PROP_DTYPE_PREFIX}{name}", "string")
            else:
                arr = np.ascontiguousarray(arr)
                _write_array(f, f"{_PROP_PREFIX}{name}", arr, comm)
                f.write_attribute(f"{_PROP_DTYPE_PREFIX}{name}", str(arr.dtype))

    if comm is not None and hasattr(comm, "Barrier"):
        comm.Barrier()


def _read_array(stream, name: str) -> np.ndarray:
    arr = stream.read(name)
    return np.asarray(arr)


def _read_local(
    cache_path: str, properties: Iterable[PropertySourceId] | None
) -> CacheBundle:
    adios2 = _import_adios2()
    path = _path(cache_path)
    with adios2.Stream(path, "r") as f:
        for _ in f.steps():  # ADIOS2 streams require iterating steps.
            core = {name: _read_array(f, name) for name in _CORE_VARS}
            sid_flat = _read_array(f, "structure_id/data")
            sid_lengths = _read_array(f, "structure_id/lengths")
            structure_id = _decode_strings(sid_flat, sid_lengths)

            available = list(f.read_attribute("properties/names"))
            if properties is None:
                wanted = available
            else:
                wanted = list(properties)
                missing = [p for p in wanted if p not in available]
                if missing:
                    raise FileNotFoundError(
                        f"Properties {missing} not found in {cache_path!r}. "
                        f"Available: {sorted(available)}"
                    )

            props: dict[PropertySourceId, np.ndarray] = {}
            for name in wanted:
                dtype_attr = f.read_attribute(f"{_PROP_DTYPE_PREFIX}{name}")
                dtype_str = dtype_attr[0] if hasattr(dtype_attr, "__len__") else dtype_attr
                if dtype_str == "string":
                    flat = _read_array(f, f"{_PROP_PREFIX}{name}/data")
                    lengths = _read_array(f, f"{_PROP_PREFIX}{name}/lengths")
                    props[name] = _decode_strings(flat, lengths)
                else:
                    props[name] = _read_array(f, f"{_PROP_PREFIX}{name}")
            break  # single-step file

    return CacheBundle(
        pos=core["pos"],
        cell=core["cell"],
        atomic_numbers=core["atomic_numbers"],
        num_atoms=core["num_atoms"],
        structure_id=structure_id,
        properties=props,
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
    adios2 = _import_adios2()

    path = _path(cache_path)
    with adios2.Stream(path, "r") as f:
        for _ in f.steps():
            names = list(f.read_attribute("properties/names"))
            break
    return sorted(names)
