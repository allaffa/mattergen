# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Backend dispatch for cached crystal datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import numpy.typing

from mattergen.common.data.types import PropertySourceId

PickleFormat = "pickle"
AdiosFormat = "adios"
SUPPORTED_FORMATS = (PickleFormat, AdiosFormat)

# Canonical filenames for each backend. A cache directory is identified by the
# presence of one of these files.
_PICKLE_FILENAME = "dataset.pkl"
_ADIOS_FILENAME = "dataset.bp"


@dataclass
class CacheBundle:
    """In-memory representation of a cached split.

    Mirrors the packed-array layout used by
    :class:`mattergen.common.data.dataset.CrystalDataset`: ``pos`` and
    ``atomic_numbers`` are concatenated across structures and addressed via
    cumulative ``num_atoms`` offsets; ``cell`` and ``structure_id`` are
    one-per-structure. Properties are stored as a dictionary of
    one-per-structure arrays, keyed by ``PropertySourceId``.
    """

    pos: numpy.typing.NDArray
    cell: numpy.typing.NDArray
    atomic_numbers: numpy.typing.NDArray
    num_atoms: numpy.typing.NDArray
    structure_id: numpy.typing.NDArray
    properties: dict[PropertySourceId, numpy.typing.NDArray] = field(default_factory=dict)


def detect_format(cache_path: str) -> str:
    """Return the storage format of ``cache_path``.

    Raises :class:`FileNotFoundError` if no recognised cache file is present.
    """
    if os.path.isfile(os.path.join(cache_path, _ADIOS_FILENAME)):
        return AdiosFormat
    if os.path.isfile(os.path.join(cache_path, _PICKLE_FILENAME)):
        return PickleFormat
    raise FileNotFoundError(
        f"No supported dataset cache found in {cache_path!r}. "
        f"Expected one of: {_PICKLE_FILENAME}, {_ADIOS_FILENAME}. "
        "If you have a legacy .npy cache, run "
        "`python -m mattergen.scripts.preprocess_dataset` to convert it."
    )


def read_cache(
    cache_path: str,
    properties: Iterable[PropertySourceId] | None = None,
    fmt: str | None = None,
    comm: Any = None,
) -> CacheBundle:
    """Read a cache directory into a :class:`CacheBundle`.

    Parameters
    ----------
    cache_path:
        Path to the cache directory.
    properties:
        Optional whitelist of property names to load. ``None`` loads every
        available property; an empty iterable loads none.
    fmt:
        Force a specific backend; otherwise auto-detected.
    comm:
        Optional MPI communicator. When provided and its size is > 1, only
        rank 0 reads from disk and the resulting :class:`CacheBundle` is
        broadcast to every rank. ``None`` performs a serial read on every
        caller (the default and the back-compat behaviour).
    """
    fmt = fmt or detect_format(cache_path)
    if fmt == PickleFormat:
        from mattergen.common.data.backends import pickle_backend

        return pickle_backend.read(cache_path, properties=properties, comm=comm)
    if fmt == AdiosFormat:
        from mattergen.common.data.backends import adios_backend

        return adios_backend.read(cache_path, properties=properties, comm=comm)
    raise ValueError(f"Unknown cache format {fmt!r}; expected one of {SUPPORTED_FORMATS}.")


def write_cache(
    cache_path: str,
    bundle: CacheBundle,
    fmt: str = PickleFormat,
    comm: Any = None,
) -> None:
    """Persist a :class:`CacheBundle` to ``cache_path`` using ``fmt``.

    When ``comm`` is provided and its size is > 1, every rank passes its
    *local* slab of structures and the backend coordinates a collective
    write: the pickle backend gathers slabs to rank 0 and writes a single
    file; the ADIOS2 backend writes a single ``.bp`` file via parallel
    collective IO.
    """
    if comm is None or getattr(comm, "Get_rank", lambda: 0)() == 0:
        os.makedirs(cache_path, exist_ok=True)
    if comm is not None:
        comm.Barrier() if hasattr(comm, "Barrier") else None
    if fmt == PickleFormat:
        from mattergen.common.data.backends import pickle_backend

        pickle_backend.write(cache_path, bundle, comm=comm)
        return
    if fmt == AdiosFormat:
        from mattergen.common.data.backends import adios_backend

        adios_backend.write(cache_path, bundle, comm=comm)
        return
    raise ValueError(f"Unknown cache format {fmt!r}; expected one of {SUPPORTED_FORMATS}.")


def list_available_properties(
    cache_path: str, fmt: str | None = None
) -> list[PropertySourceId]:
    """Return the property names stored in ``cache_path``."""
    fmt = fmt or detect_format(cache_path)
    if fmt == PickleFormat:
        from mattergen.common.data.backends import pickle_backend

        return pickle_backend.list_properties(cache_path)
    if fmt == AdiosFormat:
        from mattergen.common.data.backends import adios_backend

        return adios_backend.list_properties(cache_path)
    raise ValueError(f"Unknown cache format {fmt!r}; expected one of {SUPPORTED_FORMATS}.")


# Filenames are exposed for tests and the conversion CLI.
PICKLE_FILENAME = _PICKLE_FILENAME
ADIOS_FILENAME = _ADIOS_FILENAME


def concat_bundles(bundles: Sequence[CacheBundle]) -> CacheBundle:
    """Concatenate per-rank bundles into a single global bundle.

    Used by the collective-write paths: each rank produces a local slab of
    structures, rank 0 (or the dispatcher) joins them in rank order so the
    resulting cache file is rank-stable.
    """
    if not bundles:
        raise ValueError("concat_bundles requires at least one bundle.")
    if len(bundles) == 1:
        return bundles[0]

    pos = np.concatenate([b.pos for b in bundles], axis=0)
    cell = np.concatenate([b.cell for b in bundles], axis=0)
    atomic_numbers = np.concatenate([b.atomic_numbers for b in bundles], axis=0)
    num_atoms = np.concatenate([b.num_atoms for b in bundles], axis=0)
    structure_id = np.concatenate([b.structure_id for b in bundles], axis=0)

    prop_names: list[PropertySourceId] = []
    seen: set[PropertySourceId] = set()
    for b in bundles:
        for k in b.properties:
            if k not in seen:
                prop_names.append(k)
                seen.add(k)
    properties: dict[PropertySourceId, np.ndarray] = {}
    for name in prop_names:
        slabs = []
        for b in bundles:
            if name not in b.properties:
                raise ValueError(
                    f"Bundle is missing property {name!r}; all ranks must "
                    "contribute the same set of properties for a collective write."
                )
            slabs.append(b.properties[name])
        properties[name] = np.concatenate(slabs, axis=0)

    return CacheBundle(
        pos=pos,
        cell=cell,
        atomic_numbers=atomic_numbers,
        num_atoms=num_atoms,
        structure_id=structure_id,
        properties=properties,
    )
