# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Convert a legacy ``.npy``+``.json`` dataset cache to the pickle/ADIOS2 backends.

Example
-------
Convert all splits of a dataset to the pickle backend in place::

    python -m mattergen.scripts.preprocess_dataset \\
        --src datasets/cache/mp_20 \\
        --format pickle \\
        --in-place

Convert to ADIOS2 (requires the optional ``adios2`` package) into a new
location::

    python -m mattergen.scripts.preprocess_dataset \\
        --src datasets/cache/mp_20 \\
        --dst datasets/cache/mp_20_bp \\
        --format adios

Run the conversion under MPI to shard the per-split work across ranks
(requires the optional ``mpi4py`` extra, and ``adios2`` built with MPI for
the ADIOS2 backend)::

    mpirun -n 4 python -m mattergen.scripts.preprocess_dataset \\
        --src datasets/cache/mp_20 \\
        --format adios \\
        --in-place \\
        --mpi

When ``--src`` points at a directory containing ``train``/``val``/``test``
subdirectories, each split is converted; otherwise ``--src`` is treated as a
single split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from mattergen.common.data.backends import (
    SUPPORTED_FORMATS,
    CacheBundle,
    auto_comm,
    shard_range,
    write_cache,
)


_LEGACY_CORE = {
    "pos": "pos.npy",
    "cell": "cell.npy",
    "atomic_numbers": "atomic_numbers.npy",
    "num_atoms": "num_atoms.npy",
    "structure_id": "structure_id.npy",
}


def _is_legacy_split(path: Path) -> bool:
    return all((path / fname).is_file() for fname in _LEGACY_CORE.values())


def _read_legacy_split(path: Path) -> CacheBundle:
    """Read the entire legacy split (every rank does this; cheap on memory)."""
    arrays = {key: np.load(path / fname) for key, fname in _LEGACY_CORE.items()}
    properties: dict[str, np.ndarray] = {}
    for entry in sorted(os.listdir(path)):
        if not entry.endswith(".json"):
            continue
        prop_name = entry[: -len(".json")]
        with open(path / entry, "r") as f:
            payload = json.load(f)
        properties[prop_name] = np.array(payload["values"])
    return CacheBundle(
        pos=arrays["pos"],
        cell=arrays["cell"],
        atomic_numbers=arrays["atomic_numbers"],
        num_atoms=arrays["num_atoms"],
        structure_id=arrays["structure_id"],
        properties=properties,
    )


def _slice_bundle(bundle: CacheBundle, start: int, end: int) -> CacheBundle:
    """Return the per-rank slab covering structures ``[start, end)``."""
    if start == 0 and end == len(bundle.num_atoms):
        return bundle
    # Atom-level offsets for the per-structure pos / atomic_numbers slabs.
    cum = np.concatenate([[0], np.cumsum(bundle.num_atoms)])
    atom_start = int(cum[start])
    atom_end = int(cum[end])
    return CacheBundle(
        pos=bundle.pos[atom_start:atom_end],
        cell=bundle.cell[start:end],
        atomic_numbers=bundle.atomic_numbers[atom_start:atom_end],
        num_atoms=bundle.num_atoms[start:end],
        structure_id=bundle.structure_id[start:end],
        properties={k: v[start:end] for k, v in bundle.properties.items()},
    )


def _convert_one(
    src: Path,
    dst: Path,
    fmt: str,
    in_place: bool,
    comm,
) -> None:
    rank = comm.Get_rank()
    size = comm.Get_size()

    bundle = _read_legacy_split(src)
    n = len(bundle.num_atoms)

    if size > 1:
        start, end = shard_range(n, rank, size)
        local = _slice_bundle(bundle, start, end)
    else:
        local = bundle

    if rank == 0:
        dst.mkdir(parents=True, exist_ok=True)
    if hasattr(comm, "Barrier"):
        comm.Barrier()

    write_cache(str(dst), local, fmt=fmt, comm=comm if size > 1 else None)

    if rank == 0:
        print(
            f"Wrote {fmt} cache to {dst} "
            f"({n} structures, {len(bundle.properties)} properties, ranks={size})"
        )

    if in_place and rank == 0:
        for fname in _LEGACY_CORE.values():
            (src / fname).unlink(missing_ok=True)
        for entry in list(os.listdir(src)):
            if entry.endswith(".json"):
                (src / entry).unlink(missing_ok=True)
    if hasattr(comm, "Barrier"):
        comm.Barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Source legacy cache directory.")
    parser.add_argument(
        "--dst",
        default=None,
        help="Destination directory. Defaults to --src when --in-place is set.",
    )
    parser.add_argument(
        "--format",
        default="pickle",
        choices=SUPPORTED_FORMATS,
        help="On-disk backend to write.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write the new cache into the source directory and delete the legacy files.",
    )
    parser.add_argument(
        "--mpi",
        action="store_true",
        help=(
            "Use mpi4py + collective IO to shard work across ranks. Requires "
            "the optional `mpi4py` extra (and an MPI-enabled `adios2` build "
            "for the ADIOS2 backend). Without this flag the conversion runs "
            "serially even under `mpirun`."
        ),
    )
    args = parser.parse_args()

    comm = auto_comm(force=True if args.mpi else False)
    rank = comm.Get_rank()

    src = Path(args.src).resolve()
    if args.dst is None:
        if not args.in_place:
            if rank == 0:
                parser.error("Provide either --dst or --in-place.")
            return
        dst = src
    else:
        dst = Path(args.dst).resolve()

    splits = [name for name in ("train", "val", "test") if (src / name).is_dir()]
    if splits:
        for split in splits:
            split_dst = dst / split if args.dst is not None else src / split
            _convert_one(src / split, split_dst, args.format, args.in_place, comm)
    else:
        if not _is_legacy_split(src):
            raise SystemExit(
                f"{src} does not look like a legacy .npy cache (missing core .npy files)."
            )
        _convert_one(src, dst, args.format, args.in_place, comm)


if __name__ == "__main__":
    main()
