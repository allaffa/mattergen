"""Compute bounded-memory statistics from a HydraGNN ADIOS dataset.

Statistics declare the ADIOS fields they need. Shared fields are read once per
contiguous structure block, and mergeable online moments keep memory bounded.
ADIOS handles the ``data.*`` shards inside a ``.bp`` directory automatically.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import numpy as np
Block = dict[str, np.ndarray]


# Set before running
DEFAULT_ADIOS_FILENAME = "miniOmat24/miniOMat24.bp"
SPLIT_PREFIXES = {"train": "trainset", "val": "valset", "test": "testset"}



@dataclass(frozen=True)
class Statistic:
    """A named calculation and the ADIOS fields it consumes."""

    name: str
    fields: tuple[str, ...]
    calculate: Callable[[Block], np.ndarray]

    def __call__(self, block: Block) -> np.ndarray:
        return self.calculate(block)


def statistic(*fields: str) -> Callable[[Callable[[Block], np.ndarray]], Statistic]:
    """Make a function selectable in ``main`` and declare its dependencies."""

    def decorate(function: Callable[[Block], np.ndarray]) -> Statistic:
        if not fields:
            raise ValueError(f"Statistic {function.__name__!r} requires no fields.")
        return Statistic(function.__name__, fields, function)

    return decorate


@statistic("atomic_numbers")
def atomic_number(block: Block) -> np.ndarray:
    """One value per atom."""
    return np.asarray(block["atomic_numbers"], dtype=np.float64).reshape(-1)


@statistic("natoms", "cell")
def density(block: Block) -> np.ndarray:
    """Number density per structure: natoms / abs(det(cell))."""
    natoms = np.asarray(block["natoms"], dtype=np.int64).reshape(-1)
    cells = np.asarray(block["cell"], dtype=np.float64)
    if cells.size != len(natoms) * 9:
        raise ValueError(f"Expected {len(natoms) * 9} cell values, got {cells.size}.")

    volumes = np.abs(np.linalg.det(cells.reshape(-1, 3, 3)))
    values = np.full(len(natoms), np.nan)
    valid = np.isfinite(volumes) & (volumes > 0)
    values[valid] = natoms[valid] / volumes[valid]
    return values


@statistic("natoms", "forces")
def rms_forces(block: Block) -> np.ndarray:
    """Per-structure sqrt(mean_atom(Fx^2 + Fy^2 + Fz^2))."""
    natoms = np.asarray(block["natoms"], dtype=np.int64).reshape(-1)
    if np.any(natoms < 0):
        raise ValueError("natoms contains a negative value.")

    forces = np.asarray(block["forces"], dtype=np.float64)
    if forces.size % 3:
        raise ValueError(f"forces has {forces.size} values; expected a multiple of 3.")
    forces = forces.reshape(-1, 3)
    if len(forces) != int(natoms.sum()):
        raise ValueError(f"natoms sums to {natoms.sum()}, but forces has {len(forces)} rows.")

    values = np.full(len(natoms), np.nan)
    nonempty = natoms > 0
    if np.any(nonempty):
        ends = np.cumsum(natoms)
        starts = ends - natoms
        squared_norms = np.sum(forces * forces, axis=1, dtype=np.float64)
        values[nonempty] = np.sqrt(
            np.add.reduceat(squared_norms, starts[nonempty]) / natoms[nonempty]
        )
    return values


def _select_statistics(items: list[Statistic]) -> list[Statistic]:
    selected = list({item.name: item for item in items}.values())
    if not selected:
        raise ValueError("Select at least one statistic in main().")
    return selected


def _required_fields(statistics: list[Statistic]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(field for item in statistics for field in item.fields))


@dataclass
class RunningStats:
    """Mergeable count, extrema, mean, and population variance."""

    count: int = 0
    invalid_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        invalid = len(values) - len(finite)
        if not len(finite):
            self.invalid_count += invalid
            return

        mean = float(finite.mean())
        centered = finite - mean
        self.merge(
            RunningStats(
                count=len(finite),
                invalid_count=invalid,
                minimum=float(finite.min()),
                maximum=float(finite.max()),
                mean=mean,
                m2=float(np.sum(centered * centered)),
            )
        )

    def merge(self, other: RunningStats) -> None:
        self.invalid_count += other.invalid_count
        if not other.count:
            return
        if not self.count:
            self.count = other.count
            self.minimum = other.minimum
            self.maximum = other.maximum
            self.mean = other.mean
            self.m2 = other.m2
            return

        assert self.minimum is not None and self.maximum is not None
        assert other.minimum is not None and other.maximum is not None
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = total
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)

    def result(self) -> dict[str, int | float | None]:
        if not self.count:
            return {
                "count": 0,
                "invalid_count": self.invalid_count,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }
        return {
            "count": self.count,
            "invalid_count": self.invalid_count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": math.sqrt(max(self.m2 / self.count, 0.0)),
        }


def _import_adios2():
    try:
        import adios2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install the optional `adios2` package to read ADIOS data.") from exc
    return adios2


def _get_mpi() -> tuple[Any | None, int, int]:
    try:
        from mpi4py import MPI
    except ImportError as exc:
        size = int(
            os.getenv(
                "OMPI_COMM_WORLD_SIZE",
                os.getenv("PMI_SIZE", os.getenv("SLURM_NTASKS", "1")),
            )
        )
        if size > 1:
            raise RuntimeError(f"Detected {size} MPI ranks without mpi4py.") from exc
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def _find_adios_file(path: Path) -> Path:
    if path.name.endswith(".bp") and (path.is_file() or path.is_dir()):
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"ADIOS path does not exist: {path}")
    conventional = path / DEFAULT_ADIOS_FILENAME
    if conventional.exists():
        return conventional
    candidates = list(path.glob("*.bp"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No .bp dataset found under {path}")
    raise FileNotFoundError(f"Multiple .bp datasets found under {path}; pass one directly.")


def _choose_split_prefix(split: str, variables: dict[str, Any]) -> str:
    candidates = list(
        dict.fromkeys(
            filter(None, (SPLIT_PREFIXES.get(split), split, f"{split}set", f"{split}split"))
        )
    )
    for candidate in candidates:
        if any(name.startswith(f"{candidate}/") for name in variables):
            return candidate
    raise FileNotFoundError(f"Could not resolve split {split!r}; tried {candidates}.")


def _validate_required_variables(
    prefix: str,
    variables: dict[str, Any],
    statistics: list[Statistic],
) -> tuple[str, ...]:
    fields = _required_fields(statistics)
    for field in fields:
        name = f"{prefix}/{field}"
        required = (name, f"{name}/variable_count", f"{name}/variable_offset")
        missing = [key for key in required if key not in variables]
        if missing:
            users = [item.name for item in statistics if field in item.fields]
            raise KeyError(f"Split {prefix!r} lacks {missing}; required by {users}.")
    return fields


def _parse_shape(shape: Any, name: str) -> tuple[int, ...]:
    values = re.findall(r"\d+", shape) if isinstance(shape, str) else shape
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise ValueError(f"ADIOS variable {name!r} has no global shape.")
    return parsed


def _metadata_length(variables: dict[str, Any], name: str) -> int:
    shape = _parse_shape(variables[name]["Shape"], name)
    if len(shape) != 1:
        raise ValueError(f"ADIOS metadata variable {name!r} must be one-dimensional.")
    return shape[0]


def _read_metadata_slice(reader: Any, name: str, start: int, count: int) -> np.ndarray:
    return np.asarray(
        reader.read(name, start=[start], count=[count], step_selection=[0, 1]),
        dtype=np.int64,
    ).reshape(-1)


def _variable_dim(
    reader: Any,
    attributes: dict[str, Any],
    name: str,
    shape: tuple[int, ...],
    num_structures: int,
) -> int:
    attribute = f"{name}/variable_dim"
    if attribute in attributes:
        dim = int(np.asarray(reader.read_attribute(attribute)).reshape(-1)[0])
    else:
        if not num_structures:
            raise ValueError(f"Cannot infer missing {attribute!r} for an empty split.")
        offset = _read_metadata_slice(
            reader, f"{name}/variable_offset", num_structures - 1, 1
        )[0]
        count = _read_metadata_slice(
            reader, f"{name}/variable_count", num_structures - 1, 1
        )[0]
        extent = int(offset + count)
        candidates = [dim for dim, size in enumerate(shape) if size == extent]
        if len(candidates) != 1:
            raise ValueError(f"Cannot infer missing {attribute!r} from shape {shape}.")
        dim = candidates[0]
    if dim not in range(len(shape)):
        raise ValueError(f"Invalid variable_dim={dim} for {name!r} with shape {shape}.")
    return dim


def _packed_selection(
    name: str,
    shape: tuple[int, ...],
    dim: int,
    offsets: np.ndarray,
    counts: np.ndarray,
    num_structures: int,
) -> tuple[int, int]:
    if len(offsets) != num_structures or len(counts) != num_structures:
        raise ValueError(
            f"{name!r} has {len(offsets)} offsets and {len(counts)} counts "
            f"for a {num_structures}-structure block."
        )
    if np.any(offsets < 0) or np.any(counts < 0):
        raise ValueError(f"{name!r} has a negative offset or count.")
    if len(offsets) > 1 and np.any(offsets[1:] != offsets[:-1] + counts[:-1]):
        raise ValueError(f"{name!r} is not contiguously packed by structure.")
    if not len(offsets):
        return 0, 0
    packed_start = int(offsets[0])
    packed_count = sum(int(count) for count in counts)
    if packed_start + packed_count > shape[dim]:
        raise ValueError(f"Packed data for {name!r} exceeds shape {shape}.")
    return packed_start, packed_count


def _rank_block_indices(
    num_structures: int, block_size: int, rank: int, world_size: int
) -> range:
    num_blocks = (num_structures + block_size - 1) // block_size
    first = num_blocks * rank // world_size
    stop = num_blocks * (rank + 1) // world_size
    return range(first, stop)


def _rank_blocks(
    num_structures: int, block_size: int, rank: int, world_size: int
):
    """Yield this rank's contiguous ``(structure_start, count)`` blocks."""
    for block in _rank_block_indices(num_structures, block_size, rank, world_size):
        start = block * block_size
        yield start, min(block_size, num_structures - start)


def _build_read_plan(
    path: Path,
    requested_splits: list[str],
    block_size: int,
    world_size: int,
    statistics: list[Statistic],
) -> dict[str, Any]:
    split_plans = []
    adios2 = _import_adios2()
    with adios2.FileReader(str(path)) as reader:
        variables = reader.available_variables()
        attributes = reader.available_attributes()
        for requested_split in requested_splits:
            prefix = _choose_split_prefix(requested_split, variables)
            fields = _validate_required_variables(prefix, variables, statistics)
            variable_plans = {}
            num_structures = None

            for field in fields:
                name = f"{prefix}/{field}"
                offset_name = f"{name}/variable_offset"
                count_name = f"{name}/variable_count"
                offset_length = _metadata_length(variables, offset_name)
                count_length = _metadata_length(variables, count_name)
                if offset_length != count_length:
                    raise ValueError(
                        f"{name!r} has {offset_length} offsets and {count_length} counts."
                    )
                if num_structures is None:
                    num_structures = offset_length
                elif offset_length != num_structures:
                    raise ValueError(
                        f"{name!r} describes {offset_length} structures; "
                        f"expected {num_structures}."
                    )
                shape = _parse_shape(variables[name]["Shape"], name)
                dim = _variable_dim(reader, attributes, name, shape, num_structures)
                variable_plans[field] = {
                    "name": name,
                    "shape": shape,
                    "dim": dim,
                    "offset": offset_name,
                    "count": count_name,
                }

            assert num_structures is not None
            ndata_name = f"{prefix}/ndata"
            if ndata_name in attributes:
                ndata = int(np.asarray(reader.read_attribute(ndata_name)).reshape(-1)[0])
                if ndata != num_structures:
                    raise ValueError(f"{ndata_name!r} says {ndata}; expected {num_structures}.")
            split_plans.append(
                {
                    "requested_name": requested_split,
                    "prefix": prefix,
                    "num_structures": num_structures,
                    "variables": variable_plans,
                }
            )

    return {
        "path": str(path),
        "block_size": block_size,
        "world_size": world_size,
        "num_structures": sum(split["num_structures"] for split in split_plans),
        "statistics": [item.name for item in statistics],
        "fields": _required_fields(statistics),
        "splits": split_plans,
    }


def _distribute_plan(
    root: Path,
    splits: list[str],
    block_size: int,
    statistics: list[Statistic],
    comm: Any | None,
    rank: int,
    size: int,
) -> dict[str, Any]:
    payload = None
    if rank == 0:
        try:
            plan = _build_read_plan(
                _find_adios_file(root.resolve()), splits, block_size, size, statistics
            )
            payload = {"plan": plan, "error": None}
        except Exception as exc:
            payload = {"plan": None, "error": f"{type(exc).__name__}: {exc}"}

    payload = comm.bcast(payload, root=0) if comm is not None else payload
    assert payload is not None
    if payload["error"]:
        raise RuntimeError(f"Could not prepare the ADIOS read plan: {payload['error']}")
    return payload["plan"]


def _read_block(reader: Any, metadata: dict[str, Any], start: int, count: int) -> np.ndarray:
    selection_start = [0] * len(metadata["shape"])
    selection_count = list(metadata["shape"])
    selection_start[metadata["dim"]] = start
    selection_count[metadata["dim"]] = count
    if count:
        value = np.asarray(
            reader.read(
                metadata["name"],
                start=selection_start,
                count=selection_count,
                step_selection=[0, 1],
            )
        )
    else:
        value = np.empty(selection_count)
    return np.moveaxis(value, metadata["dim"], 0) if metadata["dim"] else value


def _empty_stats(statistics: list[Statistic]) -> dict[str, RunningStats]:
    return {item.name: RunningStats() for item in statistics}


def _process_rank(
    plan: dict[str, Any],
    statistics: list[Statistic],
    rank: int,
    progress_every: int,
) -> dict[str, RunningStats]:
    accumulators = _empty_stats(statistics)
    total = sum(
        len(
            _rank_block_indices(
                split["num_structures"], plan["block_size"], rank, plan["world_size"]
            )
        )
        for split in plan["splits"]
    )
    completed = 0
    with _import_adios2().FileReader(plan["path"]) as reader:
        for split in plan["splits"]:
            for structure_start, structure_count in _rank_blocks(
                split["num_structures"], plan["block_size"], rank, plan["world_size"]
            ):
                block = {}
                for field in plan["fields"]:
                    metadata = split["variables"][field]
                    offsets = _read_metadata_slice(
                        reader, metadata["offset"], structure_start, structure_count
                    )
                    counts = _read_metadata_slice(
                        reader, metadata["count"], structure_start, structure_count
                    )
                    packed_start, packed_count = _packed_selection(
                        metadata["name"],
                        metadata["shape"],
                        metadata["dim"],
                        offsets,
                        counts,
                        structure_count,
                    )
                    block[field] = _read_block(
                        reader, metadata, packed_start, packed_count
                    )
                structure_stop = structure_start + structure_count
                for item in statistics:
                    try:
                        accumulators[item.name].update(item(block))
                    except Exception as exc:
                        raise ValueError(
                            f"{item.name} failed for {split['prefix']} structures "
                            f"[{int(structure_start)}, {structure_stop}): {exc}"
                        ) from exc
                completed += 1
                if progress_every and completed % progress_every == 0:
                    print(f"rank {rank}: processed {completed}/{total} blocks", file=sys.stderr)
    return accumulators


def _merge_stats(
    rank_stats: list[dict[str, RunningStats]], statistics: list[Statistic]
) -> dict[str, dict[str, int | float | None]]:
    if not rank_stats:
        raise ValueError("No rank statistics were supplied.")
    merged = _empty_stats(statistics)
    for local in rank_stats:
        for name in merged:
            merged[name].merge(local[name])
    return {name: accumulator.result() for name, accumulator in merged.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Dataset root or .bp path.")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    return args

def main() -> None:
    # Comment out any statistic you do not need.
    statistics = []
    statistics.append(atomic_number)
    statistics.append(density)
    statistics.append(rms_forces)
    statistics = _select_statistics(statistics)

    # Parse arguments and distribute the read plan to all ranks.
    args = _parse_args()
    comm, rank, size = _get_mpi()
    plan = _distribute_plan(
        Path(args.root), args.splits, args.block_size, statistics, comm, rank, size
    )
    if rank == 0:
        print(
            f"Reading {plan['num_structures']} structures from {plan['path']} with "
            f"{size} rank(s), block_size={args.block_size}, statistics={plan['statistics']}",
            flush=True,
        )

    local_stats = error = None
    try:
        local_stats = _process_rank(plan, statistics, rank, args.progress_every)
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors = comm.allgather(error) if comm is not None else [error]
    if any(errors):
        raise RuntimeError("ADIOS block processing failed:\n" + "\n".join(filter(None, errors)))
    assert local_stats is not None

    gathered = comm.gather(local_stats, root=0) if comm is not None else [local_stats]
    if rank != 0:
        return
    result = {
        "metadata": {
            "adios_path": plan["path"],
            "splits": [split["requested_name"] for split in plan["splits"]],
            "num_structures": plan["num_structures"],
            "block_size": plan["block_size"],
            "world_size": plan["world_size"],
        },
        "statistics": _merge_stats(gathered, statistics),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        output_path = Path(args.out).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
