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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import numpy as np
Block = dict[str, np.ndarray]
PLOT_HISTOGRAM_BINS = 25
PLOT_LOWER_QUANTILE = 0.005
PLOT_UPPER_QUANTILE = 0.995
ELEMENT_SYMBOLS = (
    "",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)


# Set before running
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_MODULE_CONFIG = REPO_ROOT / "mattergen" / "conf" / "data_module" / "miniOmat24.yaml"
DEFAULT_ADIOS_FILENAME = "miniOmat24/miniOMat24.bp"
SPLIT_PREFIXES = {"train": "trainset", "val": "valset", "test": "testset"}



@dataclass(frozen=True)
class Statistic:
    """A named calculation and the ADIOS fields it consumes."""

    name: str
    fields: tuple[str, ...]
    calculate: Callable[[Block], np.ndarray]
    accumulator: str = "stats"
    histogram_transform: Callable[[np.ndarray], np.ndarray] | None = None
    histogram_label: str | None = None

    def __call__(self, block: Block) -> np.ndarray:
        return self.calculate(block)


def statistic(
    *fields: str,
    accumulator: str = "stats",
    histogram_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    histogram_label: str | None = None,
) -> Callable[[Callable[[Block], np.ndarray]], Statistic]:
    """Make a function selectable in ``main`` and declare its dependencies."""

    def decorate(function: Callable[[Block], np.ndarray]) -> Statistic:
        if not fields:
            raise ValueError(f"Statistic {function.__name__!r} requires no fields.")
        return Statistic(
            function.__name__,
            fields,
            function,
            accumulator,
            histogram_transform,
            histogram_label,
        )

    return decorate


def _log10_positive(values: np.ndarray) -> np.ndarray:
    """Return log10 values for positive finite inputs."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.log10(values[np.isfinite(values) & (values > 0)])


@statistic("atomic_numbers")
def atomic_number(block: Block) -> np.ndarray:
    """One value per atom."""
    return np.asarray(block["atomic_numbers"], dtype=np.float64).reshape(-1)


@statistic("atomic_numbers", accumulator="elements")
def elements(block: Block) -> np.ndarray:
    """One atomic number per atom, later converted to unique element symbols."""
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


@statistic(
    "natoms",
    "forces",
    histogram_transform=_log10_positive,
    histogram_label="log10(rms_forces)",
)
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


@statistic("natoms", accumulator="distribution")
def num_atoms_distribution(block: Block) -> np.ndarray:
    """One integer atom count per structure."""
    natoms = np.asarray(block["natoms"], dtype=np.int64).reshape(-1)
    if np.any(natoms < 0):
        raise ValueError("natoms contains a negative value.")
    return natoms


def _select_statistics(items: list[Statistic]) -> list[Statistic]:
    selected = list({item.name: item for item in items}.values())
    if not selected:
        raise ValueError("Select at least one statistic in main().")
    return selected


def _required_fields(statistics: list[Statistic]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(field for item in statistics for field in item.fields))


@dataclass
class RunningHistogram:
    """Fixed-bin streaming histogram that can be merged across ranks."""

    bins: int = 80
    counts: np.ndarray | None = None
    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float = 0.0
    m2: float = 0.0

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = np.zeros(self.bins, dtype=np.int64)

    @staticmethod
    def _range(minimum: float, maximum: float) -> tuple[float, float]:
        if minimum == maximum:
            padding = max(abs(minimum) * 0.05, 0.5)
            return minimum - padding, maximum + padding
        return minimum, maximum

    @staticmethod
    def _edges(minimum: float, maximum: float, bins: int) -> np.ndarray:
        low, high = RunningHistogram._range(minimum, maximum)
        return np.linspace(low, high, bins + 1)

    @staticmethod
    def _rebin(
        counts: np.ndarray,
        old_minimum: float,
        old_maximum: float,
        new_minimum: float,
        new_maximum: float,
    ) -> np.ndarray:
        old_edges = RunningHistogram._edges(old_minimum, old_maximum, len(counts))
        new_edges = RunningHistogram._edges(new_minimum, new_maximum, len(counts))
        centers = (old_edges[:-1] + old_edges[1:]) / 2.0
        indices = np.searchsorted(new_edges, centers, side="right") - 1
        indices = np.clip(indices, 0, len(counts) - 1)
        rebinned = np.zeros_like(counts)
        np.add.at(rebinned, indices, counts)
        return rebinned

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        if not len(finite):
            return

        block_minimum = float(finite.min())
        block_maximum = float(finite.max())
        block_mean = float(finite.mean())
        centered = finite - block_mean
        block_m2 = float(np.sum(centered * centered))
        block_count = int(len(finite))
        assert self.counts is not None
        if self.minimum is None or self.maximum is None:
            self.minimum = block_minimum
            self.maximum = block_maximum
        else:
            new_minimum = min(self.minimum, block_minimum)
            new_maximum = max(self.maximum, block_maximum)
            if new_minimum != self.minimum or new_maximum != self.maximum:
                self.counts = self._rebin(
                    self.counts, self.minimum, self.maximum, new_minimum, new_maximum
                )
                self.minimum = new_minimum
                self.maximum = new_maximum

        low, high = self._range(self.minimum, self.maximum)
        counts, _ = np.histogram(finite, bins=self.bins, range=(low, high))
        self.counts += counts.astype(np.int64, copy=False)
        self._merge_moments(block_count, block_mean, block_m2)

    def _merge_moments(self, count: int, mean: float, m2: float) -> None:
        if not count:
            return
        if not self.count:
            self.count = count
            self.mean = mean
            self.m2 = m2
            return
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def merge(self, other: "RunningHistogram") -> None:
        assert self.counts is not None
        assert other.counts is not None
        if other.minimum is None or other.maximum is None:
            return
        if self.minimum is None or self.maximum is None:
            self.minimum = other.minimum
            self.maximum = other.maximum
            self.counts = other.counts.copy()
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            return

        new_minimum = min(self.minimum, other.minimum)
        new_maximum = max(self.maximum, other.maximum)
        self_counts = self._rebin(
            self.counts, self.minimum, self.maximum, new_minimum, new_maximum
        )
        other_counts = self._rebin(
            other.counts, other.minimum, other.maximum, new_minimum, new_maximum
        )
        self.minimum = new_minimum
        self.maximum = new_maximum
        self.counts = self_counts + other_counts
        self._merge_moments(other.count, other.mean, other.m2)

    def result(self) -> dict[str, Any]:
        assert self.counts is not None
        if self.minimum is None or self.maximum is None:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "bin_edges": [],
                "counts": [],
                "probabilities": [],
            }
        edges = self._edges(self.minimum, self.maximum, self.bins)
        total = self.count
        return {
            "count": total,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": math.sqrt(max(self.m2 / self.count, 0.0)) if self.count else None,
            "bin_edges": [float(value) for value in edges],
            "counts": [int(value) for value in self.counts],
            "probabilities": [
                float(value / total) if total else 0.0 for value in self.counts
            ],
        }


@dataclass
class RunningStats:
    """Mergeable count, extrema, mean, and population variance."""

    count: int = 0
    invalid_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float = 0.0
    m2: float = 0.0
    histogram: RunningHistogram | None = None
    histogram_transform: Callable[[np.ndarray], np.ndarray] | None = None
    histogram_label: str | None = None

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
        if self.histogram is not None:
            histogram_values = (
                self.histogram_transform(finite)
                if self.histogram_transform is not None
                else finite
            )
            self.histogram.update(histogram_values)

    def merge(self, other: RunningStats) -> None:
        self.invalid_count += other.invalid_count
        if self.histogram is not None and other.histogram is not None:
            self.histogram.merge(other.histogram)
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

    def result(self) -> dict[str, Any]:
        if not self.count:
            return {
                "count": 0,
                "invalid_count": self.invalid_count,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "histogram_label": self.histogram_label,
                "histogram": self.histogram.result() if self.histogram else None,
            }
        return {
            "count": self.count,
            "invalid_count": self.invalid_count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": math.sqrt(max(self.m2 / self.count, 0.0)),
            "histogram_label": self.histogram_label,
            "histogram": self.histogram.result() if self.histogram else None,
        }


@dataclass
class RunningDistribution:
    """Mergeable histogram with probabilities computed at the end."""

    counts: dict[int, int] | None = None
    invalid_count: int = 0

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = {}

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values).reshape(-1)
        valid = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.number) else values
        self.invalid_count += len(values) - len(valid)
        unique, counts = np.unique(valid.astype(np.int64, copy=False), return_counts=True)
        assert self.counts is not None
        for value, count in zip(unique, counts):
            self.counts[int(value)] = self.counts.get(int(value), 0) + int(count)

    def merge(self, other: "RunningDistribution") -> None:
        self.invalid_count += other.invalid_count
        assert self.counts is not None
        assert other.counts is not None
        for value, count in other.counts.items():
            self.counts[value] = self.counts.get(value, 0) + count

    def result(self) -> dict[str, Any]:
        assert self.counts is not None
        total = sum(self.counts.values())
        table = [
            {
                "num_atoms": value,
                "count": count,
                "probability": count / total if total else None,
            }
            for value, count in sorted(self.counts.items())
        ]
        return {
            "count": total,
            "invalid_count": self.invalid_count,
            "table": table,
            "probabilities": {
                str(row["num_atoms"]): row["probability"] for row in table
            },
        }


@dataclass
class RunningElements:
    """Mergeable set of element symbols observed in atomic-number data."""

    elements: set[str] | None = None
    invalid_count: int = 0

    def __post_init__(self) -> None:
        if self.elements is None:
            self.elements = set()

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        rounded = np.rint(finite).astype(np.int64, copy=False)
        valid = (
            (finite == rounded)
            & (rounded > 0)
            & (rounded < len(ELEMENT_SYMBOLS))
        )
        self.invalid_count += len(values) - int(np.count_nonzero(valid))
        assert self.elements is not None
        for atomic_number in np.unique(rounded[valid]):
            self.elements.add(ELEMENT_SYMBOLS[int(atomic_number)])

    def merge(self, other: "RunningElements") -> None:
        self.invalid_count += other.invalid_count
        assert self.elements is not None
        assert other.elements is not None
        self.elements.update(other.elements)

    def result(self) -> dict[str, Any]:
        assert self.elements is not None
        return {
            "elements": sorted(self.elements),
            "invalid_count": self.invalid_count,
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


def _read_root_dir_from_data_module_config(config_path: Path) -> Path:
    """Return the first uncommented root_dir from a Hydra data-module YAML."""
    text = config_path.read_text()
    match = re.search(r"^[ \t]*root_dir[ \t]*:[ \t]*(?P<path>[^#\n]+)", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not find root_dir in {config_path}")
    raw_path = match.group("path").strip().strip("'\"")
    raw_path = raw_path.replace("${oc.env:PROJECT_ROOT}", str(REPO_ROOT))
    return Path(os.path.expandvars(os.path.expanduser(raw_path)))


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
        if name not in variables:
            users = [item.name for item in statistics if field in item.fields]
            raise KeyError(f"Split {prefix!r} lacks {name!r}; required by {users}.")
        has_offsets = f"{name}/variable_count" in variables and f"{name}/variable_offset" in variables
        if field != "natoms" and not has_offsets:
            users = [item.name for item in statistics if field in item.fields]
            missing = [
                key
                for key in (f"{name}/variable_count", f"{name}/variable_offset")
                if key not in variables
            ]
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
        if "all" in requested_splits:
            requested_splits = [
                split
                for split, prefix in SPLIT_PREFIXES.items()
                if any(name.startswith(f"{prefix}/") for name in variables)
            ]
            if not requested_splits:
                raise FileNotFoundError("Could not find any known train/val/test split.")

        for requested_split in requested_splits:
            prefix = _choose_split_prefix(requested_split, variables)
            fields = _validate_required_variables(prefix, variables, statistics)
            variable_plans = {}
            num_structures = None

            for field in fields:
                name = f"{prefix}/{field}"
                offset_name = f"{name}/variable_offset"
                count_name = f"{name}/variable_count"
                shape = _parse_shape(variables[name]["Shape"], name)
                if offset_name in variables and count_name in variables:
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
                    dim = _variable_dim(reader, attributes, name, shape, num_structures)
                    variable_plans[field] = {
                        "layout": "packed",
                        "name": name,
                        "shape": shape,
                        "dim": dim,
                        "offset": offset_name,
                        "count": count_name,
                    }
                else:
                    if len(shape) != 1:
                        raise ValueError(
                            f"Dense field {name!r} must be one-dimensional; got {shape}."
                        )
                    if num_structures is None:
                        num_structures = shape[0]
                    elif shape[0] != num_structures:
                        raise ValueError(
                            f"{name!r} describes {shape[0]} structures; "
                            f"expected {num_structures}."
                        )
                    variable_plans[field] = {
                        "layout": "dense",
                        "name": name,
                        "shape": shape,
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
    if metadata["layout"] == "dense":
        return np.asarray(
            reader.read(
                metadata["name"],
                start=[start],
                count=[count],
                step_selection=[0, 1],
            )
        )

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


def _empty_stats(statistics: list[Statistic], histogram_bins: int) -> dict[str, Any]:
    accumulators: dict[str, Any] = {}
    for item in statistics:
        if item.accumulator == "stats":
            accumulators[item.name] = RunningStats(
                histogram=RunningHistogram(histogram_bins),
                histogram_transform=item.histogram_transform,
                histogram_label=item.histogram_label,
            )
        elif item.accumulator == "distribution":
            accumulators[item.name] = RunningDistribution()
        elif item.accumulator == "elements":
            accumulators[item.name] = RunningElements()
        else:
            raise ValueError(f"Unknown accumulator type {item.accumulator!r}.")
    return accumulators


def _process_rank(
    plan: dict[str, Any],
    statistics: list[Statistic],
    histogram_bins: int,
    rank: int,
    progress_every: int,
) -> dict[str, Any]:
    accumulators = _empty_stats(statistics, histogram_bins)
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
                    if metadata["layout"] == "dense":
                        block[field] = _read_block(
                            reader, metadata, structure_start, structure_count
                        )
                    else:
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
    rank_stats: list[dict[str, Any]], statistics: list[Statistic], histogram_bins: int
) -> dict[str, dict[str, Any]]:
    if not rank_stats:
        raise ValueError("No rank statistics were supplied.")
    merged = _empty_stats(statistics, histogram_bins)
    for local in rank_stats:
        for name in merged:
            merged[name].merge(local[name])
    return {name: accumulator.result() for name, accumulator in merged.items()}


def _weighted_point_quantile(
    values: np.ndarray, counts: np.ndarray, quantile: float
) -> float:
    if not len(values):
        return math.nan
    total = int(counts.sum())
    if total <= 0:
        return math.nan
    cumulative = np.cumsum(counts)
    target = max(1, min(total, int(math.ceil(quantile * total))))
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(values[min(index, len(values) - 1)])


def _weighted_mean_std(values: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    total = int(counts.sum())
    if total <= 0:
        return math.nan, math.nan
    mean = float(np.average(values, weights=counts))
    variance = float(np.average((values - mean) * (values - mean), weights=counts))
    return mean, math.sqrt(max(variance, 0.0))


def _weighted_binned_quantile(
    edges: np.ndarray, counts: np.ndarray, quantile: float
) -> float:
    total = int(counts.sum())
    if total <= 0:
        return math.nan
    cumulative = np.cumsum(counts)
    target = quantile * total
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), len(counts) - 1)
    previous = float(cumulative[index - 1]) if index else 0.0
    bin_count = int(counts[index])
    if bin_count <= 0:
        return float((edges[index] + edges[index + 1]) / 2.0)
    fraction = (target - previous) / bin_count
    fraction = min(max(fraction, 0.0), 1.0)
    return float(edges[index] + fraction * (edges[index + 1] - edges[index]))


def _plot_range(low: float, high: float, minimum: float, maximum: float) -> tuple[float, float]:
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        low, high = minimum, maximum
    if low == high:
        padding = max(abs(low) * 0.05, 0.5)
        return low - padding, high + padding
    return low, high


def _annotate_stats(
    axis: Any,
    minimum: float | None,
    maximum: float | None,
    mean: float | None,
    sigma: float | None,
) -> None:
    if minimum is None or maximum is None or mean is None or sigma is None:
        return
    axis.text(
        0.98,
        0.96,
        f"min={minimum:.6g}\nmax={maximum:.6g}\nmean={mean:.6g}\nsig={sigma:.6g}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.85,
        },
    )


def _plot_histogram(
    axis: Any,
    name: str,
    payload: dict[str, Any],
) -> bool:
    if "table" in payload:
        table = payload["table"]
        if not table:
            return False
        xs = np.asarray([row["num_atoms"] for row in table], dtype=np.float64)
        counts = np.asarray([row["count"] for row in table], dtype=np.int64)
        minimum = float(xs.min())
        maximum = float(xs.max())
        mean, sigma = _weighted_mean_std(xs, counts)
        low = _weighted_point_quantile(xs, counts, PLOT_LOWER_QUANTILE)
        high = _weighted_point_quantile(xs, counts, PLOT_UPPER_QUANTILE)
        low, high = _plot_range(low, high, minimum, maximum)
        visible = (xs >= low) & (xs <= high)
        axis.bar(
            xs[visible],
            counts[visible],
            width=0.9,
            color="#4c78a8",
            edgecolor="white",
            linewidth=0.4,
        )
        axis.set_xlim(low - 0.5, high + 0.5)
        axis.set_xlabel("num_atoms")
        _annotate_stats(axis, minimum, maximum, mean, sigma)
    else:
        histogram = payload.get("histogram")
        if not histogram or not histogram.get("counts"):
            return False
        edges = np.asarray(histogram["bin_edges"], dtype=np.float64)
        counts = np.asarray(histogram["counts"], dtype=np.int64)
        if len(edges) != len(counts) + 1:
            return False
        minimum = payload.get("min")
        maximum = payload.get("max")
        if minimum is None or maximum is None:
            return False
        histogram_minimum = histogram.get("min")
        histogram_maximum = histogram.get("max")
        histogram_mean = histogram.get("mean")
        histogram_std = histogram.get("std")
        if histogram_minimum is None or histogram_maximum is None:
            histogram_minimum = float(edges[0])
            histogram_maximum = float(edges[-1])
        low = _weighted_binned_quantile(edges, counts, PLOT_LOWER_QUANTILE)
        high = _weighted_binned_quantile(edges, counts, PLOT_UPPER_QUANTILE)
        low, high = _plot_range(low, high, float(histogram_minimum), float(histogram_maximum))
        centers = (edges[:-1] + edges[1:]) / 2.0
        visible = (centers >= low) & (centers <= high)
        visible_counts, visible_edges = np.histogram(
            centers[visible],
            bins=PLOT_HISTOGRAM_BINS,
            range=(low, high),
            weights=counts[visible],
        )
        axis.bar(
            visible_edges[:-1],
            visible_counts,
            width=np.diff(visible_edges),
            align="edge",
            color="#4c78a8",
            edgecolor="white",
            linewidth=0.4,
        )
        axis.set_xlim(low, high)
        axis.set_xlabel(payload.get("histogram_label") or name)
        _annotate_stats(
            axis,
            float(histogram_minimum),
            float(histogram_maximum),
            None if histogram_mean is None else float(histogram_mean),
            None if histogram_std is None else float(histogram_std),
        )

    axis.set_title(name.replace("_", " "))
    axis.set_ylabel("count")
    axis.grid(True, axis="y", alpha=0.25)
    return True


def _write_histogram_plots(result: dict[str, Any], output_dir: Path) -> list[Path]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipped histogram plots")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_items = [
        (name, payload)
        for name, payload in result["statistics"].items()
        if "table" in payload or payload.get("histogram")
    ]
    if not plot_items:
        return []

    columns = 2 if len(plot_items) > 1 else 1
    rows = math.ceil(len(plot_items) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 4.2 * rows))
    axes_array = np.asarray(axes, dtype=object).reshape(rows, columns)
    for axis, (name, payload) in zip(axes_array.reshape(-1), plot_items):
        _plot_histogram(axis, name, payload)
    for axis in axes_array.reshape(-1)[len(plot_items):]:
        axis.set_visible(False)

    figure.tight_layout()
    combined_path = output_dir / "statistics_histograms.png"
    figure.savefig(combined_path, dpi=180)
    plt.close(figure)
    return [combined_path]


def _default_plot_dir(args: argparse.Namespace) -> Path:
    if args.plot_dir:
        return Path(args.plot_dir).resolve()
    if args.out:
        output_path = Path(args.out).resolve()
        return output_path.with_name(f"{output_path.stem}_histograms")
    return Path("dataset_stat_histograms").resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Dataset root or .bp path. Defaults to root_dir from "
            f"{DEFAULT_DATA_MODULE_CONFIG}."
        ),
    )
    parser.add_argument(
        "--data-module-config",
        default=str(DEFAULT_DATA_MODULE_CONFIG),
        help="Data-module YAML to read root_dir from when --root is omitted.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["all"],
        help="Splits to process, or 'all' for every available train/val/test split.",
    )
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=80,
        help=(
            "Number of streaming bins stored in JSON for continuous statistics. "
            f"PNG histograms are re-binned to {PLOT_HISTOGRAM_BINS} visible bins."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        help=(
            "Directory for histogram PNGs. Defaults to <out-stem>_histograms when "
            "--out is set, otherwise ./dataset_stat_histograms."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Compute histogram data but do not write histogram PNGs.",
    )
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if args.histogram_bins <= 0:
        parser.error("--histogram-bins must be positive")
    return args

def main() -> None:
    # Comment out any statistic you do not need.
    statistics = []
    statistics.append(num_atoms_distribution)
    statistics.append(atomic_number)
    statistics.append(elements)
    statistics.append(density)
    statistics.append(rms_forces)
    statistics = _select_statistics(statistics)

    # Parse arguments and distribute the read plan to all ranks.
    args = _parse_args()
    root = Path(args.root) if args.root else _read_root_dir_from_data_module_config(
        Path(args.data_module_config)
    )
    comm, rank, size = _get_mpi()
    plan = _distribute_plan(
        root, args.splits, args.block_size, statistics, comm, rank, size
    )
    if rank == 0:
        print(
            f"Reading {plan['num_structures']} structures from {plan['path']} with "
            f"{size} rank(s), block_size={args.block_size}, statistics={plan['statistics']}",
            flush=True,
        )

    local_stats = error = None
    try:
        local_stats = _process_rank(
            plan, statistics, args.histogram_bins, rank, args.progress_every
        )
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors = comm.allgather(error) if comm is not None else [error]
    if any(errors):
        raise RuntimeError("ADIOS block processing failed:\n" + "\n".join(filter(None, errors)))
    assert local_stats is not None

    gathered = comm.gather(local_stats, root=0) if comm is not None else [local_stats]
    if rank != 0:
        return
    statistics_result = _merge_stats(gathered, statistics, args.histogram_bins)
    elements_result = statistics_result.pop("elements", {})
    result = {
        "metadata": {
            "adios_path": plan["path"],
            "splits": [split["requested_name"] for split in plan["splits"]],
            "num_structures": plan["num_structures"],
            "block_size": plan["block_size"],
            "world_size": plan["world_size"],
        },
        "elements": elements_result.get("elements", []),
        "statistics": statistics_result,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        output_path = Path(args.out).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {output_path}")
    if not args.no_plots:
        plot_paths = _write_histogram_plots(result, _default_plot_dir(args))
        for plot_path in plot_paths:
            print(f"Wrote histogram plot {plot_path}")


if __name__ == "__main__":
    main()
