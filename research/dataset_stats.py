from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


PICKLE_FILENAME = "dataset.pkl"
ADIOS_FILENAME = "miniOmat24/miniOMat24.bp"
LEGACY_FILES = ("atomic_numbers.npy", "num_atoms.npy", "cell.npy")


def _get_mpi_info() -> tuple[int, int]:
    """Get MPI rank and size. Requires mpi4py when mpirun -n >1 is detected."""
    size_env = os.getenv("OMPI_COMM_WORLD_SIZE")
    rank_env = os.getenv("OMPI_COMM_WORLD_RANK")
    
    if size_env and rank_env:
        size = int(size_env)
        rank = int(rank_env)
        # If mpirun is being used with >1 rank, we MUST have mpi4py
        if size > 1:
            try:
                import mpi4py  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    f"Detected mpirun with {size} ranks but mpi4py is not installed. "
                    "Install mpi4py to enable MPI parallelization, or run with mpirun -n 1."
                ) from e
        return size, rank
    
    # Try mpi4py as fallback
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm.Get_size(), comm.Get_rank()
    except ImportError:
        return 1, 0


def _is_cache(path: Path) -> bool:
    return (
        (path / PICKLE_FILENAME).is_file()
        or (path / ADIOS_FILENAME).exists()
        or all((path / name).is_file() for name in LEGACY_FILES)
    )


def _find_adios_file(path: Path) -> Path | None:
    if (path.is_file() or path.is_dir()) and path.name.endswith(".bp"):
        return path
    if path.is_dir():
        candidate = path / ADIOS_FILENAME
        if candidate.exists():
            return candidate
        bp_files = list(path.glob("*.bp"))
        if len(bp_files) == 1:
            return bp_files[0]
    return None


def _is_raw_omat_adios(path: Path) -> bool:
    return _find_adios_file(path) is not None


def _split_paths(root: Path, splits: list[str]) -> list[Path]:
    if _is_raw_omat_adios(root):
        return [root]
    if _is_cache(root):
        return [root]
    paths = [root / split for split in splits]
    missing = [path for path in paths if not _is_cache(path)]
    if missing:
        raise FileNotFoundError(
            "Could not find cache data for: " + ", ".join(str(path) for path in missing)
        )
    return paths


def _read_pickle(path: Path) -> dict[str, np.ndarray]:
    with (path / PICKLE_FILENAME).open("rb") as f:
        payload = pickle.load(f)
    return {
        "atomic_numbers": np.asarray(payload["atomic_numbers"]),
        "num_atoms": np.asarray(payload["num_atoms"]),
        "cell": np.asarray(payload["cell"]),
    }


def _import_adios2():
    try:
        import adios2  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Reading ADIOS caches requires the `adios2` Python package."
        ) from e
    return adios2


def _read_adios(path: Path) -> dict[str, np.ndarray]:
    adios2 = _import_adios2()
    out: dict[str, np.ndarray] = {}
    with adios2.Stream(str(path / ADIOS_FILENAME), "r") as stream:
        for _ in stream.steps():
            out["atomic_numbers"] = np.asarray(stream.read("atomic_numbers"))
            out["num_atoms"] = np.asarray(stream.read("num_atoms"))
            out["cell"] = np.asarray(stream.read("cell"))
            break
    return out


def _read_raw_var(reader, name: str, start: int = 0, count: int = -1) -> np.ndarray:
    if count < 0:
        return np.asarray(reader.read(name, step_selection=[0, 1]))
    return np.asarray(reader.read(name, start=[start], count=[count], step_selection=[0, 1]))


def _choose_raw_split_prefix(split: str, variables: dict[str, Any]) -> str:
    split_prefixes = {"train": "trainset", "val": "valset", "test": "testset"}
    candidates = []
    mapped = split_prefixes.get(split)
    if mapped:
        candidates.append(mapped)
    candidates.extend([split, f"{split}set", f"{split}split"])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        required = [f"{candidate}/atomic_numbers", f"{candidate}/natoms", f"{candidate}/cell"]
        if all(name in variables for name in required):
            return candidate
    raise FileNotFoundError(
        f"Could not resolve split {split!r}; tried prefixes: {candidates}"
    )


def _get_split_size(reader, prefix: str, var_name: str = "natoms") -> int:
    """Get total number of structures in a split by reading variable_count metadata."""
    count_var = f"{prefix}/{var_name}/variable_count"
    if count_var in reader.available_variables():
        counts = np.asarray(reader.read(count_var, step_selection=[0, 1]))
        return len(counts)
    return 0


def _read_raw_omat_adios(path: Path, splits: list[str], rank: int = 0, size: int = 1) -> dict[str, np.ndarray]:
    adios2 = _import_adios2()
    adios_file = _find_adios_file(path)
    if adios_file is None:
        raise FileNotFoundError(f"No ADIOS .bp file found for raw dataset root: {path}")
    chunks: list[dict[str, np.ndarray]] = []
    with adios2.FileReader(str(adios_file)) as reader:
        variables = reader.available_variables()
        for split in splits:
            prefix = _choose_raw_split_prefix(split, variables)
            atomic_numbers = _read_raw_var(reader, f"{prefix}/atomic_numbers").reshape(-1)
            num_atoms = _read_raw_var(reader, f"{prefix}/natoms").reshape(-1)
            cell = _read_raw_var(reader, f"{prefix}/cell").reshape(-1, 3, 3)
            chunks.append(
                {
                    "atomic_numbers": atomic_numbers,
                    "num_atoms": num_atoms,
                    "cell": cell,
                }
            )
    return {
        "atomic_numbers": np.concatenate([chunk["atomic_numbers"] for chunk in chunks]),
        "num_atoms": np.concatenate([chunk["num_atoms"] for chunk in chunks]),
        "cell": np.concatenate([chunk["cell"] for chunk in chunks], axis=0),
    }


def _read_legacy(path: Path) -> dict[str, np.ndarray]:
    return {
        "atomic_numbers": np.load(path / "atomic_numbers.npy"),
        "num_atoms": np.load(path / "num_atoms.npy"),
        "cell": np.load(path / "cell.npy"),
    }


def read_stats_arrays(path: Path, rank: int = 0, size: int = 1) -> dict[str, np.ndarray]:
    if (path / PICKLE_FILENAME).is_file():
        return _read_pickle(path)
    if (path / ADIOS_FILENAME).exists():
        return _read_adios(path)
    if all((path / name).is_file() for name in LEGACY_FILES):
        return _read_legacy(path)
    raise FileNotFoundError(f"{path} is not a supported MatterGen cache.")


def load_splits(root: Path, splits: list[str], rank: int = 0, size: int = 1) -> dict[str, np.ndarray]:
    if _is_raw_omat_adios(root):
        return _read_raw_omat_adios(root, splits, rank, size)
    chunks = [_read for _read in (read_stats_arrays(path, rank, size) for path in _split_paths(root, splits))]
    return {
        "atomic_numbers": np.concatenate([chunk["atomic_numbers"].reshape(-1) for chunk in chunks]),
        "num_atoms": np.concatenate([chunk["num_atoms"].reshape(-1) for chunk in chunks]),
        "cell": np.concatenate([chunk["cell"] for chunk in chunks], axis=0),
    }


def compute_stats(arrays: dict[str, np.ndarray], rank: int = 0, size: int = 1) -> dict[str, Any]:
    atomic_numbers = arrays["atomic_numbers"].astype(np.int64, copy=False).reshape(-1)
    num_atoms = arrays["num_atoms"].astype(np.int64, copy=False).reshape(-1)
    cell = arrays["cell"]

    volumes = np.abs(np.linalg.det(cell))
    valid = volumes > 0
    if not np.all(valid):
        dropped = int((~valid).sum())
        if rank == 0:
            print(f"Warning: ignoring {dropped} structures with non-positive cell volume.")
    
    densities = num_atoms[valid] / volumes[valid]
    atom_values, atom_counts = np.unique(num_atoms, return_counts=True)
    distribution = {
        int(n): float(count / atom_counts.sum()) for n, count in zip(atom_values, atom_counts)
    }

    return {
        "num_structures": int(num_atoms.shape[0]),
        "num_atoms_total": int(num_atoms.sum()),
        "min_num_atoms": int(num_atoms.min()),
        "max_num_atoms": int(num_atoms.max()),
        "max_atomic_number": int(atomic_numbers.max()),
        "recommended_d3pm_dim": int(atomic_numbers.max()) + 1,
        "average_density": float(densities.mean()),
        "global_density": float(num_atoms[valid].sum() / volumes[valid].sum()),
        "min_density": float(densities.min()),
        "median_density": float(np.median(densities)),
        "max_density": float(densities.max()),
        "num_atoms_distribution": distribution,
        "_local": {
            "num_atoms_sum": int(num_atoms.sum()),
            "valid_volume_sum": float(volumes[valid].sum()),
            "all_densities": densities.tolist() if len(densities) < 10000 else None,
        }
    }


def _reduce_stats(local_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine stats from all ranks via MPI Reduce operations."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except ImportError:
        rank = 0
        comm = None

    if comm is None:
        return local_stats[0] if local_stats else {}

    # Reduce scalar metrics
    local = local_stats[0]
    num_structures = comm.allreduce(local["num_structures"], op=MPI.SUM) if comm else local["num_structures"]
    num_atoms_total = comm.allreduce(local["num_atoms_total"], op=MPI.SUM) if comm else local["num_atoms_total"]
    min_num_atoms = comm.allreduce(local["min_num_atoms"], op=MPI.MIN) if comm else local["min_num_atoms"]
    max_num_atoms = comm.allreduce(local["max_num_atoms"], op=MPI.MAX) if comm else local["max_num_atoms"]
    max_atomic_number = comm.allreduce(local["max_atomic_number"], op=MPI.MAX) if comm else local["max_atomic_number"]

    # Gather densities for global median computation
    if comm:
        all_densities = comm.allgather(np.array(local["_local"].get("all_densities") or []))
        all_densities_flat = np.concatenate([d for d in all_densities if len(d) > 0]) if any(len(d) > 0 for d in all_densities) else np.array([])
    else:
        all_densities_flat = np.array(local["_local"].get("all_densities") or [])

    # Recompute global density
    total_atoms_sum = comm.allreduce(local["_local"]["num_atoms_sum"], op=MPI.SUM) if comm else local["_local"]["num_atoms_sum"]
    total_volume_sum = comm.allreduce(local["_local"]["valid_volume_sum"], op=MPI.SUM) if comm else local["_local"]["valid_volume_sum"]
    global_density = float(total_atoms_sum / total_volume_sum) if total_volume_sum > 0 else 0.0

    # Merge num_atoms distributions
    if comm:
        all_dists = comm.allgather(local["num_atoms_distribution"])
        merged_dist = {}
        for dist in all_dists:
            for n, p in dist.items():
                merged_dist[n] = merged_dist.get(n, 0) + p
        # Normalize
        total_weight = sum(merged_dist.values())
        distribution = {n: p / total_weight for n, p in merged_dist.items()} if total_weight > 0 else {}
    else:
        distribution = local["num_atoms_distribution"]

    avg_density = np.median(all_densities_flat) if len(all_densities_flat) > 0 else 0.0

    return {
        "num_structures": num_structures,
        "num_atoms_total": num_atoms_total,
        "min_num_atoms": min_num_atoms,
        "max_num_atoms": max_num_atoms,
        "max_atomic_number": max_atomic_number,
        "recommended_d3pm_dim": max_atomic_number + 1,
        "average_density": float(avg_density),
        "global_density": global_density,
        "min_density": float(np.min(all_densities_flat)) if len(all_densities_flat) > 0 else 0.0,
        "median_density": float(np.median(all_densities_flat)) if len(all_densities_flat) > 0 else 0.0,
        "max_density": float(np.max(all_densities_flat)) if len(all_densities_flat) > 0 else 0.0,
        "num_atoms_distribution": distribution,
    }


def _print_summary(stats: dict[str, Any], distribution_name: str, rank: int = 0) -> None:
    if rank != 0:
        return
    print(json.dumps({k: v for k, v in stats.items() if k != "num_atoms_distribution"}, indent=2))
    print()
    print("Config-ready values:")
    print(f"average_density: {stats['average_density']}")
    print(
        "model_module.diffusion_module.corruption.discrete_corruptions."
        f"atomic_numbers.d3pm.dim={stats['recommended_d3pm_dim']}"
    )
    print()
    print("NUM_ATOMS_DISTRIBUTIONS snippet:")
    print(f'    "{distribution_name}": {{')
    for n, p in stats["num_atoms_distribution"].items():
        print(f"        {n}: {p},")
    print("    },")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MatterGen dataset statistics.")
    parser.add_argument("--root", required=True, help="Dataset root or a single split cache path.")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Splits to include.")
    parser.add_argument("--distribution-name", default="NEW_DATASET")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    # Get MPI rank and size
    size, rank = _get_mpi_info()
    if rank == 0:
        print(f"Running with {size} MPI rank(s)")

    arrays = load_splits(Path(args.root).resolve(), args.splits, rank, size)
    local_stats = compute_stats(arrays, rank, size)
    stats = _reduce_stats([local_stats])
    
    _print_summary(stats, args.distribution_name, rank)

    if args.out is not None and rank == 0:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump({k: v for k, v in stats.items() if k != "_local"}, f, indent=2)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
