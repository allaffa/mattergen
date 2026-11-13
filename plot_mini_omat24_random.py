#!/usr/bin/env python3
"""Plot random structures from the miniOmat24 ADIOS file.

Two PNGs are written:
  1. absolute/raw Cartesian-looking positions with the real cell box
  2. fractional positions in a unit-cube cell
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import adios2
import matplotlib
import numpy as np
from ase.build import bulk  # noqa: F401 - useful for ad-hoc ASE comparisons.
from ase.data import chemical_symbols
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from ase.io import read  # noqa: F401 - useful for ad-hoc structure inspection.
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

try:
    import imageio.v2 as imageio  # noqa: F401 - kept for quick GIF experiments.
except ImportError:
    imageio = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
MINI_OMAT_CONFIG = REPO_ROOT / "mattergen" / "conf" / "data_module" / "miniOmat24.yaml"


def read_mini_omat24_path(config_path: Path = MINI_OMAT_CONFIG) -> str:
    """Return the ADIOS path used by mattergen/conf/data_module/miniOmat24.yaml."""
    text = config_path.read_text()
    match = re.search(r"^root_dir:\s*(?P<path>[^#\n]+)", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not find root_dir in {config_path}")

    path = match.group("path").strip().strip("'\"")
    path = path.replace("${oc.env:PROJECT_ROOT}", str(REPO_ROOT))
    return os.path.expandvars(os.path.expanduser(path))


def default_png_paths(adios_file: str) -> tuple[str, str]:
    """Return default output PNG names derived from the ADIOS file basename."""
    base = Path(adios_file).stem
    return f"{base}_absolute.png", f"{base}_relative.png"


def _draw_cell_box(
    ax,
    cell,
    origin=(0.0, 0.0, 0.0),
    edgecolor="k",
    linewidth=1.2,
    alpha=0.12,
):
    """Draw a transparent parallelepiped defined by an ASE-style 3x3 cell."""
    cell = np.asarray(cell, dtype=float)
    o = np.asarray(origin, dtype=float)
    a, b, c = cell[0], cell[1], cell[2]

    v000 = o
    v100 = o + a
    v010 = o + b
    v001 = o + c
    v110 = o + a + b
    v101 = o + a + c
    v011 = o + b + c
    v111 = o + a + b + c

    faces = [
        [v000, v100, v110, v010],
        [v001, v101, v111, v011],
        [v000, v100, v101, v001],
        [v010, v110, v111, v011],
        [v000, v010, v011, v001],
        [v100, v110, v111, v101],
    ]

    poly = Poly3DCollection(
        faces,
        facecolors=(1, 1, 1, alpha),
        edgecolors=edgecolor,
        linewidths=linewidth,
    )
    ax.add_collection3d(poly)


def _set_equal_3d_limits(ax, points):
    points = np.asarray(points, dtype=float)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * (maxs - mins).max()
    if radius == 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_atoms_3d_scatter(pos, atoms, cell, ax, title=None):
    """Plot atoms and draw a transparent cell box from a 3x3 cell matrix."""
    pos = np.asarray(pos, dtype=float)
    atoms = np.asarray(atoms, dtype=np.int64).reshape(-1)
    cell = np.asarray(cell, dtype=float).reshape(3, 3)

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    colors = [jmol_colors[n] for n in atoms]
    sizes = [max(radii[n] * 100, 20) for n in atoms]

    ax.scatter(x, y, z, c=colors, s=sizes, marker="o", depthshade=True)

    for i, atomic_number in enumerate(atoms):
        ax.text(x[i], y[i], z[i], chemical_symbols[int(atomic_number)], fontsize=7)

    _draw_cell_box(ax, cell, origin=(0, 0, 0), alpha=0.12, edgecolor="k")

    corners = np.array(
        [
            [0, 0, 0],
            cell[0],
            cell[1],
            cell[2],
            cell[0] + cell[1],
            cell[0] + cell[2],
            cell[1] + cell[2],
            cell[0] + cell[1] + cell[2],
        ]
    )
    _set_equal_3d_limits(ax, np.vstack([pos, corners]))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if title:
        ax.set_title(title, fontsize=8)


def composition_formula(atomic_numbers: np.ndarray) -> str:
    """Return a compact composition formula from one structure's atomic numbers."""
    counts = Counter(int(n) for n in np.asarray(atomic_numbers).reshape(-1))
    parts = []
    for atomic_number in sorted(counts):
        symbol = chemical_symbols[atomic_number]
        count = counts[atomic_number]
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def _read_ragged_sample(reader, variable: str, idx: int, width: int) -> np.ndarray:
    count = reader.read(
        f"{variable}/variable_count",
        start=[idx],
        count=[1],
        step_selection=[0, 1],
    )[0]
    offset = reader.read(
        f"{variable}/variable_offset",
        start=[idx],
        count=[1],
        step_selection=[0, 1],
    )[0]
    return np.asarray(
        reader.read(
            variable,
            start=[int(offset), 0],
            count=[int(count), width],
            step_selection=[0, 1],
        )
    )


def _read_scalar_sample(reader, variable: str, idx: int) -> int:
    value = reader.read(variable, start=[idx], count=[1], step_selection=[0, 1])
    return int(np.asarray(value).reshape(-1)[0])


def get_split_size(adios_file: str, split: str = "trainset") -> int:
    """Return the number of structures in a split."""
    with adios2.FileReader(adios_file) as reader:
        shape = reader.available_variables()[f"{split}/natoms"]["Shape"]
    return parse_adios_shape_size(shape)


def parse_adios_shape_size(shape) -> int:
    if isinstance(shape, str):
        return int(shape.split(",")[0])
    return int(shape[0] if hasattr(shape, "__len__") else shape)


def get_file_split_sizes(adios_file: str) -> dict[str, int]:
    """Return structure counts for every known split present in an ADIOS file."""
    split_sizes = {}
    with adios2.FileReader(adios_file) as reader:
        variables = reader.available_variables()
        for split in ("trainset", "valset", "testset"):
            key = f"{split}/natoms"
            if key in variables:
                split_sizes[split] = parse_adios_shape_size(variables[key]["Shape"])
    return split_sizes


def read_adios_structure(adios_file: str, idx: int, split: str = "trainset") -> dict:
    """Read one structure directly from an ADIOS file."""
    with adios2.FileReader(adios_file) as reader:
        natoms = _read_scalar_sample(reader, f"{split}/natoms", idx)
        atomic_numbers = _read_ragged_sample(
            reader, f"{split}/atomic_numbers", idx, width=1
        ).reshape(-1)
        pos = _read_ragged_sample(reader, f"{split}/pos", idx, width=3)
        cell = _read_ragged_sample(reader, f"{split}/cell", idx, width=3).reshape(3, 3)
        forces = _read_ragged_sample(reader, f"{split}/forces", idx, width=3)

    if len(atomic_numbers) != natoms or len(pos) != natoms or len(forces) != natoms:
        raise ValueError(
            f"Sample {idx} says natoms={natoms}, but read "
            f"{len(atomic_numbers)} atom types, {len(pos)} positions, "
            f"and {len(forces)} force rows"
        )

    force_rms = np.sqrt(np.sum(np.square(forces), axis=-1).mean())

    return {
        "idx": int(idx),
        "natoms": int(natoms),
        "atomic_numbers": atomic_numbers.astype(np.int64),
        "pos": np.asarray(pos, dtype=float),
        "cell": np.asarray(cell, dtype=float),
        "forces": np.asarray(forces, dtype=float),
        "force_rms": max(float(force_rms), 1e-8),
    }


def read_random_adios_structures(
    adios_file: str,
    n_structures: int = 25,
    split: str = "trainset",
    seed: int | None = 0,
) -> list[dict]:
    """Read random structures directly from an ADIOS file."""
    n_samples = get_split_size(adios_file, split=split)
    if n_structures > n_samples:
        raise ValueError(
            f"Requested {n_structures} structures, but {split} only has {n_samples}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_samples, size=n_structures, replace=False)
    return [
        read_adios_structure(adios_file, int(idx), split=split)
        for idx in tqdm(indices, total=len(indices), desc=f"Reading {split}")
    ]


def positions_are_fractional(pos: np.ndarray, tolerance: float = 1e-6) -> bool:
    return bool(np.nanmin(pos) >= -tolerance and np.nanmax(pos) <= 1.0 + tolerance)


def fractional_to_cartesian(frac_pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return np.asarray(frac_pos, dtype=float) @ np.asarray(cell, dtype=float)


def cartesian_to_fractional(cart_pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return np.linalg.solve(
        np.asarray(cell, dtype=float).T,
        np.asarray(cart_pos, dtype=float).T,
    ).T


def lattice_direction_magnitudes(cell: np.ndarray) -> np.ndarray:
    """Return the magnitudes of the three lattice direction vectors."""
    return np.linalg.norm(np.asarray(cell, dtype=float).reshape(3, 3), axis=1)


def structure_positions(structure: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute positions and fractional positions for plotting."""
    pos = structure["pos"]
    cell = structure["cell"]
    if positions_are_fractional(pos):
        fractional = pos % 1.0
        absolute = fractional_to_cartesian(fractional, cell)
    else:
        absolute = pos
        fractional = cartesian_to_fractional(absolute, cell) % 1.0
    return absolute, fractional


def plot_structure_grid(
    structures: list[dict],
    output_png: str,
    relative: bool = False,
    rows: int = 5,
    cols: int = 5,
) -> None:
    fig, axes = plt.subplots(
        rows,
        cols,
        subplot_kw={"projection": "3d"},
        figsize=(12, 12),
        constrained_layout=True,
    )
    axes = np.asarray(axes).ravel()

    for ax in axes:
        ax.set_axis_off()

    for ax, structure in zip(axes, structures):
        absolute_pos, fractional_pos = structure_positions(structure)
        lattice_magnitudes = lattice_direction_magnitudes(structure["cell"])
        composition = composition_formula(structure["atomic_numbers"])
        if relative:
            pos = fractional_pos
            cell = np.eye(3)
        else:
            pos = absolute_pos
            cell = structure["cell"]

        plot_atoms_3d_scatter(
            pos,
            structure["atomic_numbers"],
            cell,
            ax,
            title=(
                f"idx={structure['idx']} natoms={structure['natoms']} {composition}\n"
                f"rms_force={structure['force_rms']:.4g}\n"
                f"|a|={lattice_magnitudes[0]:.3g} "
                f"|b|={lattice_magnitudes[1]:.3g} "
                f"|c|={lattice_magnitudes[2]:.3g}"
            ),
        )

    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=read_mini_omat24_path())
    parser.add_argument("--split", default="trainset")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--absolute-png", default=None)
    parser.add_argument("--relative-png", default=None)
    args = parser.parse_args()

    default_absolute_png, default_relative_png = default_png_paths(args.file)
    args.absolute_png = args.absolute_png or default_absolute_png
    args.relative_png = args.relative_png or default_relative_png

    expected_slots = args.rows * args.cols
    if args.count > expected_slots:
        raise ValueError(
            f"--count={args.count} will not fit in a {args.rows}x{args.cols} grid"
        )

    print(f"Reading ADIOS file: {args.file}")
    split_sizes = get_file_split_sizes(args.file)
    if split_sizes:
        total_structures = sum(split_sizes.values())
        split_summary = ", ".join(
            f"{split}={count}" for split, count in split_sizes.items()
        )
        print(f"Total structures in file: {total_structures} ({split_summary})")
    print(f"Split: {args.split}")
    structures = read_random_adios_structures(
        args.file,
        n_structures=args.count,
        split=args.split,
        seed=args.seed,
    )

    plot_structure_grid(
        structures,
        output_png=args.absolute_png,
        relative=False,
        rows=args.rows,
        cols=args.cols,
    )
    plot_structure_grid(
        structures,
        output_png=args.relative_png,
        relative=True,
        rows=args.rows,
        cols=args.cols,
    )

    print(f"Saved absolute-position plot to {args.absolute_png}")
    print(f"Saved fractional-position plot to {args.relative_png}")


if __name__ == "__main__":
    main()
