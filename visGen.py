#!/usr/bin/env python3
"""Plot generated MatterGen structures from an extxyz file.

Two PNGs are written:
  1. absolute/raw Cartesian positions with the generated cell box
  2. fractional positions in a unit-cube cell
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
import numpy as np
from ase.data import chemical_symbols
from ase.data import covalent_radii as radii
from ase.data.colors import jmol_colors
from ase.io import read
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EXTXYZ = (
    REPO_ROOT
    / "results"
    / "miniOMat24-forces-3M"
    / "generated_crystals.extxyz"
)


def default_png_paths(extxyz_file: str) -> tuple[str, str]:
    """Return default output PNG names derived from the extxyz file basename."""
    extxyz_path = Path(extxyz_file)
    base = extxyz_path.stem
    return (
        str(extxyz_path.with_name(f"{base}_absolute.png")),
        str(extxyz_path.with_name(f"{base}_relative.png")),
    )


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


def read_generated_structures(
    extxyz_file: str,
    n_structures: int | None = None,
    seed: int | None = 0,
) -> list[tuple[int, object]]:
    """Read all structures, or a random subset, from an extxyz file."""
    structures = read(extxyz_file, ":")
    if not isinstance(structures, list):
        structures = [structures]

    if n_structures is None:
        return list(enumerate(structures))

    if n_structures > len(structures):
        raise ValueError(
            f"Requested {n_structures} structures, but {extxyz_file} only has "
            f"{len(structures)}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(structures), size=n_structures, replace=False)
    return [(int(idx), structures[int(idx)]) for idx in indices]


def lattice_direction_magnitudes(cell: np.ndarray) -> np.ndarray:
    """Return the magnitudes of the three lattice direction vectors."""
    return np.linalg.norm(np.asarray(cell, dtype=float).reshape(3, 3), axis=1)


def plot_structure_grid(
    structures: list[tuple[int, object]],
    output_png: str,
    relative: bool = False,
    grid_size: int | None = None,
) -> None:
    if grid_size is None:
        grid_size = max(1, math.ceil(math.sqrt(len(structures))))

    fig, axes = plt.subplots(
        grid_size,
        grid_size,
        subplot_kw={"projection": "3d"},
        figsize=(3 * grid_size, 3 * grid_size),
        constrained_layout=True,
    )
    axes = np.asarray(axes).ravel()

    for ax in axes:
        ax.set_axis_off()

    for ax, (idx, structure) in zip(axes, structures):
        cell = np.asarray(structure.cell.array, dtype=float)
        atomic_numbers = structure.get_atomic_numbers()
        lattice_magnitudes = lattice_direction_magnitudes(cell)
        if relative:
            pos = structure.get_scaled_positions(wrap=True)
            plot_cell = np.eye(3)
        else:
            pos = structure.get_positions()
            plot_cell = cell

        plot_atoms_3d_scatter(
            pos,
            atomic_numbers,
            plot_cell,
            ax,
            title=(
                f"idx={idx} natoms={len(structure)}\n"
                f"{structure.get_chemical_formula()}\n"
                f"|a|={lattice_magnitudes[0]:.3g} "
                f"|b|={lattice_magnitudes[1]:.3g} "
                f"|c|={lattice_magnitudes[2]:.3g}"
            ),
        )

    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=str(DEFAULT_EXTXYZ))
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Randomly plot this many structures instead of all structures.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--absolute-png", default=None)
    parser.add_argument("--relative-png", default=None)
    args = parser.parse_args()

    default_absolute_png, default_relative_png = default_png_paths(args.file)
    args.absolute_png = args.absolute_png or default_absolute_png
    args.relative_png = args.relative_png or default_relative_png

    print(f"Reading extxyz file: {args.file}")
    structures = read_generated_structures(
        args.file,
        n_structures=args.count,
        seed=args.seed,
    )
    grid_size = max(1, math.ceil(math.sqrt(len(structures))))
    print(
        f"Plotting {len(structures)} generated structures in a "
        f"{grid_size}x{grid_size} grid"
    )

    plot_structure_grid(
        structures,
        output_png=args.absolute_png,
        relative=False,
        grid_size=grid_size,
    )
    plot_structure_grid(
        structures,
        output_png=args.relative_png,
        relative=True,
        grid_size=grid_size,
    )

    print(f"Saved absolute-position plot to {args.absolute_png}")
    print(f"Saved fractional-position plot to {args.relative_png}")


if __name__ == "__main__":
    main()
