#!/usr/bin/env python3
"""Plot conditioned generation sweeps as force rows by example columns.

Two PNGs are written by default:
  1. absolute/raw Cartesian positions with the generated cell box
  2. fractional positions in a unit-cube cell
"""

from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
import numpy as np
from ase.io import read

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visGen import lattice_direction_magnitudes, plot_atoms_3d_scatter


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONDITION_ROOT = (
    REPO_ROOT
    / "results"
    / "miniOMat24-forces-3M"
    / "conditioned"
    / "chemical_system_Ti-Ni-Pd"
)
GENERATED_EXTXYZ = "generated_crystals.extxyz"


def parse_force_rms(folder_name: str) -> float:
    match = re.search(r"force_rms_(?P<value>[0-9]+(?:p[0-9]+)?)$", folder_name)
    if match is None:
        raise ValueError(f"Could not parse force_rms from {folder_name!r}")
    return float(match.group("value").replace("p", "."))


def select_run_root(condition_root: Path, run: str | None = None) -> Path:
    """Return the run directory that contains force_rms_* folders."""
    if run is not None:
        run_root = condition_root / run
        if not run_root.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_root}")
        return run_root

    direct_force_dirs = sorted(condition_root.glob("force_rms_*"))
    if direct_force_dirs:
        return condition_root

    candidates = [
        path
        for path in condition_root.iterdir()
        if path.is_dir() and any(path.glob("force_rms_*"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No force_rms_* folders found under {condition_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def discover_force_folders(run_root: Path) -> list[tuple[float, Path]]:
    folders = []
    for folder in run_root.glob("force_rms_*"):
        if not folder.is_dir():
            continue
        extxyz = folder / GENERATED_EXTXYZ
        if extxyz.is_file():
            folders.append((parse_force_rms(folder.name), folder))
    if not folders:
        raise FileNotFoundError(f"No {GENERATED_EXTXYZ} files found under {run_root}")
    return sorted(folders, key=lambda item: abs(item[0]), reverse=True)


def read_examples(extxyz_file: Path, count: int) -> list[object]:
    structures = read(str(extxyz_file), ":")
    if not isinstance(structures, list):
        structures = [structures]
    return structures[:count]


def plot_condition_grid(
    force_folders: list[tuple[float, Path]],
    output_png: Path,
    relative: bool = False,
) -> None:
    num_rows = len(force_folders)
    num_cols = 2 * num_rows
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        subplot_kw={"projection": "3d"},
        figsize=(3.1 * num_cols, 3.1 * num_rows),
    )
    axes = np.asarray(axes).reshape(num_rows, num_cols)

    for ax in axes.ravel():
        ax.set_axis_off()

    for col in range(num_cols):
        axes[0, col].set_title(f"example {col + 1}", fontsize=10, pad=10)

    for row, (force_rms, folder) in enumerate(force_folders):
        examples = read_examples(folder / GENERATED_EXTXYZ, num_cols)
        for col, structure in enumerate(examples):
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
                axes[row, col],
                title=(
                    f"{structure.get_chemical_formula()}\n"
                    f"natoms={len(structure)} "
                    f"|a|={lattice_magnitudes[0]:.2g} "
                    f"|b|={lattice_magnitudes[1]:.2g} "
                    f"|c|={lattice_magnitudes[2]:.2g}"
                ),
            )

        for col in range(len(examples), num_cols):
            axes[row, col].text2D(
                0.5,
                0.5,
                "missing",
                transform=axes[row, col].transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )

        row_center = 1.0 - (row + 0.5) / num_rows
        fig.text(
            0.018,
            row_center,
            f"force_rms={force_rms:1.1e}",
            rotation="vertical",
            va="center",
            ha="center",
            fontsize=11,
        )

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.02, top=0.94, wspace=0.02, hspace=0.08)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def default_output_path(run_root: Path, relative: bool) -> Path:
    suffix = "relative" if relative else "absolute"
    return run_root / f"conditioned_force_grid_{suffix}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", default=str(DEFAULT_CONDITION_ROOT))
    parser.add_argument(
        "--run",
        default=None,
        help="Specific timestamp/run directory under --condition-root. Defaults to latest.",
    )
    parser.add_argument("--absolute-png", default=None)
    parser.add_argument("--relative-png", default=None)
    parser.add_argument(
        "--only",
        choices=("both", "absolute", "relative"),
        default="both",
        help="Which coordinate views to plot.",
    )
    args = parser.parse_args()

    condition_root = Path(args.condition_root).expanduser().resolve()
    run_root = select_run_root(condition_root, run=args.run)
    force_folders = discover_force_folders(run_root)
    absolute_png = Path(args.absolute_png) if args.absolute_png else default_output_path(
        run_root, relative=False
    )
    relative_png = Path(args.relative_png) if args.relative_png else default_output_path(
        run_root, relative=True
    )

    print(f"Condition root: {condition_root}")
    print(f"Run root: {run_root}")
    print(f"Found {len(force_folders)} force-conditioning folders:")
    for force_rms, folder in force_folders:
        print(f"  force_rms={force_rms:1.1e}: {folder}")

    if args.only in ("both", "absolute"):
        absolute_png.parent.mkdir(parents=True, exist_ok=True)
        print(f"Plotting {len(force_folders)}x{2 * len(force_folders)} absolute grid")
        plot_condition_grid(force_folders, output_png=absolute_png, relative=False)
        print(f"Saved absolute-position plot to {absolute_png}")

    if args.only in ("both", "relative"):
        relative_png.parent.mkdir(parents=True, exist_ok=True)
        print(f"Plotting {len(force_folders)}x{2 * len(force_folders)} relative grid")
        plot_condition_grid(force_folders, output_png=relative_png, relative=True)
        print(f"Saved fractional-position plot to {relative_png}")


if __name__ == "__main__":
    main()
