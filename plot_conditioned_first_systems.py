#!/usr/bin/env python3
"""Plot the first conditioned chemical system from each category directory.

For every category under the conditioned results root, this script selects:
  1. the lexicographically first chemical_system_* directory
  2. the force_rms_* folder with the lowest numeric force value under it
  3. a random generated structure in generated_crystals.extxyz

Two overview PNGs are written by default:
  1. absolute/raw Cartesian positions with each generated cell box
  2. fractional positions in a unit-cube cell
"""

from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
import numpy as np
from ase.io import read
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visGen import plot_atoms_3d_scatter


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONDITIONED_ROOT = (
    REPO_ROOT / "results" / "miniOMat24-forces-3M" / "conditioned"
)
GENERATED_EXTXYZ = "generated_crystals.extxyz"
TITLE_FONTSIZE = 14
ATOM_LABEL_FONTSIZE = 9
CATEGORY_TITLES = {
    "battery_and_electrochemical_materials": "Batteries and Electrochemicals",
    "carbon_management_materials": "Carbon Management",
    "catalysts_and_electrocatalysts": "Catalysts and Electrocatalysts",
    "critical_mineral_compounds": "Critical Minerals",
    "fusion_materials": "Fusion Materials",
    "geological_and_subsurface_materials": "Geological and Subsurface",
    "hydrogen_materials": "Hydrogen Materials",
    "manufacturing_materials": "Manufacturing Materials",
    "materials_for_extreme_environments": "Extreme Environments",
    "nuclear_fission_materials": "Nuclear Fission",
    "solar_and_photochemical_materials": "Solar and Photochemical",
    "structural_alloys": "Structural Alloys",
}


def resolve_figure_font() -> str:
    for family in ("Times New Roman", "Nimbus Roman", "Times"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return family
    return "serif"


FIGURE_FONT = resolve_figure_font()

plt.rcParams.update(
    {
        "font.family": FIGURE_FONT,
        "font.size": 12,
        "axes.titlesize": TITLE_FONTSIZE,
    }
)


@dataclass(frozen=True)
class SelectedExample:
    category: str
    chemical_system: str
    force_rms: float
    force_dir: Path
    structure_index: int
    structure: object


def parse_force_rms(folder_name: str) -> float:
    match = re.search(r"force_rms_(?P<value>[0-9]+(?:p[0-9]+)?)$", folder_name)
    if match is None:
        raise ValueError(f"Could not parse force_rms from {folder_name!r}")
    return float(match.group("value").replace("p", "."))


def first_chemical_system(category_dir: Path) -> Path | None:
    chemical_systems = sorted(
        path
        for path in category_dir.iterdir()
        if path.is_dir() and path.name.startswith("chemical_system_")
    )
    return chemical_systems[0] if chemical_systems else None


def lowest_force_dir(chemical_system_dir: Path) -> tuple[float, Path] | None:
    force_dirs: list[tuple[float, Path]] = []
    for force_dir in chemical_system_dir.glob("**/force_rms_*"):
        if not force_dir.is_dir():
            continue
        if not (force_dir / GENERATED_EXTXYZ).is_file():
            continue
        force_dirs.append((parse_force_rms(force_dir.name), force_dir))

    if not force_dirs:
        return None
    return sorted(force_dirs, key=lambda item: (item[0], str(item[1])))[0]


def read_random_structure(extxyz_file: Path, rng: np.random.Generator) -> tuple[int, object]:
    structures = read(str(extxyz_file), ":")
    if not isinstance(structures, list):
        structures = [structures]
    if not structures:
        raise ValueError(f"No structures found in {extxyz_file}")
    structure_index = int(rng.integers(len(structures)))
    return structure_index, structures[structure_index]


def discover_examples(
    conditioned_root: Path,
    rng: np.random.Generator,
) -> list[SelectedExample]:
    examples: list[SelectedExample] = []
    category_dirs = sorted(path for path in conditioned_root.iterdir() if path.is_dir())

    for category_dir in category_dirs:
        chemical_system_dir = first_chemical_system(category_dir)
        if chemical_system_dir is None:
            print(f"Skipping {category_dir.name}: no chemical_system_* directories")
            continue

        force_choice = lowest_force_dir(chemical_system_dir)
        if force_choice is None:
            print(
                f"Skipping {category_dir.name}/{chemical_system_dir.name}: "
                f"no {GENERATED_EXTXYZ} under force_rms_* folders"
            )
            continue

        force_rms, force_dir = force_choice
        structure_index, structure = read_random_structure(
            force_dir / GENERATED_EXTXYZ,
            rng,
        )
        examples.append(
            SelectedExample(
                category=category_dir.name,
                chemical_system=chemical_system_dir.name.replace("chemical_system_", ""),
                force_rms=force_rms,
                force_dir=force_dir,
                structure_index=structure_index,
                structure=structure,
            )
        )

    if not examples:
        raise FileNotFoundError(f"No generated examples found under {conditioned_root}")
    return examples


def default_output_paths(conditioned_root: Path) -> tuple[Path, Path]:
    return (
        conditioned_root / "first_conditioned_systems_absolute.png",
        conditioned_root / "first_conditioned_systems_relative.png",
    )


def category_title(category: str) -> str:
    return CATEGORY_TITLES.get(category, category.replace("_", " ").title())


def plot_examples(
    examples: list[SelectedExample],
    output_png: Path,
    relative: bool = False,
    ncols: int = 3,
) -> None:
    ncols = max(1, ncols)
    nrows = max(1, math.ceil(len(examples) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={"projection": "3d"},
        figsize=(3.55 * ncols, 3.05 * nrows),
    )
    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.set_axis_off()

    for ax, example in zip(axes, examples):
        structure = example.structure
        cell = np.asarray(structure.cell.array, dtype=float)
        atomic_numbers = structure.get_atomic_numbers()

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
                f"{category_title(example.category)}\n"
                f"{example.chemical_system}"
            ),
        )
        ax.title.set_fontsize(TITLE_FONTSIZE)
        ax.title.set_fontfamily(FIGURE_FONT)
        for text in ax.texts:
            text.set_fontsize(ATOM_LABEL_FONTSIZE)
            text.set_fontfamily(FIGURE_FONT)
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.98,
        wspace=0.0,
        hspace=0.02,
    )
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioned-root", default=str(DEFAULT_CONDITIONED_ROOT))
    parser.add_argument("--absolute-png", default=None)
    parser.add_argument("--relative-png", default=None)
    parser.add_argument(
        "--only",
        choices=("both", "absolute", "relative"),
        default="both",
        help="Which coordinate views to plot.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of subplot columns in the overview figure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random example selection. Defaults to a fresh random draw.",
    )
    args = parser.parse_args()

    conditioned_root = Path(args.conditioned_root).expanduser().resolve()
    default_absolute_png, default_relative_png = default_output_paths(conditioned_root)
    absolute_png = Path(args.absolute_png).expanduser() if args.absolute_png else default_absolute_png
    relative_png = Path(args.relative_png).expanduser() if args.relative_png else default_relative_png

    print(f"Conditioned root: {conditioned_root}")
    rng = np.random.default_rng(args.seed)
    examples = discover_examples(conditioned_root, rng)
    print(f"Selected {len(examples)} generated examples:")
    for example in examples:
        print(
            f"  {category_title(example.category)}: "
            f"{example.chemical_system} example {example.structure_index}"
        )

    if args.only in ("both", "absolute"):
        absolute_png.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving absolute-position overview to {absolute_png}")
        plot_examples(examples, output_png=absolute_png, relative=False, ncols=args.ncols)

    if args.only in ("both", "relative"):
        relative_png.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving fractional-position overview to {relative_png}")
        plot_examples(examples, output_png=relative_png, relative=True, ncols=args.ncols)


if __name__ == "__main__":
    main()
