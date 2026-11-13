import os
from typing import Dict, List, Tuple, Iterable

import adios2
import numpy as np
from tqdm import tqdm

SPLITS = ["train", "val"]

def collate_stack_with_offsets_counts(
    arrs: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(arrs) == 0:
        return (
            np.empty((0,), dtype=float),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    counts = np.array([a.shape[0] for a in arrs], dtype=np.int64)
    offsets = np.empty(len(arrs), dtype=np.int64)
    offsets[0] = 0
    if len(arrs) > 1:
        offsets[1:] = np.cumsum(counts[:-1])

    tail_shape = arrs[0].shape[1:]
    for i, a in enumerate(arrs):
        if a.shape[1:] != tail_shape:
            raise ValueError(f"arrs[{i}].shape[1:]={a.shape[1:]} != {tail_shape}")

    stacked = np.concatenate(arrs, axis=0)
    return stacked, offsets, counts

def write_split_adios(
    s: adios2.Stream,
    split: str,
    val: Dict[str, List[np.ndarray]],
) -> None:
    s.write_attribute(f"{split}set/keys", list(val.keys()))
    s.write_attribute(f"{split}set/ndata", len(val["natoms"]))

    for key, v in val.items():
        x_data, x_offsets, x_counts = collate_stack_with_offsets_counts(v)
        s.write_attribute(f"{split}set/{key}/variable_dim", 0)

        for x, lab in zip(
            [x_data, x_offsets, x_counts],
            ["", "variable_offset", "variable_count"],
        ):
            name = f"{split}set/{key}" if lab == "" else f"{split}set/{key}/{lab}"

            if x.ndim == 1:
                start = [0]
                count = [x.shape[0]]
            else:
                start = [0, 0]
                count = [x.shape[0], x.shape[1]]

            s.write(name, x, x.shape, start, count)

def init_buffer() -> Dict[str, List[np.ndarray]]:
    return {
        "pos": [],
        "cell": [],
        "atomic_numbers": [],
        "forces": [],
        "natoms": [],
        "chemical_composition": [],
    }

def append_structure(
    buf: Dict[str, List[np.ndarray]],
    pos: np.ndarray,
    cell: np.ndarray,
    atomic_numbers: np.ndarray,
    forces: np.ndarray,
    natoms: np.ndarray,
    chemical_composition: np.ndarray,
) -> None:
    buf["pos"].append(pos)
    buf["cell"].append(cell)
    buf["atomic_numbers"].append(atomic_numbers)
    buf["forces"].append(forces)
    buf["natoms"].append(natoms)
    buf["chemical_composition"].append(chemical_composition)

def get_num_structures(s: adios2.FileReader, split: str) -> int:
    vars_avail = s.available_variables()
    shape = vars_avail[f"{split}set/natoms"]["Shape"]

    if isinstance(shape, str):
        return int(shape.split(",")[0].strip())
    elif isinstance(shape, (list, tuple)):
        return int(shape[0])
    else:
        return int(shape)

def preload_split_metadata(s: adios2.FileReader, split: str) -> Dict[str, np.ndarray]:
    return {
        "natoms": s.read(f"{split}set/natoms"),
        "atomic_numbers_offset": s.read(f"{split}set/atomic_numbers/variable_offset"),
        "atomic_numbers_count": s.read(f"{split}set/atomic_numbers/variable_count"),
        "pos_offset": s.read(f"{split}set/pos/variable_offset"),
        "pos_count": s.read(f"{split}set/pos/variable_count"),
        "forces_offset": s.read(f"{split}set/forces/variable_offset"),
        "forces_count": s.read(f"{split}set/forces/variable_count"),
        "chemical_composition_offset": s.read(
            f"{split}set/chemical_composition/variable_offset"
        ),
        "chemical_composition_count": s.read(
            f"{split}set/chemical_composition/variable_count"
        ),
    }

def read_variable_span(
    s: adios2.FileReader,
    varname: str,
    row0: int,
    nrows: int,
    width: int,
) -> np.ndarray:
    return s.read(
        varname,
        start=[int(row0), 0],
        count=[int(nrows), int(width)],
        step_selection=[0, 1],
    )

def read_forces_block(
    s: adios2.FileReader,
    split: str,
    i0: int,
    i1: int,
    meta: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, int]:
    if i1 <= i0:
        return np.empty((0, 3), dtype=np.float32), 0

    offsets = meta["forces_offset"]
    counts = meta["forces_count"]

    global_row0 = int(offsets[i0])
    global_row1 = int(offsets[i1 - 1] + counts[i1 - 1])

    forces_block = s.read(
        f"{split}set/forces",
        start=[global_row0, 0],
        count=[global_row1 - global_row0, 3],
        step_selection=[0, 1],
    )

    return forces_block, global_row0

def get_forces_from_block(
    forces_block: np.ndarray,
    global_row0: int,
    i: int,
    meta: Dict[str, np.ndarray],
) -> np.ndarray:
    off = int(meta["forces_offset"][i])
    cnt = int(meta["forces_count"][i])

    local0 = off - global_row0
    local1 = local0 + cnt

    return forces_block[local0:local1]

def read_cell_block(
    s: adios2.FileReader,
    split: str,
    i0: int,
    i1: int,
) -> np.ndarray:
    if i1 <= i0:
        return np.empty((0, 3, 3), dtype=np.float32)

    cell_block = s.read(
        f"{split}set/cell",
        start=[3 * i0, 0],
        count=[3 * (i1 - i0), 3],
        step_selection=[0, 1],
    )
    return cell_block.reshape(i1 - i0, 3, 3)

def compute_cell_volume_ratios(cells: np.ndarray) -> np.ndarray:
    volumes = np.abs(np.linalg.det(cells))
    edge_length_products = np.prod(np.linalg.norm(cells, axis=2), axis=1)
    ratios = np.zeros(len(cells), dtype=np.float64)
    valid = np.isfinite(volumes) & np.isfinite(edge_length_products) & (edge_length_products > 0)
    ratios[valid] = volumes[valid] / edge_length_products[valid]
    return ratios

def consecutive_runs(indices: np.ndarray) -> Iterable[Tuple[int, int]]:
    """
    Yield half-open runs [a, b) from a sorted 1D integer array of indices.
    Example: [3,4,5,9,10] -> (3,6), (9,11)
    """
    if len(indices) == 0:
        return

    start = int(indices[0])
    prev = start

    for x in indices[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
        else:
            yield start, prev + 1
            start = x
            prev = x

    yield start, prev + 1

def batch_append_kept_structures(
    s: adios2.FileReader,
    split: str,
    kept_indices: np.ndarray,
    meta: Dict[str, np.ndarray],
    forces_block: np.ndarray,
    global_force_row0: int,
    out: Dict[str, List[np.ndarray]],
) -> int:
    """
    Batch-read non-force fields for accepted structures.
    Groups consecutive kept structure indices into runs and reads one slab per field per run.
    Returns number of appended structures.
    """
    if len(kept_indices) == 0:
        return 0

    appended = 0

    for run_start, run_end in consecutive_runs(kept_indices):
        # Read cell slab for structures [run_start, run_end)
        cell_block = s.read(
            f"{split}set/cell",
            start=[3 * run_start, 0],
            count=[3 * (run_end - run_start), 3],
            step_selection=[0, 1],
        )

        # Read atomic_numbers slab for this run
        an_row0 = int(meta["atomic_numbers_offset"][run_start])
        an_row1 = int(
            meta["atomic_numbers_offset"][run_end - 1]
            + meta["atomic_numbers_count"][run_end - 1]
        )
        atomic_numbers_block = read_variable_span(
            s,
            f"{split}set/atomic_numbers",
            an_row0,
            an_row1 - an_row0,
            1,
        )

        # Read pos slab for this run
        pos_row0 = int(meta["pos_offset"][run_start])
        pos_row1 = int(meta["pos_offset"][run_end - 1] + meta["pos_count"][run_end - 1])
        pos_block = read_variable_span(
            s,
            f"{split}set/pos",
            pos_row0,
            pos_row1 - pos_row0,
            3,
        )

        # Read chemical_composition slab for this run
        cc_row0 = int(meta["chemical_composition_offset"][run_start])
        cc_row1 = int(
            meta["chemical_composition_offset"][run_end - 1]
            + meta["chemical_composition_count"][run_end - 1]
        )
        chemical_composition_block = read_variable_span(
            s,
            f"{split}set/chemical_composition",
            cc_row0,
            cc_row1 - cc_row0,
            1,
        )

        for i in range(run_start, run_end):
            natoms = np.array([meta["natoms"][i]], dtype=meta["natoms"].dtype)

            # cell local slice
            cell_local0 = 3 * (i - run_start)
            cell = cell_block[cell_local0 : cell_local0 + 3]

            # atomic_numbers local slice
            an_local0 = int(meta["atomic_numbers_offset"][i] - an_row0)
            an_local1 = an_local0 + int(meta["atomic_numbers_count"][i])
            atomic_numbers = atomic_numbers_block[an_local0:an_local1]

            # pos local slice
            pos_local0 = int(meta["pos_offset"][i] - pos_row0)
            pos_local1 = pos_local0 + int(meta["pos_count"][i])
            pos = pos_block[pos_local0:pos_local1]

            # chemical_composition local slice
            cc_local0 = int(meta["chemical_composition_offset"][i] - cc_row0)
            cc_local1 = cc_local0 + int(meta["chemical_composition_count"][i])
            chemical_composition = chemical_composition_block[cc_local0:cc_local1]

            forces = get_forces_from_block(forces_block, global_force_row0, i, meta)

            append_structure(
                out,
                pos=pos,
                cell=cell,
                atomic_numbers=atomic_numbers,
                forces=forces.copy(),
                natoms=natoms,
                chemical_composition=chemical_composition,
            )
            appended += 1

    return appended

def filter_adios_by_cell(
    source: str,
    dest: str,
    min_volume_ratio: float = 0.6,
    block_structures: int = 2048,
    show_progress: bool = True,
) -> None:
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    total_seen = {split: 0 for split in SPLITS}
    total_kept = {split: 0 for split in SPLITS}

    with adios2.FileReader(source) as reader, adios2.Stream(dest, "w") as writer:
        for split in SPLITS:
            n_structures = get_num_structures(reader, split)
            print(
                f"\nProcessing split={split}, N={n_structures}, "
                f"min_volume_ratio={min_volume_ratio}, "
                f"block_structures={block_structures}"
            )
            print(f"Preloading metadata for {split}...")
            meta = preload_split_metadata(reader, split)

            out = init_buffer()

            pbar = (
                tqdm(total=n_structures, desc=f"filtering {split}")
                if show_progress
                else None
            )

            for i0 in range(0, n_structures, block_structures):
                i1 = min(i0 + block_structures, n_structures)

                forces_block, global_force_row0 = read_forces_block(
                    reader, split, i0, i1, meta
                )

                cell_block = read_cell_block(reader, split, i0, i1)
                volume_ratios = compute_cell_volume_ratios(cell_block)

                volume_keep = volume_ratios > min_volume_ratio
                keep_local = np.nonzero(volume_keep)[0]
                keep_global = i0 + keep_local

                total_seen[split] += (i1 - i0)

                total_kept[split] += batch_append_kept_structures(
                    reader,
                    split,
                    keep_global,
                    meta,
                    forces_block,
                    global_force_row0,
                    out,
                )

                if pbar is not None:
                    pbar.update(i1 - i0)

            if pbar is not None:
                pbar.close()

            pct = 100.0 * total_kept[split] / max(total_seen[split], 1)
            print(
                f"{split}: final kept {total_kept[split]} / {total_seen[split]} "
                f"({pct:.3f}%)"
            )

            print(f"Writing split={split} ...")
            write_split_adios(writer, split, out)

            # free split buffer before next split
            del out

    print("\nDone.")
    for split in SPLITS:
        print(f"{split}: final kept {total_kept[split]} / {total_seen[split]}")

if __name__ == "__main__":
    source = "/DATA-A/DATA-B_copy/Patxi/lrn070/OMat24-force-filtered.bp"
    dest = "/DATA-A/DATA-B_copy/Patxi/lrn070/OMat24-force-filtered-cell-filtered.bp"

    filter_adios_by_cell(
        source=source,
        dest=dest,
        min_volume_ratio=0.6,
        block_structures=2048,
        show_progress=True,
    )
