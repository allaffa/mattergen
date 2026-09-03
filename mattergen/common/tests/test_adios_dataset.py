from __future__ import annotations

import numpy as np
import pytest
import torch


adios2 = pytest.importorskip("adios2")

from mattergen.common.data.adiosDataset import HydraGNNAdiosCrystalDataset


def _write_array(stream, name: str, values: np.ndarray) -> None:
    shape = list(values.shape)
    stream.write(name, values, shape, [0] * values.ndim, shape)


def _write_ragged(
    stream,
    split: str,
    name: str,
    values: np.ndarray,
    counts: np.ndarray,
) -> None:
    offsets = np.zeros_like(counts)
    offsets[1:] = np.cumsum(counts[:-1])
    _write_array(stream, f"{split}/{name}", values)
    _write_array(stream, f"{split}/{name}/variable_count", counts)
    _write_array(stream, f"{split}/{name}/variable_offset", offsets)


def test_hydragnn_format_reader_uses_streaming_metadata(tmp_path):
    path = tmp_path / "tiny.bp"
    split = "trainset"
    natoms = np.asarray([2, 1], dtype=np.int32)
    atom_counts = np.asarray([2, 1], dtype=np.int64)
    cell_counts = np.asarray([3, 3], dtype=np.int64)

    atomic_numbers = np.asarray([[6], [8], [14]], dtype=np.float32)
    pos = np.asarray(
        [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
        dtype=np.float32,
    )
    forces = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float32,
    )
    cells = np.concatenate(
        [np.eye(3, dtype=np.float32), 2 * np.eye(3, dtype=np.float32)]
    )

    with adios2.Stream(str(path), "w") as stream:
        _write_array(stream, f"{split}/natoms", natoms)
        _write_ragged(stream, split, "atomic_numbers", atomic_numbers, atom_counts)
        _write_ragged(stream, split, "pos", pos, atom_counts)
        _write_ragged(stream, split, "forces", forces, atom_counts)
        _write_ragged(stream, split, "cell", cells, cell_counts)

    dataset = HydraGNNAdiosCrystalDataset(
        filename=path,
        label=split,
        keys=["pos", "cell", "atomic_numbers", "forces"],
        properties=["force_rms"],
    )
    try:
        assert len(dataset) == 2
        assert dataset.total_node_count == 3
        assert dataset.read_node_counts_range(0, 2).tolist() == [2, 1]

        first = dataset[0]
        assert first.atomic_numbers.dtype == torch.long
        assert first.atomic_numbers.tolist() == [6, 8]
        assert first.pos.shape == (2, 3)
        assert first.cell.shape == (1, 3, 3)
        assert first.num_atoms == 2
        assert first.force_rms.item() == pytest.approx(np.sqrt(2.5))
    finally:
        dataset.close()

    limited = HydraGNNAdiosCrystalDataset(
        filename=path,
        label=split,
        max_samples=1,
        keys=["pos", "cell", "atomic_numbers", "forces"],
        properties=["force_rms"],
    )
    try:
        assert len(limited) == 1
        assert limited.total_node_count == 2
        assert limited.read_node_counts_range(0, 1).tolist() == [2]
        assert limited[-1].atomic_numbers.tolist() == [6, 8]
        with pytest.raises(IndexError):
            _ = limited[1]
        with pytest.raises(IndexError):
            limited.read_node_counts_range(0, 2)
    finally:
        limited.close()

    with pytest.raises(ValueError, match="max_samples must be positive"):
        HydraGNNAdiosCrystalDataset(filename=path, label=split, max_samples=0)
    with pytest.raises(ValueError, match="contains only 2 samples"):
        HydraGNNAdiosCrystalDataset(filename=path, label=split, max_samples=3)
