from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import mattergen.common.data.node_budget_sampler as sampler_module
from mattergen.common.data.node_budget_sampler import (
    DistributedNodeBudgetBatchSampler,
    _fresh_assignments,
)


class _FakeReader:
    def __init__(self, counts, reads=None):
        self.counts = np.asarray(counts, dtype=np.int64)
        self.reads = reads if reads is not None else []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def available_variables(self):
        return {"node_count": {"Shape": str(len(self.counts))}}

    def read(self, name, *, start, count, step_selection):
        assert name == "node_count"
        assert step_selection == [0, 1]
        self.reads.append((start[0], count[0]))
        return self.counts[start[0]: start[0] + count[0]].copy()


def _install_fake_reader(monkeypatch, counts, reads=None):
    monkeypatch.setattr(
        sampler_module,
        "_open_adios_reader",
        lambda _path: _FakeReader(counts, reads),
    )


def _consume_all(sampler):
    batches = []
    for batch in sampler:
        batches.append(batch)
        sampler.mark_batch_consumed()
    return batches


def test_stated_partial_batch_example(monkeypatch):
    counts = [100, 200, 300, 1000, 1000]
    _install_fake_reader(monkeypatch, counts)
    sampler = DistributedNodeBudgetBatchSampler(
        "unused.bp",
        "node_count",
        target_node_count=800,
        node_budget=1000,
        chunk_size=5,
    )

    batches = _consume_all(sampler)

    assert batches == [[0, 1, 2], [3], [4]]
    assert [sum(counts[index] for index in batch) for batch in batches] == [600, 1000, 1000]
    assert sampler.rollover_count == 0


def test_reads_more_when_every_item_was_accepted(monkeypatch):
    counts = [100, 200, 300, 200]
    _install_fake_reader(monkeypatch, counts)
    sampler = DistributedNodeBudgetBatchSampler(
        "unused.bp",
        "node_count",
        target_node_count=800,
        node_budget=1000,
        chunk_size=2,
    )

    assert _consume_all(sampler) == [[0, 1, 2, 3]]


def test_from_dataset_reuses_dataset_owned_count_reader(monkeypatch):
    class _Dataset:
        path = "dataset-owned.bp"
        node_count_variable = "trainset/natoms"

        def __init__(self):
            self.counts = np.asarray([400, 400, 400, 400], dtype=np.int64)
            self.reads = []

        def __len__(self):
            return len(self.counts)

        def read_node_counts(self, start, count):
            self.reads.append((start, count))
            return self.counts[start : start + count]

    monkeypatch.setattr(
        sampler_module,
        "_open_adios_reader",
        lambda _path: pytest.fail("sampler opened a second ADIOS reader"),
    )
    dataset = _Dataset()
    sampler = DistributedNodeBudgetBatchSampler.from_dataset(
        dataset,
        target_node_count=800,
        node_budget=800,
        chunk_size=2,
    )

    assert _consume_all(sampler) == [[0, 1], [2, 3]]
    assert dataset.reads == [(0, 2), (2, 2)]


def test_sampler_owned_reader_remains_open_across_epochs(monkeypatch):
    readers = []

    class _ClosableReader(_FakeReader):
        def __init__(self):
            super().__init__([400, 400])
            self.closed = False

        def close(self):
            self.closed = True

    def _open(_path):
        reader = _ClosableReader()
        readers.append(reader)
        return reader

    monkeypatch.setattr(sampler_module, "_open_adios_reader", _open)
    sampler = DistributedNodeBudgetBatchSampler(
        "unused.bp",
        "node_count",
        target_node_count=800,
        node_budget=800,
        chunk_size=2,
    )

    assert _consume_all(sampler) == [[0, 1]]
    sampler.set_epoch(1)
    assert _consume_all(sampler) == [[0, 1]]
    assert len(readers) == 1
    assert not readers[0].closed

    sampler.close()
    assert readers[0].closed


def test_strict_tie_prefers_lower_total(monkeypatch):
    counts = [600, 400]
    _install_fake_reader(monkeypatch, counts)
    sampler = DistributedNodeBudgetBatchSampler(
        "unused.bp",
        "node_count",
        target_node_count=800,
        node_budget=1000,
        chunk_size=2,
    )

    assert _consume_all(sampler) == [[0], [1]]


def test_forward_window_can_swap_to_exact_target(monkeypatch):
    counts = [500, 400, 300]
    _install_fake_reader(monkeypatch, counts)
    sampler = DistributedNodeBudgetBatchSampler(
        "unused.bp",
        "node_count",
        target_node_count=800,
        node_budget=1000,
        chunk_size=2,
        forward_window=1,
    )

    assert _consume_all(sampler) == [[0, 2], [1]]


def test_exact_target_does_not_read_forward_chunk(monkeypatch):
    reads = []
    counts = [500, 300, 100, 100]
    _install_fake_reader(monkeypatch, counts, reads)
    sampler = DistributedNodeBudgetBatchSampler(
        "unused.bp",
        "node_count",
        target_node_count=800,
        node_budget=1000,
        chunk_size=2,
        forward_window=5,
    )

    iterator = iter(sampler)
    assert next(iterator) == [0, 1]
    sampler.mark_batch_consumed()
    iterator.close()
    assert reads == [(0, 2)]


def test_unacknowledged_prefetch_rolls_over_before_fresh_work(monkeypatch):
    counts = [400] * 6
    _install_fake_reader(monkeypatch, counts)
    sampler = DistributedNodeBudgetBatchSampler(
        "unused.bp",
        "node_count",
        target_node_count=800,
        node_budget=800,
        chunk_size=2,
    )

    iterator = iter(sampler)
    assert next(iterator) == [0, 1]
    assert next(iterator) == [2, 3]
    assert next(iterator) == [4, 5]
    sampler.mark_batch_consumed()
    iterator.close()

    assert sampler.rollover_count == 4
    sampler.set_epoch(1)
    assert _consume_all(sampler) == [[2, 3], [4, 5], [0, 1]]


def test_is_compatible_with_multiworker_dataloader(monkeypatch):
    counts = [400] * 6
    _install_fake_reader(monkeypatch, counts)
    sampler = DistributedNodeBudgetBatchSampler(
        "unused.bp",
        "node_count",
        target_node_count=800,
        node_budget=800,
        chunk_size=2,
    )
    loader = DataLoader(list(range(6)), batch_sampler=sampler, num_workers=2)

    batches = []
    for batch in loader:
        assert sampler.sync_batch_available(True)
        batches.append(batch.tolist())
        sampler.mark_batch_consumed()

    assert batches == [[0, 1], [2, 3], [4, 5]]
    assert sampler.rollover_count == 0


def test_fresh_assignment_example():
    assignments, cursor = _fresh_assignments(300, [0, 10, 20], fresh_cursor=300)

    assert assignments == [(300, 100), (400, 90), (490, 80)]
    assert cursor == 570


def test_oversize_is_dropped_with_aggregate_warning(monkeypatch):
    counts = [1200, 500]
    _install_fake_reader(monkeypatch, counts)
    sampler = DistributedNodeBudgetBatchSampler(
        "unused.bp",
        "node_count",
        target_node_count=800,
        node_budget=1000,
        chunk_size=2,
    )

    with pytest.warns(RuntimeWarning, match="dropped 1 sample"):
        assert _consume_all(sampler) == [[1]]
    assert sampler.dropped_count == 1


def test_nonpositive_count_is_rejected(monkeypatch):
    _install_fake_reader(monkeypatch, [0])
    sampler = DistributedNodeBudgetBatchSampler(
        "unused.bp", "node_count", target_node_count=1, node_budget=1
    )

    with pytest.raises(ValueError, match="must be positive"):
        next(iter(sampler))


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "was not found"),
        ({"node_count": {}}, "has no Shape"),
        ({"node_count": {"Shape": "2, 1"}}, "must be 1-D"),
        ({"node_count": {"Shape": "0"}}, "cannot be empty"),
    ],
)
def test_invalid_adios_metadata_is_rejected(monkeypatch, metadata, message):
    class _MetadataReader(_FakeReader):
        def available_variables(self):
            return metadata

    monkeypatch.setattr(
        sampler_module,
        "_open_adios_reader",
        lambda _path: _MetadataReader([1, 1]),
    )
    with pytest.raises(ValueError, match=message):
        DistributedNodeBudgetBatchSampler(
            "unused.bp", "node_count", target_node_count=1, node_budget=1
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_node_count": 0, "node_budget": 1}, "target_node_count"),
        ({"target_node_count": 2, "node_budget": 1}, "cannot exceed"),
        ({"target_node_count": 1, "node_budget": 1, "chunk_size": 0}, "chunk_size"),
        (
            {"target_node_count": 1, "node_budget": 1, "forward_window": -1},
            "forward_window",
        ),
    ],
)
def test_invalid_configuration_is_rejected(monkeypatch, kwargs, message):
    _install_fake_reader(monkeypatch, [1])
    with pytest.raises(ValueError, match=message):
        DistributedNodeBudgetBatchSampler("unused.bp", "node_count", **kwargs)


_DISTRIBUTED_COUNTS = [4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1]


def _distributed_worker(rank, world_size, init_file, result_queue):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    sampler_module._open_adios_reader = lambda _path: _FakeReader(_DISTRIBUTED_COUNTS)
    sampler = DistributedNodeBudgetBatchSampler(
        "unused.bp",
        "node_count",
        target_node_count=4,
        node_budget=4,
        chunk_size=2,
    )

    acknowledged = 0
    iterator = iter(sampler)
    while True:
        try:
            next(iterator)
            local_has_batch = True
        except StopIteration:
            local_has_batch = False

        if not sampler.sync_batch_available(local_has_batch):
            break
        sampler.mark_batch_consumed()
        acknowledged += 1

    iterator.close()
    rollover = sampler.rollover_count
    sampler.set_epoch(1)
    result_queue.put((rank, acknowledged, rollover, sampler.rollover_count))
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_three_rank_handshake_stops_at_shortest_rank(tmp_path):
    world_size = 3
    init_file = str(tmp_path / "gloo-init")
    context = mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, world_size, init_file, result_queue),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    results = sorted(result_queue.get() for _ in range(world_size))
    assert [acknowledged for _, acknowledged, _, _ in results] == [1, 1, 1]
    assert [rollover for _, _, rollover, _ in results] == [3, 2, 0]
    assert [balanced for _, _, _, balanced in results] == [4, 4, 4]


def test_optional_real_adios_dense_read(tmp_path):
    adios2 = pytest.importorskip("adios2")
    path = Path(tmp_path) / "counts.bp"
    counts = np.asarray([100, 200, 300, 1000], dtype=np.int32)
    with adios2.Stream(str(path), "w") as stream:
        stream.write("node_count", counts, list(counts.shape), [0], list(counts.shape))

    sampler = DistributedNodeBudgetBatchSampler(
        path,
        "node_count",
        target_node_count=800,
        node_budget=1000,
        chunk_size=4,
    )
    assert _consume_all(sampler) == [[0, 1, 2], [3]]
