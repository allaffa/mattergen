from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.backends import AdiosFormat, detect_format
from mattergen.common.data.dataloader import build_split_dataloader
from mattergen.common.data.node_budget_sampler import (
    NodeCountProvider,
    StreamingNodeBudgetBatchSampler,
    stateless_permute_index,
)


class _MetadataDataset:
    def __init__(self, counts):
        self.num_atoms = np.asarray(counts, dtype=np.int64)
        self.materialized: list[int] = []

    def __len__(self):
        return len(self.num_atoms)

    def __getitem__(self, index):
        index = int(index)
        self.materialized.append(index)
        count = int(self.num_atoms[index])
        return SimpleNamespace(num_nodes=count)


class _ChemGraphDataset(_MetadataDataset):
    def __getitem__(self, index):
        index = int(index)
        self.materialized.append(index)
        count = int(self.num_atoms[index])
        return ChemGraph(
            pos=torch.zeros(count, 3),
            cell=torch.eye(3).unsqueeze(0),
            atomic_numbers=torch.ones(count, dtype=torch.long),
            num_atoms=torch.tensor(count),
            num_nodes=count,
        )


class _IndexedCountDataset:
    def __init__(self, counts):
        self.counts = list(counts)
        self.total_node_count = sum(self.counts)
        self.requests = []

    def __len__(self):
        return len(self.counts)

    def read_node_counts(self, indices):
        self.requests.append(list(indices))
        return [self.counts[index] for index in indices]

    def __getitem__(self, _):
        raise AssertionError("metadata reader should avoid sample materialization")


class _RangeCountDataset(_IndexedCountDataset):
    read_node_counts = None

    def read_node_counts_range(self, start, count):
        self.requests.append((start, count))
        return self.counts[start : start + count]


def test_adios_bp_directory_is_detected(tmp_path):
    cache = tmp_path / "cache"
    (cache / "dataset.bp").mkdir(parents=True)
    assert detect_format(str(cache)) == AdiosFormat


@pytest.mark.parametrize("size", [1, 2, 3, 17, 257])
def test_stateless_permutation_is_a_deterministic_bijection(size):
    first = [stateless_permute_index(i, size, seed=9, traversal=3) for i in range(size)]
    repeated = [stateless_permute_index(i, size, seed=9, traversal=3) for i in range(size)]
    assert sorted(first) == list(range(size))
    assert repeated == first
    if size >= 17:
        next_traversal = [
            stateless_permute_index(i, size, seed=9, traversal=4) for i in range(size)
        ]
        assert next_traversal != first


def test_rank_ranges_are_disjoint_and_steps_are_fixed():
    dataset = _MetadataDataset([1] * 12)
    rank0 = StreamingNodeBudgetBatchSampler(
        dataset, 2, target_nodes=2, steps_per_epoch=2, num_replicas=2, rank=0, shuffle=False
    )
    rank1 = StreamingNodeBudgetBatchSampler(
        dataset, 2, target_nodes=2, steps_per_epoch=2, num_replicas=2, rank=1, shuffle=False
    )
    batches0 = list(rank0)
    batches1 = list(rank1)
    assert len(batches0) == len(batches1) == 2
    assert set(sum(batches0, [])).isdisjoint(sum(batches1, []))
    assert all(sum(dataset.num_atoms[batch]) <= 2 for batch in batches0 + batches1)


def test_stream_rolls_over_without_resetting_at_training_epoch():
    dataset = _MetadataDataset([1, 1, 1, 1])
    sampler = StreamingNodeBudgetBatchSampler(
        dataset, 2, target_nodes=2, steps_per_epoch=3, shuffle=False
    )
    assert list(sampler) == [[0, 1], [2, 3], [0, 1]]
    assert sampler.statistics().traversal_boundaries == 1
    sampler.set_epoch(1)
    assert list(sampler)[0] == [2, 3]


def test_packing_respects_node_and_graph_limits():
    dataset = _MetadataDataset([4, 3, 2, 2, 1, 1])
    sampler = StreamingNodeBudgetBatchSampler(
        dataset,
        max_nodes=6,
        target_nodes=5,
        max_graphs=2,
        steps_per_epoch=4,
        shuffle=False,
    )
    for batch in sampler:
        assert len(batch) <= 2
        assert sum(int(dataset.num_atoms[index]) for index in batch) <= 6


def test_oversized_policies():
    dataset = _MetadataDataset([7, 2, 2])
    with pytest.raises(ValueError, match="exceeding max_nodes"):
        list(
            StreamingNodeBudgetBatchSampler(
                dataset, 5, steps_per_epoch=1, shuffle=False, oversized_sample="error"
            )
        )

    skipped = StreamingNodeBudgetBatchSampler(
        dataset, 5, steps_per_epoch=1, shuffle=False, oversized_sample="skip"
    )
    assert list(skipped) == [[1, 2]]
    assert skipped.statistics().skipped_samples == 1

    single = StreamingNodeBudgetBatchSampler(
        dataset, 5, steps_per_epoch=1, shuffle=False, oversized_sample="single"
    )
    assert list(single) == [[0]]


def test_metadata_and_sample_sources_emit_identical_batches():
    auto_dataset = _MetadataDataset([2, 5, 3, 4, 1, 6, 2, 3])
    sample_dataset = _MetadataDataset(auto_dataset.num_atoms.copy())
    common = dict(
        max_nodes=8,
        target_nodes=7,
        steps_per_epoch=4,
        metadata_chunk_size=3,
        seed=17,
    )
    auto = StreamingNodeBudgetBatchSampler(auto_dataset, metadata_source="auto", **common)
    with pytest.warns(RuntimeWarning, match="materialize"):
        sample = StreamingNodeBudgetBatchSampler(
            sample_dataset, metadata_source="sample", **common
        )

    assert list(auto) == list(sample)
    assert auto.statistics().node_count_source == "resident_num_atoms"
    assert auto.statistics().sample_materializations == 0
    assert sample.statistics().node_count_source == "sample"
    assert sample.statistics().sample_materializations > 0


def test_indexed_and_range_metadata_readers_are_used_without_samples():
    indexed_dataset = _IndexedCountDataset([2, 4, 3, 5])
    indexed = NodeCountProvider(indexed_dataset)
    assert indexed.read_node_counts([3, 1]) == [5, 4]
    assert indexed.source == "indexed_reader"
    assert indexed.sample_materializations == 0

    range_dataset = _RangeCountDataset([2, 4, 3, 5])
    ranged = NodeCountProvider(range_dataset)
    assert ranged.read_node_counts([3, 1, 2]) == [5, 4, 3]
    assert ranged.source == "range_reader"
    assert ranged.sample_materializations == 0
    assert range_dataset.requests == [(1, 3)]


def test_sample_source_requires_explicit_steps_and_cache_is_bounded():
    dataset = _MetadataDataset([1] * 20)
    with pytest.warns(RuntimeWarning, match="materialize"):
        with pytest.raises(ValueError, match="requires steps_per_epoch"):
            StreamingNodeBudgetBatchSampler(dataset, 4, metadata_source="sample")

    with pytest.warns(RuntimeWarning, match="materialize"):
        provider = NodeCountProvider(dataset, metadata_source="sample", cache_size=3)
    provider.read_node_counts(list(range(10)))
    assert len(provider._cache) == 3


def test_build_split_dataloader_uses_streaming_batch_sampler():
    dataset = _ChemGraphDataset([2, 3, 4, 1, 2, 5])
    datamodule = SimpleNamespace(
        train_dataset=dataset,
        batch_size=OmegaConf.create({"train": 4}),
        num_workers=OmegaConf.create({"train": 0}),
        batching=OmegaConf.create(
            {
                "mode": "streaming_node_budget",
                "max_nodes": 6,
                "target_nodes": 5,
                "steps_per_epoch": 2,
                "shuffle": False,
            }
        ),
    )
    loader, sampler = build_split_dataloader(
        datamodule, "train", distributed=False, shuffle=True
    )
    assert isinstance(sampler, StreamingNodeBudgetBatchSampler)
    assert len(loader) == 2
    for batch in loader:
        assert int(batch.num_atoms.sum()) <= 6

    fixed_loader, fixed_sampler = build_split_dataloader(
        datamodule,
        "train",
        distributed=False,
        shuffle=False,
        use_streaming_batching=False,
    )
    assert fixed_sampler is None
    assert len(fixed_loader) == 2
