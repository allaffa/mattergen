# Copyright (c) 2026, Oak Ridge National Laboratory
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Stateless, fixed-step node-budget batching for variable-size graphs.

The streaming sampler is adapted from HydraGNN commit
``63e463774d7548063484f28fc3e7465dc7e93fcd``.  It intentionally keeps only
bounded lookahead state and never constructs a dense dataset permutation or a
complete epoch batch plan.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import math
import warnings

from torch.utils.data import Sampler


@dataclass(frozen=True)
class StreamingBatchStatistics:
    """Diagnostics for the most recently started fixed-step training epoch."""

    training_epoch: int
    configured_steps: int
    emitted_steps: int
    emitted_samples: int
    emitted_nodes: int
    min_nodes: int
    max_nodes: int
    mean_nodes: float
    min_utilization: float
    max_utilization: float
    mean_utilization: float
    start_traversal: int
    end_traversal: int
    traversal_boundaries: int
    deferred_samples: int
    oversized_samples: int
    skipped_samples: int
    node_count_source: str
    node_count_requests: int
    node_count_cache_hits: int
    sample_materializations: int


_UINT64_MASK = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
_PERMUTATION_ROUNDS = 6


def _mix_uint64(value: int) -> int:
    mixed = (int(value) + _SPLITMIX_INCREMENT) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 30)) * _SPLITMIX_MULTIPLIER_1) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * _SPLITMIX_MULTIPLIER_2) & _UINT64_MASK
    return (mixed ^ (mixed >> 31)) & _UINT64_MASK


def _permutation_key(seed: int, traversal: int) -> int:
    return _mix_uint64((int(seed) & _UINT64_MASK) ^ _mix_uint64(traversal))


def _permute_power_of_two(value: int, bit_width: int, key: int) -> int:
    mask = (1 << bit_width) - 1
    mixed = int(value) & mask
    for round_index in range(_PERMUTATION_ROUNDS):
        round_key = _mix_uint64(key + round_index * _SPLITMIX_INCREMENT)
        mixed = (mixed + (round_key & mask)) & mask
        multiplier = ((round_key >> 17) | 1) & mask
        mixed = (mixed * (multiplier or 1)) & mask
        shift = 1 + ((round_key >> 41) % bit_width)
        mixed = (mixed ^ (mixed >> shift)) & mask
    return mixed


def stateless_permute_index(index: int, dataset_size: int, seed: int, traversal: int) -> int:
    """Map one index through a traversal-keyed bijection without dense state."""

    size = int(dataset_size)
    index = int(index)
    if size <= 0:
        raise ValueError("dataset_size must be positive")
    if not 0 <= index < size:
        raise ValueError(f"index must be in [0, {size}), got {index}")
    if size == 1:
        return 0

    bit_width = (size - 1).bit_length()
    key = _permutation_key(seed, traversal)
    permuted = _permute_power_of_two(index, bit_width, key)
    while permuted >= size:
        permuted = _permute_power_of_two(permuted, bit_width, key)
    return permuted


def graph_node_cost(sample: object) -> int:
    """Return the number of nodes from a graph-like sample."""

    value = getattr(sample, "num_nodes", None)
    if value is None:
        value = getattr(sample, "num_atoms", None)
    if value is None:
        features = getattr(sample, "x", None)
        if features is None:
            raise ValueError("sample has no num_nodes, num_atoms, or node features x")
        value = len(features)
    return int(value)


class NodeCountProvider:
    """Bounded, backend-neutral access to graph node counts.

    ``metadata_source='sample'`` deliberately disables every metadata path.
    This is useful both for datasets without count metadata and for proving
    that the fallback behaves identically to a metadata-backed run.
    """

    def __init__(
        self,
        dataset,
        *,
        metadata_source: str = "auto",
        cost_fn: Callable[[object], int] = graph_node_cost,
        costs: Sequence[int] | None = None,
        total_node_count: int | None = None,
        cache_size: int = 256,
    ) -> None:
        if metadata_source not in {"auto", "sample"}:
            raise ValueError("metadata_source must be 'auto' or 'sample'")

        self.dataset = dataset
        self.metadata_source = metadata_source
        self.cost_fn = cost_fn
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[int, int] = OrderedDict()
        self.requests = 0
        self.cache_hits = 0
        self.sample_materializations = 0
        self.source = "sample"
        self.costs: Sequence[int] | None = None
        self._reader = None
        self._range_reader = None

        if metadata_source == "auto":
            self._discover_metadata(costs)
            if self.costs is not None and len(self.costs) != len(dataset):
                raise ValueError("node-count metadata must contain one value per dataset sample")

        if total_node_count is None and metadata_source == "auto":
            total_node_count = getattr(dataset, "total_node_count", None)
            if total_node_count is None:
                getter = getattr(dataset, "get_total_node_count", None)
                if callable(getter):
                    total_node_count = getter()
            if total_node_count is None and self.costs is not None:
                total_node_count = sum(int(value) for value in self.costs)
        self.total_node_count = None if total_node_count is None else int(total_node_count)

        if self.source == "sample":
            warnings.warn(
                "streaming node-budget batching will materialize dataset samples "
                "to discover node counts",
                RuntimeWarning,
                stacklevel=2,
            )

    def _discover_metadata(self, costs: Sequence[int] | None) -> None:
        if costs is not None:
            self.costs = costs
            self.source = "provided_costs"
            return

        resident = getattr(self.dataset, "num_atoms", None)
        if resident is not None and len(resident) == len(self.dataset):
            self.costs = resident
            self.source = "resident_num_atoms"
            return

        reader = getattr(self.dataset, "read_node_counts", None)
        if callable(reader):
            self._reader = reader
            self.source = "indexed_reader"
            return

        range_reader = getattr(self.dataset, "read_node_counts_range", None)
        if callable(range_reader):
            self._range_reader = range_reader
            self.source = "range_reader"
            return

        getter = getattr(self.dataset, "get_node_counts", None)
        if callable(getter):
            values = getter()
            if values is not None:
                self.costs = values
                self.source = "get_node_counts"

    def _remember(self, index: int, value: int) -> None:
        if self.cache_size == 0:
            return
        self._cache[index] = value
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _read_ranges(self, indices: Sequence[int]) -> list[int]:
        assert self._range_reader is not None
        unique = sorted(set(indices))
        runs: list[tuple[int, int]] = []
        run_start = unique[0]
        run_stop = run_start + 1
        for index in unique[1:]:
            if index == run_stop:
                run_stop += 1
            else:
                runs.append((run_start, run_stop))
                run_start, run_stop = index, index + 1
        runs.append((run_start, run_stop))

        counts_by_index: dict[int, int] = {}
        self.requests += len(runs)
        for start, stop in runs:
            values = list(self._range_reader(start, stop - start))
            if len(values) != stop - start:
                raise RuntimeError("node-count range reader returned the wrong number of values")
            counts_by_index.update(
                (start + offset, int(value)) for offset, value in enumerate(values)
            )
        return [counts_by_index[index] for index in indices]

    def read_node_counts(self, indices: Sequence[int]) -> list[int]:
        normalized = [int(index) for index in indices]
        result: list[int | None] = [None] * len(normalized)
        missing_positions: list[int] = []
        missing_indices: list[int] = []

        for position, index in enumerate(normalized):
            if not 0 <= index < len(self.dataset):
                raise IndexError(index)
            if index in self._cache:
                self.cache_hits += 1
                result[position] = self._cache[index]
                self._cache.move_to_end(index)
            else:
                missing_positions.append(position)
                missing_indices.append(index)

        if missing_indices:
            if self._range_reader is not None:
                values = self._read_ranges(missing_indices)
            elif self._reader is not None:
                self.requests += 1
                values = list(self._reader(missing_indices))
            elif self.costs is not None:
                self.requests += 1
                values = [self.costs[index] for index in missing_indices]
            else:
                self.requests += 1
                self.sample_materializations += len(missing_indices)
                values = [self.cost_fn(self.dataset[index]) for index in missing_indices]

            if len(values) != len(missing_indices):
                raise RuntimeError("node-count provider returned the wrong number of values")
            for position, index, raw_value in zip(
                missing_positions, missing_indices, values
            ):
                value = int(raw_value)
                result[position] = value
                self._remember(index, value)

        return [int(value) for value in result]

    def close(self) -> None:
        close = getattr(self.dataset, "close_node_count_reader", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class _StreamingItem:
    logical_position: int
    physical_index: int
    node_count: int
    traversal: int


class StreamingNodeBudgetBatchSampler(Sampler[list[int]]):
    """Yield fixed-step node-budget batches from a continuous shuffled stream."""

    is_streaming_node_budget = True

    def __init__(
        self,
        dataset,
        max_nodes: int,
        *,
        steps_per_epoch: int | None = None,
        target_nodes: int | None = None,
        num_replicas: int = 1,
        rank: int = 0,
        max_graphs: int | None = None,
        metadata_chunk_size: int = 32,
        forward_window: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        oversized_sample: str = "error",
        metadata_source: str = "auto",
        provider: NodeCountProvider | None = None,
        costs: Sequence[int] | None = None,
        total_node_count: int | None = None,
        cost_fn: Callable[[object], int] = graph_node_cost,
        metadata_cache_size: int | None = None,
    ) -> None:
        if len(dataset) <= 0:
            raise ValueError("streaming node-budget dataset cannot be empty")
        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        target_nodes = max_nodes if target_nodes is None else int(target_nodes)
        if target_nodes <= 0:
            raise ValueError("target_nodes must be positive")
        if target_nodes > max_nodes:
            raise ValueError("target_nodes cannot exceed max_nodes")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if len(dataset) < num_replicas:
            raise ValueError("dataset size must be at least num_replicas")
        if max_graphs is not None and max_graphs <= 0:
            raise ValueError("max_graphs must be positive when provided")
        if metadata_chunk_size <= 0:
            raise ValueError("metadata_chunk_size must be positive")
        if forward_window < 0:
            raise ValueError("forward_window cannot be negative")
        if oversized_sample not in {"error", "single", "skip"}:
            raise ValueError("oversized_sample must be one of: error, single, skip")

        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.max_nodes = int(max_nodes)
        self.target_nodes = target_nodes
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.max_graphs = max_graphs
        self.metadata_chunk_size = int(metadata_chunk_size)
        self.forward_window = int(forward_window)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.oversized_sample = oversized_sample
        cache_size = (
            8 * self.metadata_chunk_size
            if metadata_cache_size is None
            else int(metadata_cache_size)
        )
        self.provider = provider or NodeCountProvider(
            dataset,
            metadata_source=metadata_source,
            cost_fn=cost_fn,
            costs=costs,
            total_node_count=total_node_count,
            cache_size=cache_size,
        )

        if metadata_source == "sample" and steps_per_epoch is None:
            raise ValueError(
                "streaming_node_budget requires steps_per_epoch when "
                "metadata_source='sample'"
            )
        if steps_per_epoch is None:
            total = self.provider.total_node_count
            if total is None:
                raise ValueError(
                    "streaming_node_budget requires steps_per_epoch when the "
                    "dataset has no exact total_node_count metadata"
                )
            steps_per_epoch = math.ceil(total / (self.target_nodes * self.num_replicas))
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.steps_per_epoch = int(steps_per_epoch)

        self.training_epoch = 0
        self.traversal = 0
        self._started = False
        self._initial_epoch_set = False
        self._iterator_active = False
        self._pending: deque[_StreamingItem] = deque()
        self._range_start = 0
        self._range_stop = 0
        self._next_position = 0
        self._batches_in_traversal = 0
        self._last_batch_traversal = 0
        self._reset_epoch_statistics()
        self._start_traversal(0)

    @property
    def last_batch_traversal(self) -> int:
        return self._last_batch_traversal

    @property
    def logical_cursor(self) -> tuple[int, int]:
        return self.traversal, self._next_position

    def _quota(self, rank: int) -> int:
        base, remainder = divmod(self.dataset_size, self.num_replicas)
        return base + int(rank < remainder)

    def _start_traversal(self, traversal: int) -> None:
        self.traversal = int(traversal)
        self._range_start = sum(self._quota(rank) for rank in range(self.rank))
        self._range_stop = self._range_start + self._quota(self.rank)
        self._next_position = self._range_start
        self._pending.clear()
        self._batches_in_traversal = 0

    def _reset_epoch_statistics(self) -> None:
        self._stats_start_traversal = getattr(self, "traversal", 0)
        self._stats_boundaries = 0
        self._stats_steps = 0
        self._stats_samples = 0
        self._stats_nodes = 0
        self._stats_min_nodes: int | None = None
        self._stats_max_nodes = 0
        self._stats_deferred = 0
        self._stats_oversized = 0
        self._stats_skipped = 0
        provider = getattr(self, "provider", None)
        self._stats_provider_requests = getattr(provider, "requests", 0)
        self._stats_provider_hits = getattr(provider, "cache_hits", 0)
        self._stats_sample_materializations = getattr(provider, "sample_materializations", 0)

    def set_epoch(self, epoch: int) -> None:
        """Record a training epoch without resetting an active stream."""

        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        if self._iterator_active:
            raise RuntimeError("cannot change epoch while a sampler iterator is active")
        if not self._started and not self._initial_epoch_set:
            self._start_traversal(epoch)
        self._initial_epoch_set = True
        self.training_epoch = epoch
        self._reset_epoch_statistics()

    def _physical_index(self, logical_position: int) -> int:
        if not self.shuffle:
            return logical_position
        return stateless_permute_index(
            logical_position,
            self.dataset_size,
            seed=self.seed,
            traversal=self.traversal,
        )

    def _take_chunk(self) -> list[_StreamingItem]:
        chunk: list[_StreamingItem] = []
        while self._pending and len(chunk) < self.metadata_chunk_size:
            chunk.append(self._pending.popleft())

        count = min(
            self.metadata_chunk_size - len(chunk),
            self._range_stop - self._next_position,
        )
        if count <= 0:
            return chunk

        positions = list(range(self._next_position, self._next_position + count))
        indices = [self._physical_index(position) for position in positions]
        node_counts = self.provider.read_node_counts(indices)
        chunk.extend(
            _StreamingItem(position, index, nodes, self.traversal)
            for position, index, nodes in zip(positions, indices, node_counts)
        )
        self._next_position += count
        return chunk

    def _restore_deferred(self, items: Sequence[_StreamingItem]) -> None:
        self._stats_deferred += len(items)
        for item in reversed(sorted(items, key=lambda value: value.logical_position)):
            self._pending.appendleft(item)

    def _consider_add_or_swap(
        self,
        item: _StreamingItem,
        selected: list[_StreamingItem],
        deferred: list[_StreamingItem],
        total: int,
    ) -> tuple[int, bool]:
        current_distance = abs(total - self.target_nodes)
        moves: list[tuple[tuple[int, int, int, int], int | None, int]] = []

        added_total = total + item.node_count
        if (
            (self.max_graphs is None or len(selected) < self.max_graphs)
            and added_total <= self.max_nodes
            and abs(added_total - self.target_nodes) < current_distance
        ):
            moves.append(
                (
                    (
                        abs(added_total - self.target_nodes),
                        added_total,
                        0,
                        item.logical_position,
                    ),
                    None,
                    added_total,
                )
            )

        for index, old_item in enumerate(selected):
            swapped_total = total - old_item.node_count + item.node_count
            distance = abs(swapped_total - self.target_nodes)
            if 0 < swapped_total <= self.max_nodes and distance < current_distance:
                moves.append(
                    (
                        (distance, swapped_total, 1, old_item.logical_position),
                        index,
                        swapped_total,
                    )
                )

        if not moves:
            deferred.append(item)
            return total, True

        _, replacement_index, new_total = min(moves, key=lambda move: move[0])
        if replacement_index is None:
            selected.append(item)
        else:
            deferred.append(selected[replacement_index])
            selected[replacement_index] = item
        return new_total, False

    def _build_batch(self) -> list[_StreamingItem]:
        selected: list[_StreamingItem] = []
        deferred: list[_StreamingItem] = []
        total = 0
        target_reached = False
        forward_chunks_remaining = self.forward_window

        while True:
            chunk = self._take_chunk()
            if not chunk:
                break
            chunk_rejected = False

            for item_index, item in enumerate(chunk):
                if item.node_count <= 0:
                    self._restore_deferred(deferred + chunk[item_index + 1 :])
                    raise ValueError(
                        f"node count must be positive; sample "
                        f"{item.physical_index} has {item.node_count}"
                    )
                if item.node_count > self.max_nodes:
                    self._stats_oversized += 1
                    if self.oversized_sample == "error":
                        self._restore_deferred(deferred + chunk[item_index + 1 :])
                        raise ValueError(
                            f"sample {item.physical_index} has cost "
                            f"{item.node_count}, exceeding max_nodes {self.max_nodes}"
                        )
                    if self.oversized_sample == "skip":
                        self._stats_skipped += 1
                        continue
                    if selected:
                        deferred.append(item)
                        deferred.extend(chunk[item_index + 1 :])
                        self._restore_deferred(deferred)
                        return selected
                    deferred.extend(chunk[item_index + 1 :])
                    self._restore_deferred(deferred)
                    return [item]

                if not selected:
                    selected.append(item)
                    total = item.node_count
                    target_reached = total >= self.target_nodes
                elif self.max_graphs is not None and len(selected) >= self.max_graphs:
                    deferred.append(item)
                    deferred.extend(chunk[item_index + 1 :])
                    self._restore_deferred(deferred)
                    return selected
                elif target_reached:
                    total, rejected = self._consider_add_or_swap(
                        item, selected, deferred, total
                    )
                    chunk_rejected = chunk_rejected or rejected
                else:
                    candidate_total = total + item.node_count
                    improves = (
                        candidate_total <= self.max_nodes
                        and abs(candidate_total - self.target_nodes)
                        < abs(total - self.target_nodes)
                    )
                    if improves:
                        selected.append(item)
                        total = candidate_total
                        target_reached = total >= self.target_nodes
                    else:
                        deferred.append(item)
                        chunk_rejected = True

                if total == self.target_nodes:
                    deferred.extend(chunk[item_index + 1 :])
                    self._restore_deferred(deferred)
                    return selected

            if target_reached:
                if forward_chunks_remaining == 0:
                    break
                forward_chunks_remaining -= 1
            elif chunk_rejected:
                break

        self._restore_deferred(deferred)
        return selected

    def _next_batch(self) -> list[_StreamingItem]:
        while True:
            batch = self._build_batch()
            if batch:
                self._batches_in_traversal += 1
                return sorted(batch, key=lambda item: item.logical_position)
            if self._pending or self._next_position < self._range_stop:
                raise RuntimeError("streaming packer made no progress")
            if self._batches_in_traversal == 0:
                raise RuntimeError(
                    f"rank {self.rank} could not emit a valid batch from "
                    f"traversal {self.traversal}"
                )
            self._stats_boundaries += 1
            self._start_traversal(self.traversal + 1)

    def __iter__(self) -> Iterator[list[int]]:
        return self._iterate()

    def _iterate(self) -> Iterator[list[int]]:
        if self._iterator_active:
            raise RuntimeError("only one active streaming sampler iterator is supported")
        self._iterator_active = True
        self._started = True
        try:
            for _ in range(self.steps_per_epoch):
                batch = self._next_batch()
                self._last_batch_traversal = batch[0].traversal
                node_total = sum(item.node_count for item in batch)
                self._stats_steps += 1
                self._stats_samples += len(batch)
                self._stats_nodes += node_total
                self._stats_min_nodes = (
                    node_total
                    if self._stats_min_nodes is None
                    else min(self._stats_min_nodes, node_total)
                )
                self._stats_max_nodes = max(self._stats_max_nodes, node_total)
                yield [item.physical_index for item in batch]
        finally:
            self._iterator_active = False

    def __len__(self) -> int:
        return self.steps_per_epoch

    def statistics(self) -> StreamingBatchStatistics:
        min_nodes = 0 if self._stats_min_nodes is None else self._stats_min_nodes
        mean_nodes = self._stats_nodes / self._stats_steps if self._stats_steps else 0.0
        return StreamingBatchStatistics(
            training_epoch=self.training_epoch,
            configured_steps=self.steps_per_epoch,
            emitted_steps=self._stats_steps,
            emitted_samples=self._stats_samples,
            emitted_nodes=self._stats_nodes,
            min_nodes=min_nodes,
            max_nodes=self._stats_max_nodes,
            mean_nodes=mean_nodes,
            min_utilization=min_nodes / self.max_nodes,
            max_utilization=self._stats_max_nodes / self.max_nodes,
            mean_utilization=mean_nodes / self.max_nodes,
            start_traversal=self._stats_start_traversal,
            end_traversal=self.traversal,
            traversal_boundaries=self._stats_boundaries,
            deferred_samples=self._stats_deferred,
            oversized_samples=self._stats_oversized,
            skipped_samples=self._stats_skipped,
            node_count_source=self.provider.source,
            node_count_requests=self.provider.requests - self._stats_provider_requests,
            node_count_cache_hits=self.provider.cache_hits - self._stats_provider_hits,
            sample_materializations=(
                self.provider.sample_materializations - self._stats_sample_materializations
            ),
        )

    def close(self) -> None:
        self.provider.close()


__all__ = [
    "NodeCountProvider",
    "StreamingBatchStatistics",
    "StreamingNodeBudgetBatchSampler",
    "graph_node_cost",
    "stateless_permute_index",
]
