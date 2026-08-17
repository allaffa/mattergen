"""
Distributed, node-budget-aware batching for very large ADIOS datasets.
Note: This module is a standalone prototype.

1) The sampler yields lists of:
- map-style dataset indices (can therefore be passed as ``batch_sampler`` to a :class:`torch.utils.data.DataLoader`)

2)The caller must acknowledge batches *after* their optimizer step.  A typical
distributed loop is::

    data_iterator = iter(loader)
    while True:
        try:
            batch = next(data_iterator)
            local_has_batch = True
        except StopIteration:
            batch = None
            local_has_batch = False

        if not sampler.sync_batch_available(local_has_batch):
            break

        train_one_batch(batch)
        sampler.mark_batch_consumed()

    del data_iterator

3) Any batch yielded to DataLoader but not used remains in flight and is rolled over by the next ``set_epoch`` call.
Note: This is what makes the state correct in the presence of DataLoader worker prefetch.
"""

from __future__ import annotations

import re
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

_DROPPED_INDEX_PREVIEW = 8


def _open_adios_reader(path: str):
    """Open an ADIOS FileReader without requiring ADIOS at module import."""

    try:
        import adios2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "DistributedNodeBudgetBatchSampler requires the optional `adios2` "
            "package. Install the project's ADIOS extra before constructing it."
        ) from exc
    return adios2.FileReader(path)


def _shape_from_metadata(shape: Any) -> tuple[int, ...]:
    """Normalize ADIOS's string-valued ``Shape`` metadata."""

    if isinstance(shape, str):
        stripped = shape.strip().strip("{}[]()")
        if not stripped:
            return ()
        parts = [part for part in re.split(r"\s*,\s*|\s+", stripped) if part]
        try:
            return tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"Could not parse ADIOS variable shape {shape!r}.") from exc

    if isinstance(shape, Sequence):
        return tuple(int(part) for part in shape)

    raise ValueError(f"Could not parse ADIOS variable shape {shape!r}.")


def _quotas(dataset_size: int, world_size: int) -> list[int]:
    """Return the usual quotient/remainder dataset partition sizes."""

    base, remainder = divmod(dataset_size, world_size)
    return [base + int(rank < remainder) for rank in range(world_size)]


def _fresh_assignments(
    dataset_size: int,
    rollover_counts: Sequence[int],
    fresh_cursor: int,
) -> tuple[list[tuple[int, int]], int]:
    """Compute rank-local fresh stream ranges after accounting for rollover."""

    quotas = _quotas(dataset_size, len(rollover_counts))
    fresh_counts: list[int] = []
    for rank, (quota, rollover) in enumerate(zip(quotas, rollover_counts)):
        rollover = int(rollover)
        if rollover < 0:
            raise ValueError(f"Rank {rank} reported a negative rollover count: {rollover}.")
        if rollover > quota:
            raise RuntimeError(
                f"Rank {rank} has {rollover} rollover samples but its per-epoch "
                f"quota is only {quota}."
            )
        fresh_counts.append(quota - rollover)

    assignments: list[tuple[int, int]] = []
    start = int(fresh_cursor)
    for count in fresh_counts:
        assignments.append((start, count))
        start += count
    return assignments, start


@dataclass(frozen=True)
class _Item:
    logical_position: int
    node_count: int

    def physical_index(self, dataset_size: int) -> int:
        return self.logical_position % dataset_size


class DistributedNodeBudgetBatchSampler(Sampler[list[int]]):
    """
    Pack sequential ADIOS samples under a target per-batch node count as well as a hard per-batch node cap.

    Work is represented by monotonically increasing *logical* positions in a
    repeated dataset stream.  Only the bounded lookahead is materialized as
    individual ``_Item`` objects; unread work remains compact intervals.

    ``set_epoch`` supports monotonically increasing epochs only.  Exact restart
    serialization is deliberately outside this prototype's scope.
    """

    def __init__(
        self,
        adios_path: str | Path,
        node_count_variable: str,
        target_node_count: int,
        node_budget: int,
        *,
        chunk_size: int = 32,
        forward_window: int = 0,
        process_group: Any | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        communication_device: str | torch.device | None = None,
        node_count_reader: Callable[[int, int], Sequence[int]] | None = None,
        dataset_size: int | None = None,
    ) -> None:
        if target_node_count <= 0:
            raise ValueError("target_node_count must be positive.")
        if node_budget <= 0:
            raise ValueError("node_budget must be positive.")
        if target_node_count > node_budget:
            raise ValueError("target_node_count cannot exceed node_budget.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if forward_window < 0:
            raise ValueError("forward_window cannot be negative.")
        if not node_count_variable:
            raise ValueError("node_count_variable cannot be empty.")

        self.adios_path = str(adios_path)
        self.node_count_variable = node_count_variable
        self.target_node_count = int(target_node_count)
        self.node_budget = int(node_budget)
        self.chunk_size = int(chunk_size)
        self.forward_window = int(forward_window)
        self.process_group = process_group

        distributed = dist.is_available() and dist.is_initialized()
        if process_group is not None and not distributed:
            raise ValueError("process_group was provided but torch.distributed is not initialized.")

        detected_world_size = dist.get_world_size(process_group) if distributed else 1
        detected_rank = dist.get_rank(process_group) if distributed else 0
        self.world_size = detected_world_size if world_size is None else int(world_size)
        self.rank = detected_rank if rank is None else int(rank)

        if self.world_size <= 0:
            raise ValueError("world_size must be positive.")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}.")
        if distributed and (
            self.world_size != detected_world_size or self.rank != detected_rank
        ):
            raise ValueError(
                "Explicit rank/world_size do not match the initialized process group: "
                f"got ({self.rank}, {self.world_size}), expected "
                f"({detected_rank}, {detected_world_size})."
            )
        if self.world_size > 1 and not distributed:
            raise ValueError("world_size > 1 requires an initialized torch.distributed group.")

        self.communication_device = self._resolve_communication_device(communication_device)
        self._owned_reader: Any | None = None
        if node_count_reader is None:
            self._owned_reader = _open_adios_reader(self.adios_path)
            self.dataset_size = self._read_dataset_size(self._owned_reader)
            self._node_count_reader = self._read_owned_node_counts
        else:
            if dataset_size is None or int(dataset_size) <= 0:
                raise ValueError(
                    "dataset_size must be positive when node_count_reader is provided."
                )
            self.dataset_size = int(dataset_size)
            self._node_count_reader = node_count_reader
        if self.dataset_size < self.world_size:
            raise ValueError(
                f"dataset_size ({self.dataset_size}) must be at least world_size "
                f"({self.world_size}); otherwise one rank would end every epoch before "
                "any training step."
            )

        quotas = _quotas(self.dataset_size, self.world_size)
        initial_start = sum(quotas[: self.rank])
        initial_count = quotas[self.rank]

        self._pending: Deque[_Item] = deque()
        self._ranges: Deque[tuple[int, int]] = deque()
        if initial_count:
            self._ranges.append((initial_start, initial_start + initial_count))
        self._inflight: Deque[tuple[_Item, ...]] = deque()

        self._fresh_cursor = self.dataset_size
        self._epoch = 0
        self._iterator_active = False
        self._iteration_started = False
        self._dropped_count = 0
        self._dropped_indices: list[int] = []
        self._drop_warning_emitted = False

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        target_node_count: int,
        node_budget: int,
        **kwargs: Any,
    ) -> "DistributedNodeBudgetBatchSampler":
        """Build a sampler using a dataset-owned persistent count reader."""

        read_node_counts = getattr(dataset, "read_node_counts", None)
        if read_node_counts is None:
            raise TypeError(
                f"{type(dataset).__name__} must provide read_node_counts(start, count)."
            )
        path = getattr(dataset, "path", getattr(dataset, "filename", "<dataset>"))
        variable = getattr(dataset, "node_count_variable", "node_count")
        return cls(
            path,
            variable,
            target_node_count,
            node_budget,
            node_count_reader=read_node_counts,
            dataset_size=len(dataset),
            **kwargs,
        )

    @property
    def rollover_count(self) -> int:
        """Number of assigned logical occurrences not yet acknowledged."""

        inflight = sum(len(batch) for batch in self._inflight)
        unread = sum(stop - start for start, stop in self._ranges)
        return inflight + len(self._pending) + unread

    @property
    def dropped_count(self) -> int:
        """Number of over-budget occurrences dropped in the current epoch."""

        return self._dropped_count

    @property
    def epoch(self) -> int:
        return self._epoch

    def _resolve_communication_device(
        self, requested: str | torch.device | None
    ) -> torch.device:
        if requested is not None:
            return torch.device(requested)
        if self.world_size == 1:
            return torch.device("cpu")

        backend = str(dist.get_backend(self.process_group)).lower()
        if "nccl" in backend:
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL sampler communication requires an available CUDA device.")
            return torch.device("cuda", torch.cuda.current_device())
        if "xccl" in backend:
            if not hasattr(torch, "xpu") or not torch.xpu.is_available():
                raise RuntimeError("XCCL sampler communication requires an available XPU device.")
            return torch.device("xpu", torch.xpu.current_device())
        return torch.device("cpu")

    def _read_dataset_size(self, reader: Any) -> int:
        variables = reader.available_variables()
        if self.node_count_variable not in variables:
            raise ValueError(
                f"ADIOS variable {self.node_count_variable!r} was not found in "
                f"{self.adios_path!r}."
            )
        metadata = variables[self.node_count_variable]
        if "Shape" not in metadata:
            raise ValueError(
                f"ADIOS variable {self.node_count_variable!r} has no Shape metadata."
            )
        shape = _shape_from_metadata(metadata["Shape"])

        if len(shape) != 1:
            raise ValueError(
                f"ADIOS variable {self.node_count_variable!r} must be 1-D; got shape {shape}."
            )
        if shape[0] <= 0:
            raise ValueError("The node-count dataset cannot be empty.")
        return shape[0]

    def _read_owned_node_counts(self, start: int, count: int) -> Sequence[int]:
        assert self._owned_reader is not None
        return self._owned_reader.read(
            self.node_count_variable,
            start=[start],
            count=[count],
            step_selection=[0, 1],
        )

    def __iter__(self) -> Iterator[list[int]]:
        # DataLoader may create and discard an index iterator during its own
        # initialization, before advancing it.  Activate lazily in _iterate so
        # that harmless unstarted iterators do not block the real one.
        return self._iterate()

    def _iterate(self) -> Iterator[list[int]]:
        if self._iterator_active:
            raise RuntimeError("Only one active sampler iterator is supported.")
        self._iterator_active = True
        self._iteration_started = True
        try:
            while True:
                batch = self._build_next_batch()
                if not batch:
                    return
                ordered = tuple(sorted(batch, key=lambda item: item.logical_position))
                self._inflight.append(ordered)
                yield [item.physical_index(self.dataset_size) for item in ordered]
        finally:
            self._iterator_active = False
            self._emit_drop_warning()

    def mark_batch_consumed(self) -> None:
        """Acknowledge the oldest yielded batch after its optimizer step."""

        if not self._inflight:
            raise RuntimeError("No yielded batch is waiting to be acknowledged.")
        self._inflight.popleft()

    def sync_batch_available(self, local_has_batch: bool) -> bool:
        """Return whether every rank fetched a batch for the next DDP step."""

        if not isinstance(local_has_batch, (bool, np.bool_)):
            raise TypeError("local_has_batch must be a bool.")
        if self.world_size == 1:
            return bool(local_has_batch)

        flag = torch.tensor(
            [1 if local_has_batch else 0],
            dtype=torch.int32,
            device=self.communication_device,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.process_group)
        return bool(flag.item())

    def set_epoch(self, epoch: int) -> None:
        """Finalize rollover and add this rank's fresh range for ``epoch``."""

        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch cannot be negative.")
        if epoch == self._epoch:
            # ``set_epoch(0)`` before the first iterator mirrors
            # DistributedSampler's normal calling convention.
            if not self._iteration_started:
                return
            raise RuntimeError(f"Epoch {epoch} has already started.")
        if epoch != self._epoch + 1:
            raise ValueError(
                f"Epochs must advance one at a time; current={self._epoch}, requested={epoch}."
            )
        if self._iterator_active:
            raise RuntimeError(
                "Close or destroy the current DataLoader iterator before calling set_epoch()."
            )

        self._emit_drop_warning()
        self._consolidate_explicit_rollover()
        local_rollover = self.rollover_count

        if self.world_size == 1:
            assignments, new_cursor = _fresh_assignments(
                self.dataset_size, [local_rollover], self._fresh_cursor
            )
            fresh_start, fresh_count = assignments[0]
        else:
            fresh_start, fresh_count, new_cursor = self._distributed_fresh_assignment(
                local_rollover
            )

        if fresh_count:
            self._append_range(fresh_start, fresh_start + fresh_count)
        self._fresh_cursor = new_cursor
        self._epoch = epoch
        self._iteration_started = False
        self._dropped_count = 0
        self._dropped_indices.clear()
        self._drop_warning_emitted = False

    def _distributed_fresh_assignment(self, local_rollover: int) -> tuple[int, int, int]:
        local = torch.tensor(
            [local_rollover], dtype=torch.int64, device=self.communication_device
        )
        root_group_rank = 0
        if self.process_group is None:
            root_global_rank = 0
        elif hasattr(dist, "get_global_rank"):
            root_global_rank = dist.get_global_rank(self.process_group, root_group_rank)
        else:  # pragma: no cover - compatibility with old PyTorch
            root_global_rank = 0

        gathered = (
            [torch.empty_like(local) for _ in range(self.world_size)]
            if self.rank == root_group_rank
            else None
        )
        dist.gather(
            local,
            gather_list=gathered,
            dst=root_global_rank,
            group=self.process_group,
        )

        scatter_list: list[torch.Tensor] | None = None
        if self.rank == root_group_rank:
            assert gathered is not None
            rollover_counts = [int(value.item()) for value in gathered]
            assignments, new_cursor = _fresh_assignments(
                self.dataset_size, rollover_counts, self._fresh_cursor
            )
            scatter_list = [
                torch.tensor(
                    [start, count, new_cursor],
                    dtype=torch.int64,
                    device=self.communication_device,
                )
                for start, count in assignments
            ]

        assignment = torch.empty(3, dtype=torch.int64, device=self.communication_device)
        dist.scatter(
            assignment,
            scatter_list=scatter_list,
            src=root_global_rank,
            group=self.process_group,
        )
        values = assignment.tolist()
        return int(values[0]), int(values[1]), int(values[2])

    def _consolidate_explicit_rollover(self) -> None:
        explicit = list(self._pending)
        for batch in self._inflight:
            explicit.extend(batch)
        explicit.sort(key=lambda item: item.logical_position)
        self._pending = deque(explicit)
        self._inflight.clear()

    def _append_range(self, start: int, stop: int) -> None:
        if start >= stop:
            return
        if self._ranges and self._ranges[-1][1] == start:
            previous_start, _ = self._ranges.pop()
            self._ranges.append((previous_start, stop))
        else:
            self._ranges.append((start, stop))

    def _take_chunk(self) -> list[_Item]:
        chunk: list[_Item] = []
        while self._pending and len(chunk) < self.chunk_size:
            chunk.append(self._pending.popleft())

        while self._ranges and len(chunk) < self.chunk_size:
            start, stop = self._ranges[0]
            physical_start = start % self.dataset_size
            count = min(
                self.chunk_size - len(chunk),
                stop - start,
                self.dataset_size - physical_start,
            )
            values = self._node_count_reader(physical_start, count)
            array = np.asarray(values).reshape(-1)
            if array.size != count:
                raise RuntimeError(
                    f"ADIOS returned {array.size} counts for a requested chunk of {count}."
                )
            chunk.extend(
                _Item(logical_position=start + offset, node_count=int(value))
                for offset, value in enumerate(array)
            )

            next_start = start + count
            self._ranges.popleft()
            if next_start < stop:
                self._ranges.appendleft((next_start, stop))

        return chunk

    def _restore_deferred(self, deferred: Sequence[_Item]) -> None:
        for item in reversed(sorted(deferred, key=lambda value: value.logical_position)):
            self._pending.appendleft(item)

    def _record_drop(self, item: _Item) -> None:
        self._dropped_count += 1
        if len(self._dropped_indices) < _DROPPED_INDEX_PREVIEW:
            self._dropped_indices.append(item.physical_index(self.dataset_size))

    def _build_next_batch(self) -> list[_Item]:
        selected: list[_Item] = []
        deferred: list[_Item] = []
        total = 0
        target_reached = False
        forward_chunks_remaining = self.forward_window

        while True:
            chunk = self._take_chunk()
            if not chunk:
                break

            chunk_rejected = False
            item_index = 0
            while item_index < len(chunk):
                item = chunk[item_index]
                item_index += 1

                if item.node_count <= 0:
                    self._restore_deferred(deferred + chunk[item_index:])
                    raise ValueError(
                        f"Node count must be positive; index "
                        f"{item.physical_index(self.dataset_size)} has {item.node_count}."
                    )
                if item.node_count > self.node_budget:
                    self._record_drop(item)
                    continue

                if not selected:
                    selected.append(item)
                    total = item.node_count
                    target_reached = total >= self.target_node_count
                elif target_reached:
                    total, rejected = self._consider_add_or_swap(
                        item, selected, deferred, total
                    )
                    chunk_rejected = chunk_rejected or rejected
                else:
                    candidate_total = total + item.node_count
                    improves = (
                        candidate_total <= self.node_budget
                        and abs(candidate_total - self.target_node_count)
                        < abs(total - self.target_node_count)
                    )
                    if improves:
                        selected.append(item)
                        total = candidate_total
                        target_reached = total >= self.target_node_count
                    else:
                        deferred.append(item)
                        chunk_rejected = True

                if total == self.target_node_count:
                    deferred.extend(chunk[item_index:])
                    self._restore_deferred(deferred)
                    return selected

            if target_reached:
                if forward_chunks_remaining == 0:
                    break
                forward_chunks_remaining -= 1
                continue

            # Below target, another chunk is read only when every valid item in
            # this one was accepted.  This prevents arbitrarily long scans when
            # the remaining budget cannot accommodate deferred large samples.
            if chunk_rejected:
                break

        self._restore_deferred(deferred)
        return selected

    def _consider_add_or_swap(
        self,
        item: _Item,
        selected: list[_Item],
        deferred: list[_Item],
        total: int,
    ) -> tuple[int, bool]:
        """Apply the best strictly improving add or one-item swap."""

        current_distance = abs(total - self.target_node_count)
        # key, replacement index, new total.  Add wins an otherwise identical
        # tie, followed by replacement of the earliest logical position.
        moves: list[tuple[tuple[int, int, int, int], int | None, int]] = []

        added_total = total + item.node_count
        if added_total <= self.node_budget:
            distance = abs(added_total - self.target_node_count)
            if distance < current_distance:
                moves.append(((distance, added_total, 0, item.logical_position), None, added_total))

        for index, old_item in enumerate(selected):
            swapped_total = total - old_item.node_count + item.node_count
            if not 0 < swapped_total <= self.node_budget:
                continue
            distance = abs(swapped_total - self.target_node_count)
            if distance < current_distance:
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

    def _emit_drop_warning(self) -> None:
        if self._drop_warning_emitted or self._dropped_count == 0:
            return
        warnings.warn(
            f"Rank {self.rank} dropped {self._dropped_count} sample occurrence(s) in "
            f"epoch {self._epoch} because their node count exceeded node_budget="
            f"{self.node_budget}. First physical indices: {self._dropped_indices}.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._drop_warning_emitted = True

    def close(self) -> None:
        """Close the sampler-owned ADIOS reader, if any."""

        if self._owned_reader is None:
            return
        close = getattr(self._owned_reader, "close", None)
        if close is not None:
            close()
        self._owned_reader = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


__all__ = ["DistributedNodeBudgetBatchSampler"]
