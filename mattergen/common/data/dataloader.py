from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

from mattergen.common.data.collate import collate
from mattergen.common.data.node_budget_sampler import StreamingNodeBudgetBatchSampler


logger = logging.getLogger(__name__)


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


def _split_attr(name: str) -> str:
    return f"{name}_dataset"


def _node_limits(dataset: Any, batching: Any) -> tuple[int, int]:
    """Resolve explicit limits or average-sample multipliers to node counts."""

    configured_max = batching.get("max_nodes")
    configured_target = batching.get("target_nodes")
    average_max = batching.get("max_average_samples")
    average_target = batching.get("target_average_samples")

    if configured_max is not None:
        if average_max is not None or average_target is not None:
            raise ValueError(
                "Configure either max_nodes/target_nodes or average-sample limits, not both"
            )
        maximum = int(configured_max)
        target = maximum if configured_target is None else int(configured_target)
        return target, maximum

    if average_max is None:
        raise ValueError(
            "streaming_node_budget batching requires max_nodes or max_average_samples"
        )

    total_nodes = getattr(dataset, "total_node_count", None)
    if total_nodes is None:
        getter = getattr(dataset, "get_total_node_count", None)
        if callable(getter):
            total_nodes = getter()
    if total_nodes is None:
        raise ValueError(
            "average-sample node limits require exact dataset total_node_count metadata"
        )

    average_atoms = float(total_nodes) / len(dataset)
    max_multiplier = float(average_max)
    target_multiplier = (
        max_multiplier if average_target is None else float(average_target)
    )
    target = int(round(target_multiplier * average_atoms))
    maximum = int(round(max_multiplier * average_atoms))
    logger.info(
        "streaming sampler limits: average_atoms_per_sample=%.8f "
        "target=%s*average=%s max=%s*average=%s",
        average_atoms,
        target_multiplier,
        target,
        max_multiplier,
        maximum,
    )
    return target, maximum


def build_split_dataloader(
    datamodule: Any,
    split: str,
    *,
    distributed: bool,
    shuffle: bool,
    use_streaming_batching: bool = True,
) -> tuple[DataLoader | None, Sampler | None]:
    dataset = getattr(datamodule, _split_attr(split), None)
    if dataset is None:
        loader_method = getattr(datamodule, f"{split}_dataloader", None)
        if loader_method is None:
            return None, None
        if distributed:
            raise ValueError(
                f"Cannot create distributed {split} dataloader without `{split}_dataset` attribute."
            )
        return loader_method(shuffle=shuffle), None

    batch_size_cfg = getattr(datamodule, "batch_size")
    num_workers_cfg = getattr(datamodule, "num_workers")
    batch_size = int(getattr(batch_size_cfg, split))
    num_workers = int(getattr(num_workers_cfg, split))

    batching = getattr(datamodule, "batching", None)
    batching_mode = "fixed" if batching is None else str(batching.get("mode", "fixed"))
    if batching_mode not in {"fixed", "streaming_node_budget"}:
        raise ValueError(f"Unknown batching mode {batching_mode!r}.")

    if use_streaming_batching and split == "train" and batching_mode == "streaming_node_budget":
        target_nodes, max_nodes = _node_limits(dataset, batching)

        num_replicas = 1
        rank = 0
        if distributed:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("distributed streaming batching requires initialized torch.distributed")
            num_replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

        batch_sampler = StreamingNodeBudgetBatchSampler(
            dataset,
            max_nodes=max_nodes,
            target_nodes=target_nodes,
            steps_per_epoch=batching.get("steps_per_epoch"),
            num_replicas=num_replicas,
            rank=rank,
            max_graphs=batching.get("max_graphs"),
            metadata_chunk_size=int(batching.get("metadata_chunk_size", 32)),
            metadata_cache_size=batching.get("metadata_cache_size"),
            forward_window=int(batching.get("forward_window", 1)),
            shuffle=bool(batching.get("shuffle", shuffle)),
            seed=int(batching.get("seed", 0)),
            oversized_sample=str(batching.get("oversized_sample", "error")),
            metadata_source=str(batching.get("metadata_source", "auto")),
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, batch_sampler

    sampler = None
    dataloader_shuffle = shuffle
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader_shuffle = False

    loader = DataLoader(
        dataset,
        shuffle=dataloader_shuffle,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate,
        # pin_memory=True,
        # persistent_workers=True,
        # prefetch_factor=4
    )
    return loader, sampler
