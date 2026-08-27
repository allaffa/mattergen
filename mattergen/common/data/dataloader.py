from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

from mattergen.common.data.collate import collate
from mattergen.common.data.node_budget_sampler import StreamingNodeBudgetBatchSampler


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
        if batching.get("max_nodes") is None:
            raise ValueError("streaming_node_budget batching requires max_nodes")

        num_replicas = 1
        rank = 0
        if distributed:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("distributed streaming batching requires initialized torch.distributed")
            num_replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

        batch_sampler = StreamingNodeBudgetBatchSampler(
            dataset,
            max_nodes=int(batching.get("max_nodes")),
            target_nodes=batching.get("target_nodes"),
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
