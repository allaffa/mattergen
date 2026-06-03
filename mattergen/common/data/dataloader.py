from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mattergen.common.data.collate import collate


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
) -> tuple[DataLoader | None, DistributedSampler | None]:
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
    )
    return loader, sampler
