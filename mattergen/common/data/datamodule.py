# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mattergen.common.data.dataloader import build_split_dataloader
from mattergen.common.data.dataset import CrystalDataset


class CrystDataModule:
    def __init__(
        self,
        train_dataset: CrystalDataset,
        num_workers: DictConfig,
        batch_size: DictConfig,
        val_dataset: CrystalDataset | None = None,
        test_dataset: CrystalDataset | None = None,
        **_,
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.datasets = [train_dataset, val_dataset, test_dataset]

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        loader, _ = build_split_dataloader(self, "train", distributed=False, shuffle=shuffle)
        assert loader is not None
        return loader

    def val_dataloader(self, shuffle: bool = False) -> DataLoader | None:
        loader, _ = build_split_dataloader(self, "val", distributed=False, shuffle=shuffle)
        return loader

    def test_dataloader(self, shuffle: bool = False) -> DataLoader | None:
        loader, _ = build_split_dataloader(self, "test", distributed=False, shuffle=shuffle)
        return loader

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )
