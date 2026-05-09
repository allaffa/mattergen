# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Generic, Mapping, Optional, Protocol, Sequence, TypeVar, Union

import torch
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer

from mattergen.diffusion.config import Config, resolve_model_module_cfg
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.training_components import (
    build_optimizers_and_schedulers,
    calc_loss,
    get_default_optimizer,
)

T = TypeVar("T", bound=BatchedData)


class OptimizerPartial(Protocol):
    """Callable to instantiate an optimizer."""

    def __call__(self, params: Any) -> Optimizer:
        raise NotImplementedError


class SchedulerPartial(Protocol):
    """Callable to instantiate a learning rate scheduler."""

    def __call__(self, optimizer: Optimizer) -> Any:
        raise NotImplementedError


def _config_from_checkpoint_dict(checkpoint_config: Mapping[str, Any]) -> Config:
    """Build a ``Config`` from a checkpoint payload, migrating legacy keys.

    Older checkpoints persisted the model wrapper config under the
    ``lightning_module`` key. The current ``Config`` only knows about
    ``model_module``, so we migrate the legacy key and drop any other
    keys that are no longer part of the dataclass.
    """
    raw = dict(checkpoint_config)
    if not raw.get("model_module") and raw.get("lightning_module"):
        raw["model_module"] = raw["lightning_module"]
    raw.pop("lightning_module", None)
    known = {f.name for f in dataclasses.fields(Config)}
    filtered = {k: v for k, v in raw.items() if k in known}
    return Config(**filtered)


class DiffusionModelModule(torch.nn.Module, Generic[T]):
    """Model wrapper for instantiating a DiffusionModule and optimizer/scheduler factories."""

    def __init__(
        self,
        diffusion_module: DiffusionModule[T],
        optimizer_partial: Optional[OptimizerPartial] = None,
        scheduler_partials: Optional[Sequence[Dict[str, Union[Any, SchedulerPartial]]]] = None,
    ):
        """_summary_

        Args:
            diffusion_module: The diffusion module to use.
            optimizer_partial: Used to instantiate optimizer.
            scheduler_partials: used to instantiate learning rate schedulers
        """
        super().__init__()
        scheduler_partials = scheduler_partials or []
        optimizer_partial = optimizer_partial or get_default_optimizer

        self.diffusion_module = diffusion_module
        self._optimizer_partial = optimizer_partial
        self._scheduler_partials = scheduler_partials

    @property
    def optimizer_partial(self) -> OptimizerPartial:
        return self._optimizer_partial

    @property
    def scheduler_partials(self) -> Sequence[Dict[str, Union[Any, SchedulerPartial]]]:
        return self._scheduler_partials

    @property
    def device(self) -> torch.device:
        """Return the device hosting this module's parameters.

        PyTorch Lightning exposed ``.device`` on the previous wrapper. Keep the
        same interface for sampling code while using a plain ``torch.nn.Module``.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: Optional[str] = None,
        **kwargs,
    ) -> DiffusionModelModule:
        """Load model from checkpoint. kwargs are passed to hydra's instantiate and can override
        arguments from the checkpoint config."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # The config should have been saved in the checkpoint by AddConfigCallback in run.py
        config = _config_from_checkpoint_dict(checkpoint["config"])
        try:
            model_module = instantiate(resolve_model_module_cfg(config), **kwargs)
        except InstantiationException as e:
            print("Could not instantiate model from the checkpoint.")
            print(
                "If the error is due to an unexpected argument because the checkpoint and the code have diverged, try using load_from_checkpoint_and_config instead."
            )
            raise e
        assert isinstance(model_module, cls)

        # Restore state of the DiffusionModelModule.
        model_module.load_state_dict(checkpoint["state_dict"])
        return model_module

    @classmethod
    def load_from_checkpoint_and_config(
        cls,
        checkpoint_path: str,
        config: DictConfig,
        map_location: Optional[str] = None,
        strict: bool = True,
    ) -> tuple[DiffusionModelModule, torch.nn.modules.module._IncompatibleKeys]:
        """Load model from checkpoint, but instead of using the config stored in the checkpoint,
        use the config passed in as an argument. This is useful when, e.g., an unused argument was
        removed in the code but is still present in the checkpoint config."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model_module = instantiate(config)
        assert isinstance(model_module, cls)

        # Restore state of the DiffusionModelModule.
        result = model_module.load_state_dict(checkpoint["state_dict"], strict=strict)

        return model_module, result

    def configure_optimizers(self) -> Any:
        return build_optimizers_and_schedulers(
            diffusion_module=self.diffusion_module,
            optimizer_partial=self._optimizer_partial,
            scheduler_partials=self._scheduler_partials,
        )

    def training_step(self, train_batch: T, batch_idx: int) -> torch.Tensor:
        return self._calc_loss(train_batch, True)

    def validation_step(self, val_batch: T, batch_idx: int) -> torch.Tensor:
        return self._calc_loss(val_batch, False)

    def test_step(self, test_batch: T, batch_idx: int) -> torch.Tensor:
        return self._calc_loss(test_batch, False)

    def _calc_loss(self, batch: T, train: bool) -> torch.Tensor:
        """Calculate loss and metrics given a batch of clean data."""
        loss, _metrics = calc_loss(self.diffusion_module, batch)
        return loss


# Backward-compatible alias for the previous (PyTorch Lightning era) name.
DiffusionLightningModule = DiffusionModelModule

__all__ = [
    "DiffusionModelModule",
    "DiffusionLightningModule",
    "OptimizerPartial",
    "SchedulerPartial",
]
