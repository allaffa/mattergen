from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, TypeVar, Union

import torch
from torch.optim import AdamW, Optimizer

from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule

T = TypeVar("T", bound=BatchedData)


class OptimizerPartial(Protocol):
    def __call__(self, params: Any) -> Optimizer:
        raise NotImplementedError


class SchedulerPartial(Protocol):
    def __call__(self, optimizer: Optimizer) -> Any:
        raise NotImplementedError


def get_default_optimizer(params):
    return AdamW(params=params, lr=1e-4, weight_decay=0, amsgrad=True)


def build_optimizers_and_schedulers(
    diffusion_module: DiffusionModule[T],
    optimizer_partial: Optional[OptimizerPartial] = None,
    scheduler_partials: Optional[Sequence[dict[str, Union[Any, SchedulerPartial]]]] = None,
) -> Any:
    scheduler_partials = scheduler_partials or []
    optimizer_partial = optimizer_partial or get_default_optimizer
    optimizer = optimizer_partial(params=diffusion_module.parameters())
    if scheduler_partials:
        lr_schedulers = [
            {
                **scheduler_dict,
                "scheduler": scheduler_dict["scheduler"](
                    optimizer=optimizer,
                ),
            }
            for scheduler_dict in scheduler_partials
        ]

        return [
            optimizer,
        ], lr_schedulers
    return optimizer


def calc_loss(diffusion_module: DiffusionModule[T], batch: T) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return diffusion_module.calc_loss(batch)
