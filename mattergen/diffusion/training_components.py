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


def get_warmup_piecewise_constant_lr_scheduler(
    optimizer: Optimizer,
    warmup_iters: int = 100_000,
    start_factor: float = 0.01,
    end_factor: float = 1.0,
    step_boundaries: Sequence[int] = (500_000, 1_000_000),
    factors: Sequence[float] = (1.0, 0.5, 0.1),
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm up linearly, then keep piecewise-constant LR multipliers."""
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative.")
    if len(factors) != len(step_boundaries) + 1:
        raise ValueError("factors must have exactly one more value than step_boundaries.")

    boundaries = tuple(int(boundary) for boundary in step_boundaries)
    if any(boundary <= 0 for boundary in boundaries):
        raise ValueError("step_boundaries must be positive.")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("step_boundaries must be strictly increasing.")

    def lr_lambda(current_step: int) -> float:
        if warmup_iters > 0 and current_step < warmup_iters:
            progress = current_step / warmup_iters
            return start_factor + (end_factor - start_factor) * progress

        for boundary, factor in zip(boundaries, factors):
            if current_step < boundary:
                return factor
        return factors[-1]

    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


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
