# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Generic, TypeVar

import torch

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.losses import Loss
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.model_utils import convert_model_out_to_score
from mattergen.diffusion.score_models.base import ScoreModel
from mattergen.diffusion.timestep_samplers import TimestepSampler, UniformTimestepSampler

T = TypeVar("T", bound=BatchedData)
BatchTransform = Callable[[T], T]


def identity(x: T) -> T:
    return x

class DiffusionModule(torch.nn.Module, Generic[T]):
    """Denoising diffusion model for a multi-part state"""

    def __init__(
        self,
        model: ScoreModel[T],
        corruption: MultiCorruption[T],
        loss_fn: Loss,
        pre_corruption_fn: BatchTransform | None = None,
        timestep_sampler: TimestepSampler | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.corruption = corruption
        self.loss_fn = loss_fn
        self.pre_corruption_fn = pre_corruption_fn or identity
        self.model_targets = {k: ModelTarget(v) for k, v in loss_fn.model_targets.items()}

        self.timestep_sampler = timestep_sampler or UniformTimestepSampler(
            min_t=1e-5,
            max_t=corruption.T,
        )

        self._register_corruption_modules()

    def _register_corruption_modules(self):
        """
        Register corruptions that are instances of `torch.nn.Module`s for proper device, parameter,
        etc handling.
        """
        assert isinstance(self.corruption, MultiCorruption)
        for idx, (key, _corruption) in enumerate(self.corruption._corruptions.items()):
            if isinstance(_corruption, torch.nn.Module):
                self.register_module(f"MultiCorruption:{idx}:{key}", _corruption)

    def forward(
        self, batch: T, node_is_unmasked: torch.LongTensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculate loss and metrics given a batch of clean data which may include
        context/conditioning fields. Add noise, predict score using score model, then calculate
        loss.

        This is implemented in `forward()` so DistributedDataParallel can safely wrap it and
        training can call `model(...)`.
        """
        batch = self.pre_corruption_fn(batch)

        noisy_batch, t = self._corrupt_batch(batch)

        score_model_output = self.model(noisy_batch, t)
        loss, metrics = self.loss_fn(
            multi_corruption=self.corruption,
            batch=batch,
            noisy_batch=noisy_batch,
            score_model_output=score_model_output,
            t=t,
            node_is_unmasked=node_is_unmasked,
        )
        assert loss.numel() == 1

        return loss, metrics

    def calc_loss(
        self, batch: T, node_is_unmasked: torch.LongTensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Backward-compatible alias. Prefer calling `model(...)` in training, especially under DDP.
        """
        return self.forward(batch, node_is_unmasked=node_is_unmasked)

    def _corrupt_batch(
        self,
        batch: T,
    ) -> tuple[T, torch.Tensor]:
        t = self.sample_timesteps(batch)
        noisy_batch = self.corruption.sample_marginal(batch, t)
        return noisy_batch, t

    def score_fn(self, x: T, t: torch.Tensor) -> T:
        model_out: T = self.model(x, t)
        fns = {k: convert_model_out_to_score for k in self.corruption.sdes.keys()}

        scores = apply(
            fns=fns,
            model_out=model_out,
            broadcast=dict(t=t, batch=x),
            sde=self.corruption.sdes,
            model_target=self.model_targets,
            batch_idx=self.corruption._get_batch_indices(x),
        )

        return model_out.replace(**scores)

    def sample_timesteps(self, batch: T) -> torch.Tensor:
        return self.timestep_sampler(
            batch_size=batch.get_batch_size(),
            device=self._get_device(batch),
        )

    def _get_device(self, batch: T) -> torch.device:
        return next(batch[k].device for k in self.corruption.sdes.keys())
