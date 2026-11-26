# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Dict, Literal, Optional

import torch

from mattergen.diffusion.losses import LossReduction, SummedFieldLoss, denoising_score_matching
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.training.field_loss import FieldLoss, d3pm_loss
from mattergen.diffusion.wrapped.wrapped_normal_loss import wrapped_normal_loss
from mattergen.property_embeddings import _USE_UNCONDITIONAL_EMBEDDING


class MaterialsLoss(SummedFieldLoss):
    def __init__(
        self,
        reduce: LossReduction = "mean",
        d3pm_hybrid_lambda: float = 0.0,
        include_pos: bool = True,
        include_cell: bool = True,
        include_atomic_numbers: bool = True,
        weights: Optional[Dict[str, float]] = None,
    ):
        field_reduce: Literal["sum", "mean"] = "sum" if reduce == "per_atom_mean" else reduce
        model_targets = {"pos": ModelTarget.score_times_std, "cell": ModelTarget.score_times_std}
        self.fields_to_score = []
        self.categorical_fields = []
        loss_fns: Dict[str, FieldLoss] = {}
        if include_pos:
            self.fields_to_score.append("pos")
            loss_fns["pos"] = partial(
                wrapped_normal_loss,
                reduce=field_reduce,
                model_target=model_targets["pos"],
            )
        if include_cell:
            self.fields_to_score.append("cell")
            loss_fns["cell"] = partial(
                denoising_score_matching,
                reduce=field_reduce,
                model_target=model_targets["cell"],
            )
        if include_atomic_numbers:
            model_targets["atomic_numbers"] = ModelTarget.logits
            self.fields_to_score.append("atomic_numbers")
            self.categorical_fields.append("atomic_numbers")
            loss_fns["atomic_numbers"] = partial(
                d3pm_loss,
                reduce=field_reduce,
                d3pm_hybrid_lambda=d3pm_hybrid_lambda,
            )
        self.reduce = reduce
        self.d3pm_hybrid_lambda = d3pm_hybrid_lambda
        super().__init__(
            loss_fns=loss_fns,
            weights=weights,
            model_targets=model_targets,
            reduce=reduce,
        )

    def extra_metrics(self, batch, loss_per_sample_per_field):
        # Sums and counts (not means) so ratios stay exact after mean-reducing across ranks.
        try:
            use_uncond = batch[_USE_UNCONDITIONAL_EMBEDDING]
        except (KeyError, AttributeError):
            return {}
        with torch.no_grad():
            total = sum(
                self.loss_weights[k] * v.detach().float()
                for k, v in loss_per_sample_per_field.items()
            )
            metrics = {}
            for field, uncond in use_uncond.items():
                cond = ~uncond.reshape(-1).to(total.device)
                metrics[f"cond/{field}/loss_sum_on"] = total[cond].sum()
                metrics[f"cond/{field}/n_on"] = cond.sum().float()
                metrics[f"cond/{field}/loss_sum_off"] = total[~cond].sum()
                metrics[f"cond/{field}/n_off"] = (~cond).sum().float()
        return metrics
