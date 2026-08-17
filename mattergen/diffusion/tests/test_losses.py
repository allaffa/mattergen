# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Dict, List, Type

import pytest
import torch

import mattergen.diffusion.losses as losses_module
from mattergen.diffusion.corruption.corruption import Corruption
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.corruption.sde_lib import SDE, VESDE
from mattergen.diffusion.data.batched_data import SimpleBatchedData, collate_fn
from mattergen.diffusion.losses import DenoisingScoreMatchingLoss, SummedFieldLoss
from mattergen.diffusion.tests.conftest import SDE_TYPES
from mattergen.diffusion.training.field_loss import (
    aggregate_per_sample,
    compute_noise_given_sample_and_corruption,
)
from mattergen.diffusion.wrapped.wrapped_normal_loss import wrapped_normal_loss
from mattergen.diffusion.wrapped.wrapped_sde import WrappedVESDE


def get_multi_corruption(corruption_type, keys: List[str]):
    discrete_corruptions = {
        k: corruption_type()
        for k in keys
        if issubclass(corruption_type, Corruption) and not issubclass(corruption_type, SDE)
    }
    sdes = {k: corruption_type() for k in keys if issubclass(corruption_type, SDE)}
    return MultiCorruption(sdes=sdes, discrete_corruptions=discrete_corruptions)


@pytest.mark.parametrize("corruption_type", SDE_TYPES)
def test_calc_loss(tiny_state_batch, corruption_type: Type[Corruption]):
    """Check that calc_loss returns expected values for a few examples."""

    clean_batch = tiny_state_batch
    multi_corruption = get_multi_corruption(corruption_type=corruption_type, keys=["foo", "bar"])

    t = torch.ones(clean_batch.get_batch_size())
    noisy_batch = multi_corruption.sample_marginal(batch=clean_batch, t=t)

    raw_noise = apply(
        {k: compute_noise_given_sample_and_corruption for k in multi_corruption.corrupted_fields},
        x=clean_batch,
        x_noisy=noisy_batch,
        corruption=multi_corruption.corruptions,
        batch_idx=clean_batch.batch_idx,
        broadcast={"t": t, "batch": clean_batch},
    )

    zero_scores = {k: torch.zeros_like(v) for k, v in clean_batch.data.items()}
    calc_loss = partial(
        DenoisingScoreMatchingLoss(
            model_targets={"foo": "score_times_std"},
        ),
        multi_corruption=multi_corruption,
        t=t,
        batch=clean_batch,
    )

    score_model_output = SimpleBatchedData(data=zero_scores, batch_idx=clean_batch.batch_idx)
    loss, _ = calc_loss(score_model_output=score_model_output, noisy_batch=noisy_batch)
    target_loss = aggregate_per_sample(
        raw_noise["foo"].pow(2),
        batch_idx=clean_batch.batch_idx["foo"],
        reduce="mean",
        batch_size=clean_batch.get_batch_size(),
    ).mean()
    torch.testing.assert_allclose(loss, target_loss)

    # Errors in bar should not affect the loss, only foo.
    score_model_output = score_model_output.replace(bar=score_model_output["bar"] + 100)

    loss_with_bad_bar, _ = calc_loss(score_model_output=score_model_output, noisy_batch=noisy_batch)

    torch.testing.assert_allclose(loss, loss_with_bad_bar)

    # Increasing error in foo should increase the loss; doubling raw noise leads to 4x loss.
    raw_noise.update(foo=raw_noise["foo"] * 2)
    mean, std = multi_corruption.corruptions["foo"].marginal_prob(
        x=clean_batch["foo"],
        t=t[clean_batch.batch_idx["foo"]],
        batch_idx=clean_batch.batch_idx["foo"],
        batch=clean_batch,
    )
    noisy_batch = clean_batch.replace(foo=raw_noise["foo"] * std + mean)
    loss, _ = calc_loss(score_model_output=score_model_output, noisy_batch=noisy_batch)
    torch.testing.assert_allclose(
        loss,
        target_loss * 4,
    )


@pytest.mark.parametrize("corruption_type", SDE_TYPES)
def test_weighted_summed_field_loss(
    tiny_state_batch,
    corruption_type: Type[Corruption],
):
    """Check that SummedFieldLoss returns expected values for a few examples."""

    clean_batch = tiny_state_batch
    multi_corruption = get_multi_corruption(
        corruption_type=corruption_type,
        keys=[
            "foo",
            "bar",
        ],
    )
    zero_scores = {k: torch.zeros_like(v) for k, v in clean_batch.data.items()}
    score_model_output = SimpleBatchedData(data=zero_scores, batch_idx=clean_batch.batch_idx)
    t = torch.ones(clean_batch.get_batch_size())
    noisy_batch = multi_corruption.sample_marginal(batch=clean_batch, t=t)

    weights = {
        "foo": 1.0,
        "bar": 2.9,
    }
    model_targets: Dict[str, str] = {
        k: "score_times_std" for k in multi_corruption.corrupted_fields
    }
    unweighted_loss_fn = DenoisingScoreMatchingLoss(model_targets=model_targets)
    weighted_loss_fn = DenoisingScoreMatchingLoss(
        weights=weights,
        model_targets=model_targets,
    )

    unweighted_loss, unweighted_loss_per_field = unweighted_loss_fn(
        batch=clean_batch,
        multi_corruption=multi_corruption,
        t=t,
        score_model_output=score_model_output,
        noisy_batch=noisy_batch,
    )
    weighted_loss, weighted_loss_per_field = weighted_loss_fn(
        batch=clean_batch,
        multi_corruption=multi_corruption,
        t=t,
        score_model_output=score_model_output,
        noisy_batch=noisy_batch,
    )
    torch.testing.assert_allclose(
        weighted_loss,
        unweighted_loss_per_field["foo"] * weights["foo"]
        + unweighted_loss_per_field["bar"] * weights["bar"],
    )
    torch.testing.assert_allclose(
        torch.stack([unweighted_loss_per_field[k] for k in unweighted_loss_per_field.keys()]),
        torch.stack([weighted_loss_per_field[k] for k in weighted_loss_per_field.keys()]),
    )
    torch.testing.assert_allclose(sum(weighted_loss_per_field.values()), unweighted_loss)


def test_per_atom_mean_reduces_atom_fields_by_atom_count_and_dense_fields_by_batch():
    clean_batch = collate_fn(
        [
            {
                "atomic_numbers": torch.ones(2, dtype=torch.long),
                "foo": torch.randn(2, 3),
                "cell": torch.randn(1, 3),
            },
            {
                "atomic_numbers": torch.ones(5, dtype=torch.long),
                "foo": torch.randn(5, 3),
                "cell": torch.randn(1, 3),
            },
        ],
        dense_field_names=("cell",),
    )
    multi_corruption = MultiCorruption(sdes={"foo": VESDE(), "cell": VESDE()})
    t = torch.ones(clean_batch.get_batch_size())
    noisy_batch = multi_corruption.sample_marginal(batch=clean_batch, t=t)
    raw_noise = apply(
        {k: compute_noise_given_sample_and_corruption for k in multi_corruption.corrupted_fields},
        x=clean_batch,
        x_noisy=noisy_batch,
        corruption=multi_corruption.corruptions,
        batch_idx=multi_corruption._get_batch_indices(clean_batch),
        broadcast={"t": t, "batch": clean_batch},
    )

    zero_scores = {k: torch.zeros_like(v) for k, v in clean_batch.data.items()}
    score_model_output = SimpleBatchedData(data=zero_scores, batch_idx=clean_batch.batch_idx)
    loss_fn = DenoisingScoreMatchingLoss(
        model_targets={"foo": "score_times_std", "cell": "score_times_std"},
        reduce="per_atom_mean",
    )

    loss, loss_per_field = loss_fn(
        batch=clean_batch,
        multi_corruption=multi_corruption,
        t=t,
        score_model_output=score_model_output,
        noisy_batch=noisy_batch,
    )

    foo_per_sample_sum = aggregate_per_sample(
        raw_noise["foo"].pow(2),
        batch_idx=clean_batch.batch_idx["foo"],
        reduce="sum",
        batch_size=clean_batch.get_batch_size(),
    )
    cell_per_sample = aggregate_per_sample(
        raw_noise["cell"].pow(2),
        batch_idx=None,
        reduce="sum",
        batch_size=clean_batch.get_batch_size(),
    )
    expected_foo = foo_per_sample_sum.sum() / clean_batch["atomic_numbers"].shape[0]
    expected_cell = cell_per_sample.mean()

    torch.testing.assert_allclose(loss_per_field["foo"], expected_foo)
    torch.testing.assert_allclose(loss_per_field["cell"], expected_cell)
    torch.testing.assert_allclose(loss, expected_foo + expected_cell)


def test_sum_reduces_atom_fields_within_structure_then_means_over_structures():
    clean_batch = collate_fn(
        [
            {
                "atomic_numbers": torch.ones(2, dtype=torch.long),
                "foo": torch.randn(2, 3),
                "cell": torch.randn(1, 3),
            },
            {
                "atomic_numbers": torch.ones(5, dtype=torch.long),
                "foo": torch.randn(5, 3),
                "cell": torch.randn(1, 3),
            },
        ],
        dense_field_names=("cell",),
    )
    multi_corruption = MultiCorruption(sdes={"foo": VESDE(), "cell": VESDE()})
    t = torch.ones(clean_batch.get_batch_size())
    noisy_batch = multi_corruption.sample_marginal(batch=clean_batch, t=t)
    raw_noise = apply(
        {k: compute_noise_given_sample_and_corruption for k in multi_corruption.corrupted_fields},
        x=clean_batch,
        x_noisy=noisy_batch,
        corruption=multi_corruption.corruptions,
        batch_idx=multi_corruption._get_batch_indices(clean_batch),
        broadcast={"t": t, "batch": clean_batch},
    )

    zero_scores = {k: torch.zeros_like(v) for k, v in clean_batch.data.items()}
    score_model_output = SimpleBatchedData(data=zero_scores, batch_idx=clean_batch.batch_idx)
    loss_fn = DenoisingScoreMatchingLoss(
        model_targets={"foo": "score_times_std", "cell": "score_times_std"},
        reduce="sum",
    )

    loss, loss_per_field = loss_fn(
        batch=clean_batch,
        multi_corruption=multi_corruption,
        t=t,
        score_model_output=score_model_output,
        noisy_batch=noisy_batch,
    )

    foo_per_sample_sum = aggregate_per_sample(
        raw_noise["foo"].pow(2),
        batch_idx=clean_batch.batch_idx["foo"],
        reduce="sum",
        batch_size=clean_batch.get_batch_size(),
    )
    cell_per_sample = aggregate_per_sample(
        raw_noise["cell"].pow(2),
        batch_idx=None,
        reduce="sum",
        batch_size=clean_batch.get_batch_size(),
    )
    expected_foo = foo_per_sample_sum.mean()
    expected_cell = cell_per_sample.mean()

    torch.testing.assert_allclose(loss_per_field["foo"], expected_foo)
    torch.testing.assert_allclose(loss_per_field["cell"], expected_cell)
    torch.testing.assert_allclose(loss, expected_foo + expected_cell)


def test_sum_uses_global_structure_count_for_unequal_rank_batches(monkeypatch):
    per_rank_losses = {
        2: torch.tensor([2.0, 4.0]),
        3: torch.tensor([6.0, 8.0, 10.0]),
    }
    global_batch_size = sum(per_rank_losses)
    world_size = len(per_rank_losses)

    monkeypatch.setattr(
        losses_module,
        "_distributed_sum",
        lambda value: value.new_tensor(global_batch_size),
    )
    monkeypatch.setattr(losses_module, "_world_size", lambda: world_size)
    monkeypatch.setattr(
        losses_module,
        "apply",
        lambda *, broadcast, **_: {"foo": per_rank_losses[broadcast["batch_size"]]},
    )

    loss_fn = SummedFieldLoss(
        loss_fns={"foo": lambda **_: torch.empty(0)},
        model_targets={"foo": "score_times_std"},
        weights={"foo": 2.5},
        reduce="sum",
    )
    multi_corruption = MultiCorruption(sdes={"foo": VESDE()})
    rank_losses = []
    rank_metrics = []
    for local_batch_size in per_rank_losses:
        batch = SimpleBatchedData(
            data={"foo": torch.zeros(local_batch_size, 1)},
            batch_idx={"foo": None},
        )
        local_loss, local_metrics = loss_fn(
            batch=batch,
            multi_corruption=multi_corruption,
            t=torch.ones(local_batch_size),
            score_model_output=batch,
            noisy_batch=batch,
        )
        rank_losses.append(local_loss)
        rank_metrics.append(local_metrics["foo"])

    ddp_loss = torch.stack(rank_losses).mean()
    ddp_metric = torch.stack(rank_metrics).mean()
    expected_metric = torch.cat(list(per_rank_losses.values())).mean()

    torch.testing.assert_close(ddp_metric, expected_metric)
    torch.testing.assert_close(ddp_loss, expected_metric * 2.5)


def test_wrapped_normal_loss(tiny_state_batch):
    # Simulate the case that wrapping has basically no effect and the loss is equivalent to DenoisingScoreMatchingLoss
    clean_batch = tiny_state_batch.replace(
        foo=tiny_state_batch["foo"] + 500, bar=tiny_state_batch["bar"][:, :3] + 500
    )
    fields = ["foo", "bar"]
    multi_corruption: MultiCorruption = MultiCorruption(
        sdes={k: WrappedVESDE(wrapping_boundary=1000.0, sigma_max=1.0) for k in fields}
    )
    model_targets = {k: "score_times_std" for k in fields}
    zero_scores = {k: torch.zeros_like(v) for k, v in clean_batch.data.items()}
    score_model_output = SimpleBatchedData(data=zero_scores, batch_idx=clean_batch.batch_idx)
    t = torch.rand(clean_batch.get_batch_size())
    noisy_batch = multi_corruption.sample_marginal(batch=clean_batch, t=t)
    wrapped_loss_foo = wrapped_normal_loss(
        corruption=multi_corruption.sdes["foo"],
        score_model_output=score_model_output["foo"],
        t=t,
        batch_idx=clean_batch.get_batch_idx("foo"),
        batch_size=clean_batch.get_batch_size(),
        x=clean_batch["foo"],
        noisy_x=noisy_batch["foo"],
        batch=clean_batch,
        reduce="mean",
    ).mean()
    wrapped_loss_bar = wrapped_normal_loss(
        corruption=multi_corruption.sdes["bar"],
        score_model_output=score_model_output["bar"],
        t=t,
        batch_idx=clean_batch.get_batch_idx("bar"),
        batch_size=clean_batch.get_batch_size(),
        x=clean_batch["bar"],
        noisy_x=noisy_batch["bar"],
        batch=clean_batch,
        reduce="mean",
    ).mean()
    wrapped_loss = {"foo": wrapped_loss_foo, "bar": wrapped_loss_bar}
    non_wrapped_loss_fn = DenoisingScoreMatchingLoss(
        model_targets=model_targets,
    )
    _, non_wrapped_loss_per_field = non_wrapped_loss_fn(
        batch=clean_batch,
        multi_corruption=multi_corruption,
        t=t,
        score_model_output=score_model_output,
        noisy_batch=noisy_batch,
    )
    torch.testing.assert_allclose(
        torch.stack([wrapped_loss[k] for k in wrapped_loss.keys()]),
        torch.stack([non_wrapped_loss_per_field[k] for k in non_wrapped_loss_per_field.keys()]),
    )
