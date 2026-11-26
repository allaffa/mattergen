import torch

from mattergen.common.loss import MaterialsLoss
from mattergen.diffusion.native_ddp import _format_cond_metrics
from mattergen.property_embeddings import _USE_UNCONDITIONAL_EMBEDDING


def test_extra_metrics_split_by_embedding_type():
    loss = MaterialsLoss(reduce="sum", weights={"pos": 0.1, "cell": 1.0, "atomic_numbers": 1.0})
    per_sample = {
        "pos": torch.tensor([10.0, 20.0, 30.0]),
        "cell": torch.tensor([1.0, 2.0, 3.0]),
        "atomic_numbers": torch.tensor([1.0, 1.0, 1.0]),
    }
    batch = {_USE_UNCONDITIONAL_EMBEDDING: {"force_rms": torch.tensor([[False], [True], [False]])}}

    metrics = loss.extra_metrics(batch, per_sample)

    # weighted totals: [3, 5, 7]
    assert metrics["cond/force_rms/loss_sum_on"].item() == 10.0
    assert metrics["cond/force_rms/n_on"].item() == 2.0
    assert metrics["cond/force_rms/loss_sum_off"].item() == 5.0
    assert metrics["cond/force_rms/n_off"].item() == 1.0

    summary = _format_cond_metrics(metrics)
    assert summary == "force_rms: cond=5.0000 uncond=5.0000 gap=+0.0000"


def test_extra_metrics_without_conditioning_is_empty():
    loss = MaterialsLoss(reduce="sum")
    assert loss.extra_metrics({}, {"pos": torch.zeros(2)}) == {}


def test_format_handles_empty_group():
    metrics = {
        "cond/chemical_system/loss_sum_on": torch.tensor(4.0),
        "cond/chemical_system/n_on": torch.tensor(2.0),
        "cond/chemical_system/loss_sum_off": torch.tensor(0.0),
        "cond/chemical_system/n_off": torch.tensor(0.0),
    }
    summary = _format_cond_metrics(metrics)
    assert summary.startswith("chemical_system: cond=2.0000 uncond=nan")
