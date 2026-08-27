from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.corruption.sde_lib import VESDE
from mattergen.diffusion.data.batched_data import SimpleBatchedData
from mattergen.diffusion.losses import SummedFieldLoss


def _prediction_square_loss(*, score_model_output, **_):
    return score_model_output.square().reshape(-1)


def _rank_local_sum_worker(rank: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        model = DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            model.module.weight.fill_(1.0)

        values = (
            torch.tensor([[1.0], [2.0]])
            if rank == 0
            else torch.tensor([[3.0], [4.0], [5.0]])
        )
        prediction = model(values)
        batch = SimpleBatchedData(data={"foo": values}, batch_idx={"foo": None})
        output = SimpleBatchedData(data={"foo": prediction}, batch_idx={"foo": None})
        loss_fn = SummedFieldLoss(
            loss_fns={"foo": _prediction_square_loss},
            model_targets={"foo": "score_times_std"},
            reduce="sum",
        )
        loss, fields = loss_fn(
            multi_corruption=MultiCorruption(sdes={"foo": VESDE()}),
            batch=batch,
            noisy_batch=batch,
            score_model_output=output,
            t=torch.ones(batch.get_batch_size()),
        )
        loss.backward()

        # rank sums are 5 and 50; DDP averages their gradients:
        # (2*5 + 2*50) / 2 = 55. A global sample mean would be 22.
        torch.testing.assert_close(model.module.weight.grad, torch.tensor([[55.0]]))

        reported = fields["foo"].detach().clone()
        dist.all_reduce(reported, op=dist.ReduceOp.SUM)
        reported /= dist.get_world_size()
        torch.testing.assert_close(reported, torch.tensor(27.5))
    finally:
        dist.destroy_process_group()


def test_ddp_averages_unequal_rank_local_sums(tmp_path):
    mp.spawn(
        _rank_local_sum_worker,
        args=(str(tmp_path / "ddp-init"),),
        nprocs=2,
        join=True,
    )
