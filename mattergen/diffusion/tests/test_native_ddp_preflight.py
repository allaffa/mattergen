from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mattergen.diffusion.native_ddp import (
    _TRAINING_LIMIT_NONE,
    _TRAINING_LIMIT_STEPS,
    _TRAINING_LIMIT_TIME,
    _ddp_all_gather_preflight,
    _capture_optimizer_lr_settings,
    _restore_optimizer_lr_settings,
    _step_schedulers,
    _synchronize_training_limit,
    _training_limit_code,
)


def _loopback_sockets_available() -> bool:
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            probe.listen()
    except OSError:
        return False
    return True


requires_loopback = pytest.mark.skipif(
    not _loopback_sockets_available(),
    reason="this environment forbids the loopback sockets required by Gloo",
)


def _preflight_worker(rank: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        _ddp_all_gather_preflight(torch.device("cpu"), rank, 2)
    finally:
        dist.destroy_process_group()


@requires_loopback
def test_pre_ddp_all_gather_validates_rank_order(tmp_path):
    mp.spawn(
        _preflight_worker,
        args=(str(tmp_path / "preflight-init"),),
        nprocs=2,
        join=True,
    )


def test_training_limits_stop_at_first_configured_boundary():
    assert (
        _training_limit_code(
            global_step=599,
            max_steps=600,
            elapsed_seconds=1799.0,
            max_train_seconds=1800.0,
        )
        == _TRAINING_LIMIT_NONE
    )
    assert (
        _training_limit_code(
            global_step=600,
            max_steps=600,
            elapsed_seconds=10.0,
            max_train_seconds=1800.0,
        )
        == _TRAINING_LIMIT_STEPS
    )
    assert (
        _training_limit_code(
            global_step=10,
            max_steps=600,
            elapsed_seconds=1800.0,
            max_train_seconds=1800.0,
        )
        == _TRAINING_LIMIT_TIME
    )


def test_optimizer_lr_can_be_restored_after_loading_checkpoint_state():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-5)
    configured_lr_settings = _capture_optimizer_lr_settings(optimizer)

    checkpoint_parameter = torch.nn.Parameter(torch.tensor(1.0))
    checkpoint_optimizer = torch.optim.Adam([checkpoint_parameter], lr=1e-4)
    optimizer.load_state_dict(checkpoint_optimizer.state_dict())

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)

    _restore_optimizer_lr_settings(optimizer, configured_lr_settings)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


class _CountingScheduler:
    def __init__(self) -> None:
        self.steps: list[float | None] = []

    def step(self, value: float | None = None) -> None:
        self.steps.append(value)


def test_step_scheduler_accepts_plural_steps_interval():
    scheduler = _CountingScheduler()

    _step_schedulers(
        [{"scheduler": scheduler, "interval": "steps", "frequency": 1}],
        when="step",
        event_count=1,
    )

    assert scheduler.steps == [None]


def test_scheduler_frequency_is_honored():
    scheduler = _CountingScheduler()
    scheduler_cfgs = [{"scheduler": scheduler, "interval": "steps", "frequency": 2}]

    _step_schedulers(scheduler_cfgs, when="step", event_count=1)
    _step_schedulers(scheduler_cfgs, when="step", event_count=2)

    assert scheduler.steps == [None]


def test_monitored_scheduler_uses_named_metric():
    scheduler = _CountingScheduler()

    _step_schedulers(
        [{"scheduler": scheduler, "interval": "epoch", "monitor": "loss_train"}],
        when="epoch",
        event_count=1,
        metrics={"loss_train": 1.25, "loss_val": 9.0},
    )

    assert scheduler.steps == [1.25]


def _training_limit_worker(rank: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        local_code = _TRAINING_LIMIT_STEPS if rank == 0 else _TRAINING_LIMIT_NONE
        synchronized_code = _synchronize_training_limit(
            local_code,
            device=torch.device("cpu"),
            distributed=True,
            rank=rank,
        )
        assert synchronized_code == _TRAINING_LIMIT_STEPS
    finally:
        dist.destroy_process_group()


@requires_loopback
def test_training_limit_is_synchronized_from_rank_zero(tmp_path):
    mp.spawn(
        _training_limit_worker,
        args=(str(tmp_path / "training-limit-init"),),
        nprocs=2,
        join=True,
    )
