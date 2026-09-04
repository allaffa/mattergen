from __future__ import annotations

import pytest

from PosTestJobs.summarize_pos_loss import RUN_RE, parse_metric_lines


def test_parse_position_loss_metric_line():
    text = (
        "INFO:mattergen.diffusion.native_ddp:"
        "epoch=7 step=3 global_step=18 elapsed_train_seconds=54.125 "
        "lr=1.00e-04 loss_train=2.500000 pos_train=0.125000 "
        "cell_train=0.250000 atom_train=0.500000\n"
    )

    assert parse_metric_lines(text) == [
        {
            "epoch": 7,
            "epoch_step": 3,
            "global_step": 18,
            "elapsed_train_seconds": pytest.approx(54.125),
            "lr": pytest.approx(1.0e-4),
            "loss_train": pytest.approx(2.5),
            "pos_train": pytest.approx(0.125),
            "cell_train": pytest.approx(0.25),
            "atom_train": pytest.approx(0.5),
        }
    ]


def test_parse_position_loss_metric_line_preserves_nan():
    text = (
        "epoch=0 step=0 global_step=1 elapsed_train_seconds=3.0 "
        "lr=1e-4 loss_train=nan pos_train=NaN cell_train=1 atom_train=2"
    )

    row = parse_metric_lines(text)[0]
    assert row["global_step"] == 1
    assert row["loss_train"] != row["loss_train"]
    assert row["pos_train"] != row["pos_train"]


def test_timestep_sweep_run_directory_name():
    match = RUN_RE.fullmatch("run-2-samples-1024-t-exp-4-5415000")

    assert match is not None
    assert tuple(map(int, match.groups())) == (2, 1024, 4, 5415000)
    assert RUN_RE.fullmatch("run-2-samples-16384-5415000") is None
