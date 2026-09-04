#!/usr/bin/env python3
"""Combine rank-zero position-loss logs from the Frontier timestep sweep."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


RUN_RE = re.compile(r"run-(\d+)-samples-(\d+)-t-exp-(\d+)-(\d+)$")
NUMBER = r"[-+]?(?:nan|inf|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
METRIC_RE = re.compile(
    r"epoch=(\d+)\s+step=(\d+)\s+global_step=(\d+)\s+"
    rf"elapsed_train_seconds=({NUMBER})\s+lr=({NUMBER})\s+"
    rf"loss_train=({NUMBER})\s+pos_train=({NUMBER})\s+"
    rf"cell_train=({NUMBER})\s+atom_train=({NUMBER})",
    re.IGNORECASE,
)
EXPECTED_T_EXPONENTS = (0, 2, 4, 8)


def parse_metric_lines(text: str) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for match in METRIC_RE.finditer(text):
        epoch, epoch_step, global_step = (int(match.group(i)) for i in range(1, 4))
        elapsed, lr, loss, pos, cell, atom = (
            float(match.group(i)) for i in range(4, 10)
        )
        rows.append(
            {
                "epoch": epoch,
                "epoch_step": epoch_step,
                "global_step": global_step,
                "elapsed_train_seconds": elapsed,
                "lr": lr,
                "loss_train": loss,
                "pos_train": pos,
                "cell_train": cell,
                "atom_train": atom,
            }
        )
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, runs: dict[int, list[dict]]) -> None:
    if not runs:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; wrote CSV summaries without a plot")
        return

    figure, axis = plt.subplots(figsize=(9, 6))
    for run_index, rows in sorted(runs.items()):
        axis.plot(
            [row["global_step"] for row in rows],
            [row["pos_train"] for row in rows],
            label=(
                f"max t=2^-{int(rows[0]['t_exponent'])} "
                f"({float(rows[0]['max_t']):.6g})"
            ),
            linewidth=1.4,
        )
    if all(row["pos_train"] > 0 for rows in runs.values() for row in rows):
        axis.set_yscale("log")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Training position loss")
    axis.set_title("OMat timestep-range sweep: position loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    log_root = repo_root / "jobOutputs" / "PosTest"
    output_root = repo_root / "outputs" / "PosTest"
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    runs: dict[int, list[dict]] = defaultdict(list)
    latest_run_dirs: dict[int, tuple[int, int, int, Path]] = {}
    for run_dir in sorted(log_root.glob("run-*-samples-*-t-exp-*-*")):
        match = RUN_RE.fullmatch(run_dir.name)
        if match is None:
            continue
        run_index, sample_count, t_exponent, job_id = map(int, match.groups())
        if not 0 <= run_index < len(EXPECTED_T_EXPONENTS):
            continue
        if t_exponent != EXPECTED_T_EXPONENTS[run_index]:
            continue
        previous = latest_run_dirs.get(run_index)
        if previous is None or job_id > previous[0]:
            latest_run_dirs[run_index] = (
                job_id,
                sample_count,
                t_exponent,
                run_dir,
            )

    for run_index, (job_id, sample_count, t_exponent, run_dir) in sorted(
        latest_run_dirs.items()
    ):
        rank_zero_logs = sorted(run_dir.glob("slurm-rank-0-*.out"))
        for log_path in rank_zero_logs:
            for metric in parse_metric_lines(log_path.read_text(errors="replace")):
                row = {
                    "run_index": run_index,
                    "sample_count": sample_count,
                    "t_exponent": t_exponent,
                    "max_t": 2.0 ** -t_exponent,
                    "nodes": 1,
                    "ranks": 8,
                    "job_id": job_id,
                    **metric,
                    "log_path": str(log_path),
                }
                all_rows.append(row)
                runs[run_index].append(row)

    all_rows.sort(key=lambda row: (row["run_index"], row["job_id"], row["global_step"]))
    metric_fields = [
        "run_index", "sample_count", "t_exponent", "max_t", "nodes", "ranks",
        "job_id", "epoch",
        "epoch_step", "global_step", "elapsed_train_seconds", "lr", "loss_train",
        "pos_train", "cell_train", "atom_train", "log_path",
    ]
    _write_csv(output_root / "pos_metrics.csv", metric_fields, all_rows)

    summary_rows: list[dict] = []
    for run_index, rows in sorted(runs.items()):
        rows.sort(key=lambda row: (row["job_id"], row["global_step"]))
        first = rows[0]
        final = rows[-1]
        finite_rows = [row for row in rows if math.isfinite(row["pos_train"])]
        minimum_pos = (
            min(row["pos_train"] for row in finite_rows) if finite_rows else math.nan
        )
        ratio = (
            final["pos_train"] / first["pos_train"]
            if math.isfinite(first["pos_train"])
            and math.isfinite(final["pos_train"])
            and first["pos_train"] != 0
            else math.nan
        )
        summary_rows.append(
            {
                "run_index": run_index,
                "sample_count": first["sample_count"],
                "t_exponent": first["t_exponent"],
                "max_t": first["max_t"],
                "nodes": first["nodes"],
                "ranks": first["ranks"],
                "job_id": final["job_id"],
                "logged_steps": len(rows),
                "first_pos_train": first["pos_train"],
                "final_pos_train": final["pos_train"],
                "minimum_pos_train": minimum_pos,
                "final_over_first": ratio,
                "final_elapsed_train_seconds": final["elapsed_train_seconds"],
            }
        )
    summary_fields = [
        "run_index", "sample_count", "t_exponent", "max_t", "nodes", "ranks",
        "job_id", "logged_steps",
        "first_pos_train", "final_pos_train", "minimum_pos_train",
        "final_over_first", "final_elapsed_train_seconds",
    ]
    _write_csv(output_root / "pos_summary.csv", summary_fields, summary_rows)
    _write_plot(output_root / "pos_loss.png", runs)

    print(f"Wrote {output_root / 'pos_metrics.csv'} ({len(all_rows)} rows)")
    print(f"Wrote {output_root / 'pos_summary.csv'} ({len(summary_rows)} runs)")
    if (output_root / "pos_loss.png").exists():
        print(f"Wrote {output_root / 'pos_loss.png'}")


if __name__ == "__main__":
    main()
