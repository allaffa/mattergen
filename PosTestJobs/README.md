# Position-loss scaling sweep

This sweep trains on progressively larger prefixes of the full OMat
`trainset`. It does not create or copy dataset files. Every run starts from
the same seed and stops after 600 optimizer steps or 1,800 training seconds,
whichever comes first. Only position loss contributes to gradients; cell and
atomic-number losses remain enabled for diagnostic logging with zero weight.

| Run | Samples | Nodes | GPU ranks |
|---:|---:|---:|---:|
| 0 | 1,024 | 1 | 8 |
| 1 | 4,096 | 4 | 32 |
| 2 | 16,384 | 16 | 128 |
| 3 | 65,536 | 64 | 512 |
| 4 | 262,144 | 64 | 512 |
| 5 | 1,048,576 | 64 | 512 |
| 6 | 4,194,304 | 64 | 512 |
| 7 | 16,777,216 | 64 | 512 |

## Run on Frontier

Run these commands from the repository root. First inspect all submissions:

```bash
bash PosTestJobs/run_sweep.sh --dry-run
```

Then launch the sequential controller so it survives an SSH disconnect:

```bash
nohup bash PosTestJobs/run_sweep.sh \
  > PosTestJobs/sweep-launcher.log 2>&1 < /dev/null &
```

Follow it with:

```bash
tail -f PosTestJobs/sweep-launcher.log
```

The Frontier debug QOS permits only one job in any state and prohibits job
chaining. The controller therefore uses `sbatch --wait` and submits the next
job only after the current one succeeds. It stops on the first failure.

To resume at a particular run after fixing a failure:

```bash
nohup bash PosTestJobs/run_sweep.sh --start-at 3 \
  > PosTestJobs/sweep-launcher-from-3.log 2>&1 < /dev/null &
```

The expected environment archive and OMat path are inherited from
`JamieTest.sh`. Override them before starting the controller when needed:

```bash
export MATTERGEN_ENV_ARCHIVE=/path/to/mattergen-rocm711.tar.gz
export OMAT_DATA_PATH=/path/to/OMat24-v2.bp
```

Per-rank logs are written beneath `jobOutputs/PosTest/`. Resolved configs and
Hydra outputs are written beneath `outputs/PosTest/`. After each successful
job, the summarizer refreshes:

- `outputs/PosTest/pos_metrics.csv`
- `outputs/PosTest/pos_summary.csv`
- `outputs/PosTest/pos_loss.png`

The limits can be overridden for a short smoke test:

```bash
export POS_TEST_MAX_STEPS=2
export POS_TEST_MAX_TRAIN_SECONDS=120
bash PosTestJobs/run_sweep.sh --start-at 0 --stop-at 0
```

Clear those overrides before launching the full sweep:

```bash
unset POS_TEST_MAX_STEPS POS_TEST_MAX_TRAIN_SECONDS
```
