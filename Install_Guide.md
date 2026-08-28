# MatterGen Installation and Training on Frontier

Run all commands from the top level of a MatterGen checkout on a Frontier
login node. Choose any checkout location visible from Frontier:

```bash
git clone <repository-url> mattergen-ornl
cd mattergen-ornl
```

For an existing checkout, simply change into its top-level directory. The
installer resolves its location from the script itself, and the batch job uses
Slurm's submission directory. No absolute checkout path is embedded in either
workflow.

## 1. Build the Frontier environment

Create a clean environment and its relocatable archive:

```bash
bash installation_scripts/setup_mattergen_env_frontier_rocm711.sh --recreate
```

`--recreate` removes an existing environment at the target path before
rebuilding it. Use it after changing package pins or when repairing a
contaminated environment.

The default outputs are:

```text
Environment: /lustre/orion/lrn070/proj-shared/$USER/envs/mattergen-rocm711
Archive:     /lustre/orion/lrn070/proj-shared/$USER/envs/mattergen-rocm711.tar.gz
```

The installer creates Python 3.12 with the tested PyTorch 2.10/ROCm 7.1.1,
PyG, ADIOS2, and MatterGen dependency versions. It verifies the compiled PyG
extensions, the `emmet-core`/`pymatgen` compatibility pins, and local MatterGen
imports before writing the archive. GPU verification is deferred to the batch
job when installation occurs on a login node.

To use another project-Lustre location:

```bash
bash installation_scripts/setup_mattergen_env_frontier_rocm711.sh \
    --recreate \
    --env-path /lustre/orion/<project>/proj-shared/<user>/envs/mattergen-rocm711 \
    --archive-path /lustre/orion/<project>/proj-shared/<user>/envs/mattergen-rocm711.tar.gz
```

The environment must be on a filesystem visible to every allocated node. Do
not place the source archive in a login-node-only temporary directory.

## 2. Submit the training job

For members of project `lrn070`, using the default environment paths and the
default 64-node allocation:

```bash
sbatch JamieTest.sh
```

`JamieTest.sh` requests eight ranks and eight GPUs per node. The default is 64
nodes, or 512 distributed ranks. It broadcasts the packed environment once to
every node, extracts it onto node-local NVMe, and trains using the code in the
submitted checkout.

The launcher uses RCCL's socket transport over Frontier's Slingshot interfaces
(`hsn0` through `hsn3`). This is the configuration that successfully completed
DDP initialization and forward/backward training during diagnosis.

For a smaller test, override the node count at submission time:

```bash
sbatch -N 2 JamieTest.sh
```

Command-line `sbatch` options override the corresponding `#SBATCH` lines. For
example, the time limit can also be changed:

```bash
sbatch -N 2 -t 00:30:00 JamieTest.sh
```

When using custom environment paths, pass them explicitly:

```bash
sbatch \
    --export=ALL,MATTERGEN_ENV_PATH=/lustre/orion/<project>/proj-shared/<user>/envs/mattergen-rocm711,MATTERGEN_ENV_ARCHIVE=/lustre/orion/<project>/proj-shared/<user>/envs/mattergen-rocm711.tar.gz \
    JamieTest.sh
```

The checked-in launcher charges project `LRN070`. A user from another project
can override the account without editing the script:

```bash
sbatch -A <project> JamieTest.sh
```

Users outside `lrn070` should also use `--env-path`, `--archive-path`, and the
corresponding submission variables shown above. They must select a dataset path
their project can access if the configured OMat24 dataset is unavailable.

## 3. Monitor the job

Record the job ID printed by `sbatch`, then check its state:

```bash
squeue -j <JOBID>
```

Follow the main launcher output:

```bash
tail -f JamieTest-<JOBID>.out
```

Follow rank zero's training output:

```bash
tail -f jobOutputs/JamieTest-<JOBID>/slurm-rank-0-*.out
```

Follow rank zero's diagnostic stages:

```bash
tail -f jobOutputs/JamieTest-<JOBID>/trace-rank-0-*.log
```

Summarize all rank logs after or during a run:

```bash
python analyze_jamie_test.py jobOutputs/JamieTest-<JOBID>
```

Cancel a job when necessary:

```bash
scancel <JOBID>
```

## 4. Outputs and checkpoints

Rank-local diagnostic files are written to:

```text
jobOutputs/JamieTest-<JOBID>/
```

Training outputs and checkpoints are written to:

```text
outputs/JamieTest-<JOBID>/
outputs/JamieTest-<JOBID>/checkpoints/
```

The current checkpoint configuration saves from rank zero:

- `last.ckpt` every 500 distributed optimizer steps, overwriting the previous
  `last.ckpt`.
- `last.ckpt` again at the end of every epoch.
- The best validation checkpoint at epoch end, retaining one best checkpoint
  because `save_top_k` is 1.

The default diagnostic allocation has a ten-minute time limit. A short run may
finish or time out before reaching the first 500-step checkpoint.
