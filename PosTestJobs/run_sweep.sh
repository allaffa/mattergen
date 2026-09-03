#!/bin/bash
# Sequentially submit the position-loss scaling sweep from a Frontier login
# node. The debug QOS permits only one job in any state, so --wait is
# deliberate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/PosTestJob.sh"
DRY_RUN=0
START_AT=0
STOP_AT=7

usage() {
    echo "Usage: $0 [--dry-run] [--start-at RUN_INDEX] [--stop-at RUN_INDEX]"
}

while (( $# > 0 )); do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --start-at)
            (( $# >= 2 )) || { echo "ERROR: --start-at requires a value" >&2; exit 2; }
            START_AT="$2"
            shift 2
            ;;
        --stop-at)
            (( $# >= 2 )) || { echo "ERROR: --stop-at requires a value" >&2; exit 2; }
            STOP_AT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "${START_AT}" =~ ^[0-7]$ ]] || {
    echo "ERROR: --start-at must be an integer from 0 through 7" >&2
    exit 2
}
[[ "${STOP_AT}" =~ ^[0-7]$ ]] || {
    echo "ERROR: --stop-at must be an integer from 0 through 7" >&2
    exit 2
}
(( START_AT <= STOP_AT )) || {
    echo "ERROR: --start-at cannot be greater than --stop-at" >&2
    exit 2
}

samples=(1024 4096 16384 65536 262144 1048576 4194304 16777216)
nodes=(1 4 16 64 64 64 64 64)

cd "${REPO_ROOT}"
for (( run_index=START_AT; run_index<=STOP_AT; run_index++ )); do
    sample_count="${samples[run_index]}"
    node_count="${nodes[run_index]}"
    job_name="PosN${run_index}-${sample_count}"
    command=(
        sbatch
        --wait
        --nodes="${node_count}"
        --job-name="${job_name}"
        --export="ALL,MATTERGEN_REPO_ROOT=${REPO_ROOT},POS_TEST_RUN_INDEX=${run_index},POS_TEST_SAMPLES=${sample_count}"
        "${JOB_SCRIPT}"
    )

    printf 'run=%d samples=%d nodes=%d ranks=%d command=' \
        "${run_index}" "${sample_count}" "${node_count}" "$((node_count * 8))"
    printf '%q ' "${command[@]}"
    printf '\n'

    if (( DRY_RUN )); then
        continue
    fi

    set +e
    "${command[@]}"
    rc=$?
    set -e
    if (( rc != 0 )); then
        echo "ERROR: run ${run_index} failed; stopping the sweep (rc=${rc})." >&2
        exit "${rc}"
    fi
done

if (( DRY_RUN )); then
    echo "Dry run complete; no jobs were submitted."
else
    python "${SCRIPT_DIR}/summarize_pos_loss.py" --repo-root "${REPO_ROOT}" || true
    echo "All position-loss sweep jobs completed successfully."
fi
