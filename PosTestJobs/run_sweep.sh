#!/bin/bash
# Sequentially submit the position-loss timestep sweep from a Frontier login
# node. The debug QOS permits only one job in any state, so --wait is
# deliberate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/PosTestJob.sh"
DRY_RUN=0
START_AT=0
STOP_AT=3

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

[[ "${START_AT}" =~ ^[0-3]$ ]] || {
    echo "ERROR: --start-at must be an integer from 0 through 3" >&2
    exit 2
}
[[ "${STOP_AT}" =~ ^[0-3]$ ]] || {
    echo "ERROR: --stop-at must be an integer from 0 through 3" >&2
    exit 2
}
(( START_AT <= STOP_AT )) || {
    echo "ERROR: --start-at cannot be greater than --stop-at" >&2
    exit 2
}

t_exponents=(0 2 4 8)
t_maximums=(1.0 0.25 0.0625 0.00390625)
sample_count=1024
node_count=1

cd "${REPO_ROOT}"
for (( run_index=START_AT; run_index<=STOP_AT; run_index++ )); do
    t_exponent="${t_exponents[run_index]}"
    t_maximum="${t_maximums[run_index]}"
    job_name="PosT${t_exponent}"
    command=(
        sbatch
        --wait
        --nodes="${node_count}"
        --job-name="${job_name}"
        --export="ALL,MATTERGEN_REPO_ROOT=${REPO_ROOT},POS_TEST_RUN_INDEX=${run_index},POS_TEST_SAMPLES=${sample_count},POS_TEST_T_EXP=${t_exponent},POS_TEST_MAX_T=${t_maximum}"
        "${JOB_SCRIPT}"
    )

    printf 'run=%d samples=%d nodes=%d ranks=%d t_range=[0,2^-%d]=[0,%s] command=' \
        "${run_index}" "${sample_count}" "${node_count}" "$((node_count * 8))" \
        "${t_exponent}" "${t_maximum}"
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
