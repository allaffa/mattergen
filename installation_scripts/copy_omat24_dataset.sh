#!/usr/bin/env bash
#SBATCH -A LRN110
#SBATCH -J copy-omat24
#SBATCH -o jobOutputs/copy-omat24-%j.out
#SBATCH -e jobOutputs/copy-omat24-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1

# Copy the OMat24-v2 ADIOS dataset out of the purgeable lrn070 world-shared area and verify it byte-for-byte.
set -euo pipefail

SRC="${SRC:-/lustre/orion/world-shared/lrn070/jyc/frontier/dataset_sc26_restripe/OMat24-v2.bp}"
DST="${DST:-${SLURM_SUBMIT_DIR}/datasets/OMat24-v2.bp}"
NPROC="${NPROC:-28}"

mkdir -p "${DST}"
# Wide striping keeps 4096-rank reads from hot-spotting on a few OSTs, matching the source layout.
lfs setstripe -c -1 -S 1M "${DST}"

cd "${SRC}"
echo "copy start $(date) ${SRC} -> ${DST}"
ls -1 | xargs -P "${NPROC}" -I{} cp -p "{}" "${DST}/{}"
echo "copy done $(date)"

echo "verifying sizes and contents"
ls -1 | xargs -P "${NPROC}" -I{} cmp "{}" "${DST}/{}"
[[ "$(ls -1 | wc -l)" == "$(ls -1 "${DST}" | wc -l)" ]]
du -s --apparent-size --block-size=1 "${SRC}" "${DST}"
echo "VERIFY_OK $(date)"
