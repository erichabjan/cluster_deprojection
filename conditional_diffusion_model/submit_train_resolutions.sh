#!/bin/bash
# Submit one training job per cube resolution, each with its own memory.
#
#   bash submit_train_resolutions.sh            # all four: 16 32 64 128
#   bash submit_train_resolutions.sh 16 32      # a subset
#
# Memory is what the pooled BAHAMAS + FLAMINGO set (5000 samples, loaded at
# one N only) needs in host RAM, with headroom: ~55 GB at N = 128 (cubes are
# 8 MB each), ~20 GB or less below. Every job runs on one A100 with the
# #SBATCH defaults in run_train_conditional_diffusion.slurm otherwise.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/cond_diff_slurm_logs
declare -A MEM=([16]=64G [32]=64G [64]=64G [128]=128G)

RESOLUTIONS=("$@")
[ ${#RESOLUTIONS[@]} -eq 0 ] && RESOLUTIONS=(16 32 64 128)

for N in "${RESOLUTIONS[@]}"; do
    if [ -z "${MEM[$N]+x}" ]; then
        echo "unknown resolution $N (expected 16, 32, 64 or 128)" >&2; exit 1
    fi
    sbatch --job-name="cond_diff_N${N}" \
           --mem="${MEM[$N]}" \
           --output="${LOGDIR}/cond_diff_N${N}_%j.log" \
           "${HERE}/run_train_conditional_diffusion.slurm" "$N"
done
