#!/bin/bash
#SBATCH --job-name=xray_photons_flamingo
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/shape_dynamics_slurm_logs/xray_photons_flamingo_%A_%a.log
#SBATCH --array=0-999%28
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --partition=short

# Stage 1 of the FLAMINGO dataset pipeline: one pyXSIM photon list per
# (run, cluster), written to CFG_XRAY.photon_dir (xray_list_files/FLAMINGO).
#
# One array task per (run, cluster): 10 runs x 250 clusters = 2500 tasks, each
# making CFG_DATA.n_projections = 1 list at that sample's exact z_lens from
# config.sample_z_lens. NoCooling is not in the list (no stars, so no BCG).
#
# Explorer caps array indices at 1000 (MaxArraySize = 1001), so the 2500 tasks
# go in as three submissions, with OFFSET added to the array index:
#   sbatch --export=ALL,OFFSET=0    xray_photons.slurm                # tasks    0-999
#   sbatch --export=ALL,OFFSET=1000 xray_photons.slurm                # tasks 1000-1999
#   sbatch --export=ALL,OFFSET=2000 --array=0-499%28 xray_photons.slurm  # 2000-2499
# The three can be queued at once; each throttles itself to 28 running.
#
# Restartable: lists whose .h5 already exists *and opens cleanly* are skipped,
# so re-running the array after a timeout picks up where it left off.
#
# NOTE: the filenames and redshifts depend on CFG_LENS.z_lens_seed,
# z_lens_min/max and CFG_DATA.n_projections. Change any of those and every list
# must be regenerated; build_flamingo_dataset.py checks for the mismatch.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

# `conda activate` leaves the shared Anaconda ahead of the env in PATH, so a
# bare `python3` has no soxs/pyxsim/yt. Call the env's interpreter directly.
PYTHON="$CONDA_PREFIX/bin/python"

# Must match config.FLAMINGO_SIMULATIONS (same order).
models=(L1_m9 fgas+2sigma fgas-2sigma Jet Mstar-1sigma Planck PlanckNu0p24Fix PlanckNu0p24Var PlanckNu0p48Fix L2p8_m9)

task_id=$(( SLURM_ARRAY_TASK_ID + ${OFFSET:-0} ))
if (( task_id >= ${#models[@]} * 250 )); then
    echo "task $task_id is past the last (run, cluster); nothing to do"
    exit 0
fi

model_id=$(( task_id / 250 ))
sim_id=$(( task_id % 250 ))                      # cluster_000..cluster_249

run=${models[$model_id]}
cluster_id=$(printf "%03d" $sim_id)

echo "Making X-ray photons for run: $run | cluster: $cluster_id"

"$PYTHON" /home/habjan.e/TNG/cluster_deprojection/dataset/FLAMINGO/xray_photons.py \
    "$cluster_id" "$run"
