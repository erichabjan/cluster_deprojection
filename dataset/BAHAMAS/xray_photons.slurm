#!/bin/bash
#SBATCH --job-name=xray_photons
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/shape_dynamics_slurm_logs/xray_photons_%A_%a.log
#SBATCH --array=0-499%28
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --partition=short

# Stage 1 of the BAHAMAS dataset pipeline: one pyXSIM photon list per
# (simulation, cluster, projection), written to CFG_XRAY.photon_dir.
#
# One array task per (simulation, cluster): CFG_DATA.n_projections = 5 lists,
# each at that sample's exact z_lens from config.sample_z_lens. That replaced a
# 9-bin redshift binning, so it is both cheaper (5 lists per cluster, not 9) and
# exact -- the X-ray map and the lensing geometry are at the same distance.
#
# Measured on the most massive cluster (CDMb 001), whose lowest-z list is the
# worst case in the whole set: 3.4-5.4 min each and 1.8 GB peak RSS, so ~27 min
# and well inside the 8 GB above. A typical cluster has ~1/3 the hot particles
# and is faster (CDMb 060 took 90 s for one list).
#
# Restartable: projections whose .h5 already exists *and opens cleanly* are
# skipped, so re-running the array after a timeout picks up where it left off
# without inheriting a half-written file.
#
# NOTE: the filenames and redshifts depend on CFG_LENS.z_lens_seed,
# z_lens_min/max and CFG_DATA.n_projections. Change any of those and every list
# must be regenerated; build_bahamas_dataset.py checks for the mismatch and
# refuses to start rather than quietly using maps at the wrong distance.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

# `conda activate` swaps the env into the slot the base env already occupied in
# PATH, which on this cluster sits *behind* /shared/EL9/.../anaconda3/2024.06/bin.
# A bare `python3` therefore resolves to the shared Anaconda 3.12, which has no
# soxs/pyxsim/yt -- the whole 9083975 array died on `import soxs` for this
# reason. Call the env's interpreter by absolute path instead.
PYTHON="$CONDA_PREFIX/bin/python"

models=(CDMb SIDM0.1b SIDM0.3b SIDM1b vdSIDMb)

model_id=$(( SLURM_ARRAY_TASK_ID / 100 ))
sim_id=$(( SLURM_ARRAY_TASK_ID % 100 + 1 ))

dm_model=${models[$model_id]}
cluster_id=$(printf "%03d" $sim_id)

echo "Making X-ray photons for model: $dm_model | cluster: $cluster_id"

"$PYTHON" /home/habjan.e/TNG/cluster_deprojection/dataset/BAHAMAS/xray_photons.py \
    "$cluster_id" "$dm_model"
