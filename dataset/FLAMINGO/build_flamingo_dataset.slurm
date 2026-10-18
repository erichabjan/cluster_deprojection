#!/bin/bash
#SBATCH --job-name=flamingo_dataset
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/shape_dynamics_slurm_logs/build_flamingo_dataset_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=120G
#SBATCH --partition=short

# Stage 2: needs the photon lists from FLAMINGO/xray_photons.slurm to already be
# in CFG_XRAY.photon_dir; the build refuses to start if any are missing.
#
# 2500 samples (10 runs x 250 clusters x 1 projection), the same count as the
# BAHAMAS build, at a similar ~1 min per sample for the single ACIS-I pointing,
# so a few hours on 16 workers. Resources mirror build_bahamas_dataset.slurm.
#
# The adaptive-SPH density cube adds a ~1 min 128^3 deposit per sample and a
# kNN per cluster (cached under /scratch/habjan.e/cube_intermediates/FLAMINGO;
# one projection per cluster, so the cache only pays off on re-runs). The m9
# runs have BAHAMAS-like particle counts: L1_m9 cluster 100 took 1:50 for the
# whole sample at 1.5 GB. ~80-100 core-h, ~6 h on 16 workers.
#
# Set CFG_XRAY.enabled = False for a dynamics/lensing-only build, which runs in
# minutes and needs none of the photon lists.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SPH_THREADS=1
export NUMBA_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

# `conda activate` leaves the shared Anaconda ahead of the env in PATH, so a
# bare `python3` has no soxs/pyxsim/yt. Call the env's interpreter directly.
PYTHON="$CONDA_PREFIX/bin/python"

"$PYTHON" /home/habjan.e/TNG/cluster_deprojection/dataset/FLAMINGO/build_flamingo_dataset.py
