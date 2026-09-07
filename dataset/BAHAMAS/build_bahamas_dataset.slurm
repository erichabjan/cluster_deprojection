#!/bin/bash
#SBATCH --job-name=bahamas_dataset
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/shape_dynamics_slurm_logs/build_bahamas_dataset_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=120G
#SBATCH --partition=short

# Stage 2: needs the photon lists from xray_photons.slurm to already be in
# CFG_XRAY.photon_dir; the build refuses to start if any are missing.
#
# The X-ray observation dominates the cost. It is now a single ACIS-I pointing
# per sample rather than a 3x3 mosaic (LoVoCCS's X-ray coverage is archival
# Chandra/XMM, not a designed survey), which measured at ~52 s per sample on a
# mid-sized cluster against the mosaic's 2-4.5 min. 2500 samples is then
# ~36 core-hours, a few hours on 16 workers.
#
# Memory: the old 3x3 mosaic peaked at 3.7 GB per worker on the worst case
# (most massive cluster, lowest-z photon list, fattest event list); one pointing
# is well under that. 120G lands on one of the 257 GB nodes with room to spare.
# Drop to --cpus-per-task=12 --mem=62G if the queue for a large-memory node
# is long.
#
# Each sample also solves R500c and sums L_X500c from the particles (~2 s,
# memoised per worker per cluster) so the sample can be checked against
# LoVoCCS's L_X > 1e44 erg/s selection after the fact.
#
# Set CFG_XRAY.enabled = False for a dynamics/lensing-only build, which runs in
# minutes and needs none of this.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

# `conda activate` swaps the env into the slot the base env already occupied in
# PATH, which on this cluster sits *behind* /shared/EL9/.../anaconda3/2024.06/bin.
# A bare `python3` therefore resolves to the shared Anaconda 3.12, which has no
# soxs/pyxsim/yt. Call the env's interpreter by absolute path instead.
PYTHON="$CONDA_PREFIX/bin/python"

"$PYTHON" /home/habjan.e/TNG/cluster_deprojection/dataset/BAHAMAS/build_bahamas_dataset.py
