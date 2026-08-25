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
# The X-ray mosaic dominates the cost at ~2-4.5 min per sample (9 ACIS-I
# pointings, 9 exposure maps, then the reprojection), so 2500 samples is
# ~85-190 core-hours: 6-12 h on 16 workers, against the 2-day partition limit.
#
# Memory: a worker peaked at 3.7 GB on the worst case (most massive cluster,
# lowest-z photon list, so the fattest event list), hence 16 x 3.7 = 59 GB of
# genuine demand. 62G left no headroom at all; 120G lands on one of the 257 GB
# nodes with room to spare. Drop to --cpus-per-task=12 --mem=62G instead if the
# queue for a large-memory node is long.
#
# Set CFG_XRAY.enabled = False for a dynamics/lensing-only build, which runs in
# minutes and needs none of this.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source activate cl_dyn

python3 /home/habjan.e/TNG/cluster_deprojection/dataset/BAHAMAS/build_bahamas_dataset.py
