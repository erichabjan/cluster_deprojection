#!/bin/bash
#SBATCH --job-name=tng300_dataset
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/shape_dynamics_slurm_logs/build_tng300_dataset_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-06:00:00
#SBATCH --mem=210G
#SBATCH --partition=short

# Stage 2: needs the photon lists from TNG300/xray_photons.slurm to already be
# in CFG_XRAY.photon_dir; the build refuses to start if any are missing.
#
# 2500 samples (3 resolutions x 100 halos x 8-9 projections), the same count
# as the BAHAMAS and FLAMINGO builds.
#
# Resources = upper-bound estimate x 1.5. The build itself was NOT timed (the
# one-sample test was skipped); the estimates come from the TNG300-1 halo 0
# load (measured: 17.5 GB and ~4.5 min to load all four cutouts in float64)
# and the FLAMINGO/BAHAMAS builds (~1 min per sample of X-ray observation):
#   memory per worker  ~2 x the 17.5 GB load for halo 0 (rotated float64
#                      copies of each component) = ~35 GB upper
#   workers            4: the job list is cluster-major, so all 4 workers can
#                      be on TNG300-1 halo 0 at once -> 4 x 35 GB = 140 GB
#                      -> x 1.5 = 210 GB (--mem; one of the ~250 GB nodes)
#   time per sample    upper ~3 min TNG300-1 (most halos are far smaller than
#                      halo 0), ~1.5 min TNG300-2, ~1 min TNG300-3
#   total              833 x (3 + 1.5 + 1) min = ~76 core-h / 4 workers = ~19 h
#                      -> x 1.5 = ~30 h (--time)
# If the log shows memory well under the limit after the TNG300-1 halo 0
# samples, a re-run can use more workers (raise --cpus-per-task, keeping
# --mem >= 52 GB x workers).
#
# Set CFG_XRAY.enabled = False for a dynamics/lensing-only build, which runs in
# minutes and needs none of the photon lists.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

# `conda activate` leaves the shared Anaconda ahead of the env in PATH, so a
# bare `python3` has no soxs/pyxsim/yt. Call the env's interpreter directly.
PYTHON="$CONDA_PREFIX/bin/python"

echo "[estimate] ~19 h on 4 workers, <= ~35 GB per worker; limits 30 h / 210 GB (1.5x)"

"$PYTHON" /home/habjan.e/TNG/cluster_deprojection/dataset/TNG300/build_tng300_dataset.py
