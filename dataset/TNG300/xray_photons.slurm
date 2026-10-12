#!/bin/bash
#SBATCH --job-name=xray_photons_tng300
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/shape_dynamics_slurm_logs/xray_photons_tng300_%A_%a.log
#SBATCH --array=0-299%28
#SBATCH --cpus-per-task=1
#SBATCH --time=21:00:00
#SBATCH --mem=20G
#SBATCH --partition=short

# Stage 1 of the TNG300 dataset pipeline: pyXSIM photon lists, one per
# (resolution, halo, projection), written to CFG_XRAY.photon_dir
# (xray_list_files/TNG300).
#
# One array task per (resolution, halo): 3 x 100 = 300 tasks, index ->
# (SIMS[i / 100], halo i % 100). Each task makes config.n_projections_for(sim,
# halo) lists -- 8, or 9 for the 100 halos that bring the suite to 2500 -- each
# at that sample's exact z_lens from config.sample_z_lens.
#
# Needs the halo's gas (with X-ray fields), stars and bh cutouts; a task whose
# cutouts are not ready stops with a message and can simply be re-run.
#
# Restartable: lists whose .h5 already exists *and opens cleanly* are skipped.
#
# Resources = upper bound for the largest halo x 1.5. Measured on TNG300-1
# halo 0 (2026-10-12, partial run on a shared login node):
#   load + yt dataset  ~2 min; 30.1M hot gas cells within 2 R200
#   one photon list    ~75-90 min (pyXSIM at ~5600-6800 cells/s); upper 90 min
#   peak RSS           13.5 GB (steady through the photon loop)
#   halo 0, 9 lists    9 x 90 min + 2 min = ~13.6 h  -> x 1.5 = ~21 h (--time)
#   memory             13.5 GB                       -> x 1.5 = 20 GB (--mem)
# Everything else is cheaper: hot-cell counts (and so time per list) drop
# ~8x for TNG300-2 and ~64x for TNG300-3, and most TNG300-1 halos are several
# times smaller than halo 0 (halo 99 has 5.7M gas cells against 62M).
# Expected array wall time: a few hours for TNG300-2/-3 tasks; TNG300-1 tasks
# from ~1 h (small halos) to ~14 h (halo 0).

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn
# `conda activate` leaves the shared Anaconda ahead of the env in PATH, so a
# bare `python3` has no soxs/pyxsim/yt. Call the env's interpreter directly.
PYTHON="$CONDA_PREFIX/bin/python"

# Must match config.TNG300_SIMULATIONS (same order).
SIMS=(TNG300-1 TNG300-2 TNG300-3)
N_HALOS=100

SIM=${SIMS[$(( SLURM_ARRAY_TASK_ID / N_HALOS ))]}
HALO_ID=$(printf "%03d" $(( SLURM_ARRAY_TASK_ID % N_HALOS )))
if [[ -z "${SIM}" ]]; then
  echo "[error] array index ${SLURM_ARRAY_TASK_ID} is past the last (resolution, halo)"
  exit 2
fi

echo "Making X-ray photons for: $SIM | halo: $HALO_ID"
echo "[estimate] worst case (TNG300-1 halo 0): ~13.6 h, 13.5 GB; limits 21 h / 20 GB (1.5x)"

"$PYTHON" /home/habjan.e/TNG/cluster_deprojection/dataset/TNG300/xray_photons.py \
    "$HALO_ID" "$SIM"
