#!/bin/bash
#SBATCH --job-name=xray_photons
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/shape_dynamics_slurm_logs/xray_photons_%A_%a.log
#SBATCH --array=0-499%28
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --partition=short

# Stage 1 of the BAHAMAS dataset pipeline: one pyXSIM photon list per
# (simulation, cluster, redshift bin), written to CFG_XRAY.photon_dir.
#
# One array task per (simulation, cluster): 9 redshift bins. Measured on the
# most massive cluster (CDMb 001), whose lowest-z bin is the worst case in the
# whole set: 3.4-5.4 min per bin and 1.8 GB peak RSS, so ~45 min and well inside
# the 8 GB above. A typical cluster has ~1/3 the hot particles and is faster.
#
# Restartable: bins whose .h5 already exists are skipped, so re-running the array
# after a timeout picks up where it left off.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

models=(CDMb SIDM0.1b SIDM0.3b SIDM1b vdSIDMb)

model_id=$(( SLURM_ARRAY_TASK_ID / 100 ))
sim_id=$(( SLURM_ARRAY_TASK_ID % 100 + 1 ))

dm_model=${models[$model_id]}
cluster_id=$(printf "%03d" $sim_id)

echo "Making X-ray photons for model: $dm_model | cluster: $cluster_id"

python3 /home/habjan.e/TNG/cluster_deprojection/dataset/BAHAMAS/xray_photons.py \
    "$cluster_id" "$dm_model"
