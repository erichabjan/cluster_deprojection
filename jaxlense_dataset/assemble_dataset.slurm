#!/bin/bash
#SBATCH --job-name=jaxlense_assemble
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/jax_lense_slurm_logs/assemble_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --partition=short
#
# Override the final resolutions via env vars (defaults: 16, 16):
#   FINAL_IMG_RES=32 FINAL_CUBE_RES=32 sbatch assemble_dataset.slurm
# (lens_recon_resolution must be divisible by FINAL_IMG_RES;
#  interim_cube_resolution must be divisible by FINAL_CUBE_RES)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

python3 /home/habjan.e/TNG/cluster_deprojection/jaxlense_dataset/assemble_dataset.py \
  --final-image-resolution ${FINAL_IMG_RES:-16} \
  --final-cube-resolution ${FINAL_CUBE_RES:-16}
