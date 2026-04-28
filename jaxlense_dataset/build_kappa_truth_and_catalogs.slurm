#!/bin/bash
#SBATCH --job-name=jaxlense_stage1
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/jax_lense_slurm_logs/stage1_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=4:00:00
#SBATCH --mem=24G
#SBATCH --partition=short

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

python3 /home/habjan.e/TNG/cluster_deprojection/jaxlense_dataset/build_kappa_truth_and_catalogs.py
