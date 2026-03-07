#!/bin/bash
#SBATCH --job-name=cond_diff_data_process
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/cond_diff_slurm_logs/data_process_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH --mem=15G
#SBATCH --partition=short

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source activate cl_dyn

python3 /home/habjan.e/TNG/cluster_deprojection/conditional_diffusion_model/conditional_diffusion_data.py