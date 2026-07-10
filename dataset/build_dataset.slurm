#!/bin/bash
#SBATCH --job-name=shape_dynamics_data
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/shape_dynamics_slurm_logs/build_dataset_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --partition=short

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source activate cl_dyn

python3 /home/habjan.e/TNG/cluster_deprojection/dataset/build_shape_dynamics_dataset.py
