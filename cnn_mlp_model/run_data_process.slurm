#!/bin/bash
#SBATCH --job-name=cnnmlp_data_process
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/cnnmlp_slurm_logs/data_process_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0:30:00
#SBATCH --mem=15G
#SBATCH --partition=short

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source activate cl_dyn

python3 /home/habjan.e/TNG/cluster_deprojection/cnn_mlp_model/make_cnn_mlp_data.py