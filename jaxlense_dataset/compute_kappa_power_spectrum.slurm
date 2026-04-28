#!/bin/bash
#SBATCH --job-name=jaxlense_ps
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/jax_lense_slurm_logs/ps_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --partition=short

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

python3 /home/habjan.e/TNG/cluster_deprojection/jaxlense_dataset/compute_kappa_power_spectrum.py
