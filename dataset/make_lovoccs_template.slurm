#!/bin/bash
#SBATCH --job-name=lovoccs_nz_template
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/shape_dynamics_slurm_logs/lovoccs_template_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --partition=short

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source activate cl_dyn

python3 /home/habjan.e/TNG/cluster_deprojection/dataset/build_lovoccs_redshift_template.py
