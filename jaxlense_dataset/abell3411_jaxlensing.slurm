#!/bin/bash
#SBATCH --job-name=a3411_jaxlense
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/jax_lense_slurm_logs/a3411_%j.log
#SBATCH --time=2:00:00
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

module load cuda/12.3.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax_lense

# Same CUDA environment as train_score_for_clusters.slurm. The +cuda111
# jaxlib wheel needs the CUDA 11 runtime libs that live in the jax_lense
# conda env; ptxas + libdevice come from the cluster's CUDA 12.3 module.
# LD_LIBRARY_PATH must NOT include $CUDA_HOME/lib64 (that would shadow the
# CUDA 11 libs).
unset LD_LIBRARY_PATH
unset XLA_FLAGS
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export PATH="$CUDA_HOME/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

nvidia-smi
python -c "import jax; print('JAX', jax.__version__, 'devices:', jax.devices())"

python3 /home/habjan.e/TNG/cluster_deprojection/jaxlense_dataset/abell3411_jaxlensing.py
