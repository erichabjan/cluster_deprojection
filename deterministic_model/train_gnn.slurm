#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/gnn_slurm_logs/gnn_%j.log
#SBATCH --time=8:00:00
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Load CUDA module
module load cuda/12.3.0

# Activate the conda environment with JAX+CUDA support
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

# WandB login and settings
wandb login
export WANDB_DIR=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/wandb

# Unset any previously set CUDA-related environment variables
unset LD_LIBRARY_PATH
unset XLA_FLAGS

# Set correct CUDA environmental variables
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

# Print debug information
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
nvidia-smi  # Show GPU state before running

python -c "import jax; print('JAX version:', jax.__version__); print('Available devices:', jax.devices()); print('Using GPU:', any(d.platform == 'gpu' for d in jax.devices()))"

# Run your JAX script
python3 /home/habjan.e/TNG/TNG_cluster_dynamics/GNN/train_gnn.py