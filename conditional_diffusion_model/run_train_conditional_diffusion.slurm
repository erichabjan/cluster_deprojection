#!/bin/bash
#SBATCH --job-name=cond_diff_train
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/cond_diff_slurm_logs/cond_diff_%j.log
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# One cube resolution per job: sbatch run_train_conditional_diffusion.slurm N
# with N in 16 32 64 128. submit_train_resolutions.sh submits all four, with
# per-N memory and log names; the #SBATCH lines above are the N <= 64 defaults.
RES=${1:?usage: sbatch run_train_conditional_diffusion.slurm <N: 16|32|64|128>}
echo "Cube resolution N=$RES"

# Load CUDA module
module load cuda/12.3.0

# Activate the conda environment with JAX+CUDA support
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

# WandB login and settings
wandb login
export WANDB_DIR=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/cnnmlp_wandb/wandb

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
nvidia-smi

"$CONDA_PREFIX/bin/python" -c "import jax; print('JAX version:', jax.__version__); print('Available devices:', jax.devices()); print('Using GPU:', any(d.platform == 'gpu' for d in jax.devices()))"

# The env's interpreter by absolute path: `conda activate` can leave the shared
# Anaconda ahead of the env in PATH, so a bare python3 may not be cl_dyn.
"$CONDA_PREFIX/bin/python" /home/habjan.e/TNG/cluster_deprojection/conditional_diffusion_model/run_conditional_diffusion.py --resolution "$RES"