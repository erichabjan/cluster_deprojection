#!/bin/bash
#SBATCH --job-name=cd_posterior_sampler
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/cond_diff_slurm_logs/cd_posterior_%A_%a.log
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5

# One array task per (model resolution N, TNG300 simulation), 100 clusters
# each, projection 0:
#   task  0 1 2 -> N=16, TNG300-1 TNG300-2 TNG300-3
#   task  3 4 5 -> N=32, TNG300-1 TNG300-2 TNG300-3
#   sbatch run_cond_diff_sampler.slurm                  # all six
#   sbatch --array=3 run_cond_diff_sampler.slurm        # just N=32, TNG300-1
# A task that hits the walltime can be resubmitted as is: clusters already
# written are skipped.
RESOLUTIONS=(16 32)
SIMULATIONS=(TNG300-1 TNG300-2 TNG300-3)
TASK=${SLURM_ARRAY_TASK_ID:?submit with sbatch (array job)}
RES=${RESOLUTIONS[$((TASK / 3))]}
SIM=${SIMULATIONS[$((TASK % 3))]}
echo "Task $TASK: N=$RES, simulation=$SIM"

# Load CUDA module
module load cuda/12.3.0

# Activate the conda environment with JAX+CUDA support
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cl_dyn

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
"$CONDA_PREFIX/bin/python" /home/habjan.e/TNG/cluster_deprojection/conditional_diffusion_model/cond_diff_sampler.py \
    --resolution "$RES" --simulation "$SIM"
