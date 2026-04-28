#!/bin/bash
#SBATCH --job-name=jaxlense_score_train
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/jax_lense_slurm_logs/score_train_%j.log
#SBATCH --time=8:00:00
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

module load cuda/12.3.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax_lense

# Runtime libs from conda (cudatoolkit=11.1, cudnn=8.2 — match the +cuda111
# jaxlib wheel). ptxas + libdevice from the cluster's CUDA 12.3 module
# (newer ptxas is backwards-compatible to older sm arches). LD_LIBRARY_PATH
# must NOT include $CUDA_HOME/lib64 — that would shadow the CUDA 11 runtime
# libs the +cuda111 wheel expects.
unset LD_LIBRARY_PATH
unset XLA_FLAGS
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export PATH="$CUDA_HOME/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

nvidia-smi
python -c "import jax; print('JAX', jax.__version__, 'devices:', jax.devices())"

python3 /home/habjan.e/TNG/cluster_deprojection/jaxlense_dataset/train_score_for_clusters.py
