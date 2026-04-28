#!/bin/bash
#SBATCH --job-name=jaxlense_dlposterior
#SBATCH --output=/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/jax_lense_slurm_logs/dlposterior_%A_%a.log
#SBATCH --time=8:00:00
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#
# To parallelize across multiple GPU jobs:
#   sbatch --array=0-9 run_dlposterior.slurm
# (the script splits the manifest evenly across the array tasks below)
#
# To run a specific subrange in a single job:
#   START=0 STOP=50 sbatch run_dlposterior.slurm

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

if [[ -n "${SLURM_ARRAY_TASK_ID}" && -n "${SLURM_ARRAY_TASK_COUNT}" ]]; then
  N_TOTAL=$(python3 -c "import numpy as np, os; m=np.load(os.path.join('/projects/mccleary_group/habjan.e/TNG/Data/jaxlense_dataset/','manifest.npz')); print(int(m['n_total']))")
  CHUNK=$(( (N_TOTAL + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
  START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
  STOP=$(( START + CHUNK ))
  echo "array task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT: range [$START, $STOP) of $N_TOTAL"
  EXTRA_ARGS="--start $START --stop $STOP"
else
  EXTRA_ARGS="--start ${START:-0} --stop ${STOP:--1}"
fi

nvidia-smi
python -c "import jax; print('JAX', jax.__version__, 'devices:', jax.devices())"

python3 /home/habjan.e/TNG/cluster_deprojection/jaxlense_dataset/run_dlposterior.py $EXTRA_ARGS
