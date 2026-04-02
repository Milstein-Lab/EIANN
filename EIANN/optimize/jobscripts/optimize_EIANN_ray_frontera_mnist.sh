#!/bin/bash -l
#SBATCH -J optimize_EIANN_mnist_ray
#SBATCH -o /scratch2/11358/yashchennawar5555/logs/EIANN/optimize_EIANN_mnist_ray.%j.o
#SBATCH -e /scratch2/11358/yashchennawar5555/logs/EIANN/optimize_EIANN_mnist_ray.%j.e
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=rtx
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -euo pipefail
set -x

mkdir -p $SCRATCH/logs/EIANN
mkdir -p $SCRATCH/data/EIANN

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_START_METHOD=thread

module purge
module load cuda/12.2
module load intel/23.1.0

source /work2/11358/yashchennawar5555/frontera/miniconda3/etc/profile.d/conda.sh
conda activate eiann7
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd $HOME/EIANN/EIANN
DEVICE="${2:-cpu}"

export RAY_TMPDIR="$SCRATCH/ray/${SLURM_JOB_ID}"
mkdir -p $RAY_TMPDIR

ray start --head --port=6379 \
  --num-cpus=8 --num-gpus=4 \
  --temp-dir "$RAY_TMPDIR" --disable-usage-stats

cleanup_ray() {
  set +e
  stop_output=$(ray stop --force 2>&1) || true
  echo "$stop_output" | grep -E "Stopped all [0-9]+ Ray processes|No active Ray processes" || true
}
trap cleanup_ray EXIT

python -m nested.optimize --config-file-path=$1 \
  --output-dir=$SCRATCH/data/EIANN --framework=ray --disp \
  --pop_size=4 --max_iter=2 --path_length=2 --num_cpus=1 --num_gpus=0.5 --device=$DEVICE

# num models per generation = pop_size * num_instances (5 seeds) -> how many models evaluated in parallel
# num generations = max_iter * path_length
# total model evals = num models per generation * num generations
# Ray workers = min(floor(total_gpus/num_gpus), floor(total_cpus/num_cpus))
# must match num_cpus, num_gpus in ray start to the SBATCH lines

# -----------------------------------------------------

# cd $HOME/EIANN/EIANN/optimize/jobscripts
# sbatch optimize_EIANN_ray_frontera_mnist.sh optimize/optimize_config/mnist/20231129_nested_optimize_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G.yaml cuda

# See logs:
# cd $SCRATCH/logs/EIANN

# Interactive node:
# idev -p rtx-dev -N 1 -n 1 -t 02:00:00
# export I_MPI_FABRICS=shm

# -----------------------------------------------------

# van_bp_2_hidden: 
# pop_size=4, max_iter=2, path_length=2

# ray (single): 
# nodes=1, ntasks=1, cpus-per-task=8
# ray start num-cpus=8 num-gpus=4
# python num_cpus=1 num_gpus=0.5
#   - 7626472: 4 generations took 665.22 s
#   - 7626649: 4 generations took 743.14 s (with gpu)

# ray (multi):
# nodes=3, ntasks-per-node=1, cpus-per-task=16
#   - 7626603: 4 generations took 256.53 s
#   - 7626650: 4 generations took 259.73 s (with explicit gpu)

# mpi: 
# nodes=1, ntasks=21
#   - 7626474: 4 generations took 233.39 s
#   - 7626648: 4 generations took 234.63 s (with explicit cpu)


# TODO: ray should be faster