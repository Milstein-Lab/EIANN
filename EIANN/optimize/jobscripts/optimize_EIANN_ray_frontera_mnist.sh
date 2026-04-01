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

export RAY_TMPDIR="$SCRATCH/ray/${SLURM_JOB_ID}"
mkdir -p $RAY_TMPDIR

ray start --head --port=6379 \
  --num-cpus=8 --num-gpus=4 \
  --temp-dir "$RAY_TMPDIR" --disable-usage-stats

cleanup_ray() {
  set +e
  ray stop --force || true
}
trap cleanup_ray EXIT

python -m nested.optimize --config-file-path=$1 \
  --output-dir=$SCRATCH/data/EIANN --framework=ray --disp \
  --pop_size=4 --max_iter=2 --path_length=2 --num_gpus=0.5 --num_cpus=1

# num models tested = pop_size * num_seeds (5)
# num generations = max_iter * path_length
# Ray workers = min(floor(total_gpus/num_gpus), floor(total_cpus/num_cpus))
# must match num_cpus, num_gpus in ray start to the SBATCH lines

# cd $HOME/EIANN/EIANN/optimize/jobscripts
# sbatch optimize_EIANN_ray_frontera_mnist.sh optimize/optimize_config/mnist/20231129_nested_optimize_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G.yaml

# See logs:
# cd $SCRATCH/logs/EIANN

# Interactive node:
# idev -p rtx-dev -N 1 -n 1 -t 02:00:00
# export I_MPI_FABRICS=shm



# -----------------------------------------------------
# van_bp_0_hidden: 
# ray: 
#   - 4 generations took 232.09 s (7613482)
#   - 45 generations took 8466.89 s (7613450)
# mpi: 
#   - 4 generations took 458.27 s (7613483)

# van_bp_2_hidden:
# ray:
#   - 4 generations took 680.48 s (7626273)
# mpi:
#   - 4 generations took 1110.16 s

