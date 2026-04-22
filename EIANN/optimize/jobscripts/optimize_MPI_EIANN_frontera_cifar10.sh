#!/bin/bash -l
#SBATCH -J optimize_EIANN_cifar10_MPI
#SBATCH -o /scratch2/11358/yashchennawar5555/logs/EIANN/optimize_EIANN_cifar10_MPI.%j.o
#SBATCH -e /scratch2/11358/yashchennawar5555/logs/EIANN/optimize_EIANN_cifar10_MPI.%j.e
#SBATCH -p development
#SBATCH --nodes=2
#SBATCH --ntasks=46
#SBATCH --time=2:00:00
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -euo pipefail
set -x

mkdir -p "$SCRATCH/logs/EIANN"
mkdir -p "$SCRATCH/data/EIANN"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source /work2/11358/yashchennawar5555/frontera/miniconda3/etc/profile.d/conda.sh
conda activate eiann7
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd "$HOME/EIANN/EIANN"

DEVICE="${2:-cpu}"

export MPI4PY_RC_RECV_MPROBE=false

ibrun -n 46 python -m mpi4py.futures -m nested.optimize --config-file-path=$1 \
  --output-dir=$SCRATCH/data/EIANN --framework=mpi --disp \
  --pop_size=9 --max_iter=2 --path_length=2 --device="$DEVICE"

# -n: num procs = 1 master + pop_size * num_seeds (5)
# num generations = max_iter * path_length
# python -n must match ntasks in SBATCH lines

# cd $HOME/EIANN/EIANN/optimize/jobscripts 
# sbatch optimize_MPI_EIANN_frontera_cifar10.sh optimize/optimize_config/cifar10/20250814_nested_optimize_EIANN_2_hidden_convnet_cifar10_van_bp_relu_SGD_CE_config_G_learned_bias.yaml cpu

# See logs:
# cd $SCRATCH/logs/EIANN