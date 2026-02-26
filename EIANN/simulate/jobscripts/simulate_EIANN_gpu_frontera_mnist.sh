#!/bin/bash -l
#SBATCH -J eiann_gpu_mnist
#SBATCH -o /scratch2/11358/yashchennawar5555/logs/EIANN/eiann_gpu_mnist.%j.o
#SBATCH -e /scratch2/11358/yashchennawar5555/logs/EIANN/eiann_gpu_mnist.%j.e
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=rtx
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

mkdir -p $SCRATCH/logs/EIANN
mkdir -p $SCRATCH/data/EIANN

export OMP_NUM_THREADS=4
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

cd $HOME/EIANN

python EIANN/simulate/run_EIANN_mnist.py \
  --network-config-file-name=20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml \
  --data-dir=$SCRATCH/data/EIANN --network-seed=66049 --device=cuda --debug


# cd $HOME/EIANN/EIANN/simulate/jobscripts
# sbatch simulate_EIANN_gpu_frontera_mnist.sh

# See logs:
# cd $SCRATCH/logs/EIANN

# See progress:
# watch -n 1 squeue -u $USER

# Request interactive RTX node:
# idev -p rtx -N 1 -n 1 -t 00:30:00

