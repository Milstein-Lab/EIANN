#!/bin/bash -l
#SBATCH -J eiann_gpu_mnist_multi_seed
#SBATCH -o /ocean/projects/bio240068p/chennawa/logs/EIANN/eiann_gpu_mnist_multi_seed.%j.o
#SBATCH -e /ocean/projects/bio240068p/chennawa/logs/EIANN/eiann_gpu_mnist_multi_seed.%j.e
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=5                        # 5 MPI tasks for 5 seeds
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:2              # Request 2 GPUs (will rotate 5 workers across them)
#SBATCH --mem=80G
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH -A bio240068p
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

mkdir -p /ocean/projects/bio240068p/$USER/logs/EIANN
mkdir -p /ocean/projects/bio240068p/$USER/data/EIANN

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_START_METHOD=thread
export MPI4PY_RC_RECV_MPROBE=false

module purge
module load cuda/12.4.0
module load openmpi/4.0.5-nvhpc22.9

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate eiann

cd ~/EIANN

# Run with MPI
mpirun -n 5 python EIANN/simulate/run_EIANN_mnist_mpi.py \
  --network-config-file-name=20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml \
  --data-dir=/ocean/projects/bio250022p/$USER/data/EIANN \
  --num-seeds=5

# To submit:
# cd ~/EIANN/EIANN/simulate/jobscripts
# sbatch run_EIANN_gpu_bridges_mnist_multiseed.sh

# See logs:
# cd /ocean/projects/bio240068p/$USER/logs/EIANN