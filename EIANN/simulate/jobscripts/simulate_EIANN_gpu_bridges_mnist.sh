#!/bin/bash -l
#SBATCH -J eiann_gpu_mnist
#SBATCH -o eiann_gpu_mnist.%j.o
#SBATCH -e eiann_gpu_mnist.%j.e
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:4
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -A bio240068p
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

mkdir -p /ocean/projects/bio240068p/$USER/logs/EIANN
mkdir -p /ocean/projects/bio240068p/$USER/data/EIANN

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_START_METHOD=thread

module purge
module load cuda/11.7.1

LOG=/ocean/projects/bio240068p/$USER/logs/EIANN/debug_test.log
echo "SLURM job started on $(hostname)" > $LOG
date >> $LOG

echo "Activating conda..." >> $LOG
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh || { echo "Failed to source conda" >> $LOG; exit 1; }
conda activate eiann || { echo "Failed to activate eiann" >> $LOG; exit 1; }

cd ~/EIANN || { echo "Failed to cd into EIANN" >> $LOG; exit 1; }

echo "Running EIANN MNIST script..." >> $LOG
python EIANN/simulate/run_EIANN_mnist.py \
  --network-config-file-name=20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml \
  --data-dir=/ocean/projects/bio240068p/$USER/data/EIANN \
  >> $LOG 2>&1


# cd $HOME/EIANN/EIANN/simulate/jobscripts
# sbatch simulate_EIANN_gpu_bridges_mnist.sh van_bp_relu

# See logs:
# cd /ocean/projects/bio240068p/$USER/logs/EIANN

# See progress:
# watch -n 1 squeue -u $USER

# Request one node:
# interact -p GPU-shared --gres=gpu:v100-32:4 -t 30:00 -A bio240068p

# /ocean/projects/bio240068p/$USER/logs/EIANN/