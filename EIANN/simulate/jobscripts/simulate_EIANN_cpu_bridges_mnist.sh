#!/bin/bash -l
#SBATCH -J eiann_cpu_mnist
#SBATCH -o /ocean/projects/bio240068p/chennawa/logs/EIANN/eiann_cpu_mnist.%j.o
#SBATCH -e /ocean/projects/bio240068p/chennawa/logs/EIANN/eiann_cpu_mnist.%j.e
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=RM
#SBATCH --mem=32G
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
export WANDB_START_METHOD=thread

module purge

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate eiann

cd ~/EIANN

python EIANN/simulate/run_EIANN_mnist.py \
  --network-config-file-name=20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml \
  --data-dir=/ocean/projects/bio250022p/$USER/data/EIANN --network-seed=66049 --device=cpu --debug


# cd $HOME/EIANN/EIANN/simulate/jobscripts
# sbatch simulate_EIANN_cpu_bridges_mnist.sh

# See logs:
# cd /ocean/projects/bio240068p/$USER/logs/EIANN

# See progress:
# watch -n 1 squeue -u $USER