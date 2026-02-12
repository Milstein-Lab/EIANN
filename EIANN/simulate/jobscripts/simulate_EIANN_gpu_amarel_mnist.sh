#!/bin/bash

# Export the current date and time for job labeling
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$1"
export JOB_NAME=eiann_gpu_mnist_"$LABEL"_"$DATE"

# Environment variables to optimize performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_START_METHOD=thread

# Create directories for logs and scratch data
mkdir -p /scratch/${USER}/logs/eiann
mkdir -p /scratch/${USER}/data/eiann

# Submit the job
sbatch <<EOT
#!/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch/${USER}/logs/eiann/$JOB_NAME.%j.o
#SBATCH -e /scratch/${USER}/logs/eiann/$JOB_NAME.%j.e
#SBATCH --requeue
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

module purge
module use /projects/community/modulefiles
module load gcc/10.2.0-bz186
module load cuda/11.7

set -x

cd $HOME/EIANN/

source ~/miniconda/etc/profile.d/conda.sh
conda activate eiann5
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python EIANN/simulate/run_EIANN_mnist.py \
--network-config-file-name=20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml \
--data-dir=/scratch/${USER}/data/eiann --device=cuda
EOT

# Submit job:
# cd $HOME/EIANN/EIANN/simulate/jobscripts
# sbatch simulate_EIANN_gpu_amarel_mnist.sh van_bp_relu

# See logs:
# cd /scratch/$USER/logs/eiann

# See output pkl files:
# cd /scratch/${USER}/data/eiann/

# See the progress
# watch -n 1 squeue -u $USER

# before python call:
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
