#!/bin/bash -l
#SBATCH -J eiann_gpu_mnist_ray
#SBATCH -o /ocean/projects/bio240068p/chennawa/logs/EIANN/eiann_gpu_mnist_ray.%j.o
#SBATCH -e /ocean/projects/bio240068p/chennawa/logs/EIANN/eiann_gpu_mnist_ray.%j.e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:3
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -A bio240068p
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

module purge
module load cuda/12.4.0

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate eiann

cd ~/EIANN

# Start Ray (local mode)
ray start --head --num-gpus=3 --num-cpus=12

python EIANN/simulate/run_EIANN_mnist_ray.py \
  --network-config-file-name=20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml \
  --data-dir=/ocean/projects/bio250022p/$USER/data/EIANN \
  --num-seeds=5

ray stop

# To submit:
# cd ~/EIANN/EIANN/simulate/jobscripts
# sbatch run_EIANN_gpu_bridges_mnist_ray.sh

# See logs:
# cd /ocean/projects/bio240068p/$USER/logs/EIANN