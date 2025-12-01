#!/bin/bash -l
#SBATCH -J eiann_gpu_mnist_ray
#SBATCH -o /ocean/projects/bio240068p/chennawa/logs/EIANN/eiann_gpu_mnist_ray.%j.o
#SBATCH -e /ocean/projects/bio240068p/chennawa/logs/EIANN/eiann_gpu_mnist_ray.%j.e
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=GPU
#SBATCH --gres=gpu:v100-32:8
#SBATCH --mem=80G
#SBATCH --cpus-per-task=15
#SBATCH --time=02:00:00
#SBATCH -A bio240068p
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

module purge
module load cuda/12.4.0

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate eiann

cd ~/EIANN

# --- RAY CLUSTER LAUNCH ---

# 1. Get the list of nodes and the head node IP
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
# Get the IP address of the head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "Head node IP: $head_node_ip"

# 2. Start the Ray Head Node
# Use --block (and "&") so the process stays alive in the background
echo "Starting Head node on $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus=15 --num-gpus=8 --block &

# 3. Start Ray Worker Nodes
# Loop over the rest of the nodes (starting from index 1)
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i=1; i<=worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Worker node on $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" ray start --address "$ip_head" --num-cpus=15 --num-gpus=8 --block &
done

# 4. Wait for cluster to initialize
sleep 20

# 5. Export the address so ray.init() in python finds the cluster
export RAY_ADDRESS=$ip_head

# --- RUN SCRIPT ---

python EIANN/simulate/run_EIANN_mnist_ray.py \
  --network-config-file-name=20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml \
  --data-dir=/ocean/projects/bio250022p/$USER/data/EIANN \
  --num-seeds=32


# To submit:
# cd ~/EIANN/EIANN/simulate/jobscripts
# sbatch run_EIANN_gpu_multi_node_bridges_mnist_ray.sh

# See logs:
# cd /ocean/projects/bio240068p/$USER/logs/EIANN