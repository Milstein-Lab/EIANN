#!/bin/bash -l
#SBATCH -J optimize_EIANN_mnist_ray_multi
#SBATCH -o /scratch2/11358/yashchennawar5555/logs/EIANN/optimize_EIANN_mnist_ray_multi.%j.o
#SBATCH -e /scratch2/11358/yashchennawar5555/logs/EIANN/optimize_EIANN_mnist_ray_multi.%j.e
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rtx
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
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

cd $HOME/EIANN/EIANN

# --- RAY CLUSTER LAUNCH ---

# 1. Get the list of nodes and the head node IP
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "Head node IP: $head_node_ip"

# 2. Start the Ray Head Node
echo "Starting Head node on $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus=16 --num-gpus=4 --block &

# 3. Start Ray Worker Nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i=1; i<=worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Worker node on $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" ray start --address "$ip_head" --num-cpus=16 --num-gpus=4 --block &
done

# 4. Wait for cluster to initialize
sleep 20

# 5. Export the address so ray.init() in python finds the cluster
export RAY_ADDRESS=$ip_head

# --- RUN OPTIMIZATION ---

python -m nested.optimize --config-file-path=$1 \
  --output-dir=$SCRATCH/data/EIANN --framework=ray --disp \
  --pop_size=20 --max_iter=15 --path_length=3

# cd $HOME/EIANN/EIANN/optimize/jobscripts 
# sbatch optimize_EIANN_ray_multinode_frontera_mnist.sh optimize/optimize_config/mnist/20250103_nested_optimize_EIANN_0_hidden_mnist_van_bp_relu_SGD_config_G.yaml

# See logs:
# cd $SCRATCH/logs/EIANN
