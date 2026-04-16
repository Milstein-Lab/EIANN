#!/bin/bash -l
#SBATCH -J optimize_EIANN_mnist_ray_multi
#SBATCH -o /scratch2/11358/yashchennawar5555/logs/EIANN/optimize_EIANN_mnist_ray_multi.%j.o
#SBATCH -e /scratch2/11358/yashchennawar5555/logs/EIANN/optimize_EIANN_mnist_ray_multi.%j.e
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rtx
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
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
DEVICE="${2:-cpu}"

export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p $RAY_TMPDIR

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
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus=16 --num-gpus=4 \
  --temp-dir "$RAY_TMPDIR" --disable-usage-stats &

# 3. Start Ray Worker Nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i=1; i<=worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Worker node on $node_i"
  srun --overlap --nodes=1 --ntasks=1 -w "$node_i" ray start --address "$ip_head" --num-cpus=16 --num-gpus=4 \
  --disable-usage-stats &
done

# 4. Wait for cluster to initialize
sleep 20

# 4b. Verify that ray is reachable from the head node before launching optimize.
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" ray status --address "$ip_head"

cleanup_ray() {
  set +e
  for node in "${nodes_array[@]}"; do
    stop_output=$(srun --overlap --nodes=1 --ntasks=1 -w "$node" ray stop --force 2>&1) || true
    echo "$stop_output" | grep -E "Stopped all [0-9]+ Ray processes|No active Ray processes" || true
  done
}
trap cleanup_ray EXIT

# 5. Export the address so ray.init() in python finds the cluster
export RAY_ADDRESS=$ip_head

# --- RUN OPTIMIZATION ---

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" python -m nested.optimize --config-file-path=$1 \
  --output-dir=$SCRATCH/data/EIANN --framework=ray --disp \
  --pop_size=9 --max_iter=1 --path_length=1 --num_gpus=0.5 --num_cpus=1 --device=$DEVICE

# srun --num-gpus=4 must equal real number of gpus in frontera rtx node (4)
# if we need more gpus, we can increase --nodes
# total worker gpus = nodes * gpus per node (4) * num_gpus in python (how many gpus can run stuff)

# cd $HOME/EIANN/EIANN/optimize/jobscripts 
# sbatch optimize_EIANN_ray_multinode_frontera_mnist.sh optimize/optimize_config/mnist/20231129_nested_optimize_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G.yaml cuda
# sbatch optimize_EIANN_ray_multinode_frontera_mnist.sh optimize/optimize_config/mnist/20231129_nested_optimize_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G.yaml cuda

# See logs:
# cd $SCRATCH/logs/EIANN

# TODO: larger backprop test
# TODO: try ibrun (and check frontera docs)
# TODO: try 0.25 gpu