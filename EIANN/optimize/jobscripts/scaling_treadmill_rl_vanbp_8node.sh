#!/bin/bash -l
# ---------------------------------------------------------------------------------------------------
# STRONG-SCALING benchmark for nested.optimize (RL vanilla backprop).
#   Fixed workload across all four scripts: pop_size=200 x num_instances=5 = 1000 evaluations per
#   generation, 3 generations (max_iter=1, path_length=3). ONLY the node/worker count changes:
#       1node -> 127 workers (8 waves) | 2node -> 255 (4) | 4node -> 511 (2) | 8node -> 1023 (1 wave)
#   Compare wall time (or nested's per-generation time from --disp) across the four to get the
#   speedup curve. OMP_NUM_THREADS=1 is required so torch does not oversubscribe threads.
# THIS SCRIPT: 8 nodes (1000 evaluations fit in a single wave).
# ---------------------------------------------------------------------------------------------------
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME="rl_vanbp_scaling_8node_${DATE}"
export CONFIG_FILE_PATH="optimize/optimize_config/treadmill_RL/nested_optimize_EIANN_2_hidden_treadmill_RL_van_bp_relu_SGD_config.yaml"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /expanse/lustre/scratch/rpemmaraju/temp_project/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e /expanse/lustre/scratch/rpemmaraju/temp_project/logs/EIANN/$JOB_NAME.%j.e
#SBATCH -p compute
#SBATCH -N 8
#SBATCH -n 1024
#SBATCH -t 00:15:00
#SBATCH --mem=249208M
#SBATCH --export=ALL
#SBATCH --account=sua199
#SBATCH --mail-user=rp933@rwjms.rutgers.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint="lustre"
#SBATCH --no-requeue

set -x
source $HOME/cpu_py311.sh
cd $PROJECT/EIANN/EIANN

# one worker thread per rank; ranks are packed ~128/node, so multi-threaded torch would oversubscribe
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun -n 1024 --mpi=pmi2 python -m mpi4py.futures -m nested.optimize \
  --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=$SCRATCH/data/EIANN \
  --pop_size=200 --num_instances=5 --max_iter=1 --path_length=3 \
  --train_episodes=200 \
  --disp --framework=mpi
EOT
