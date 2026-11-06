#!/bin/bash -l
# nested optimization of a Crossing_CL (gridworld tasks A, B, C in blocks) model on SDSC Expanse.
# The script calls sbatch itself; run from anywhere:
#   bash optimize/jobscripts/optimize_EIANN_expanse_cpu_Crossing_CL.sh <optimize config> <label> [history file]
# e.g.
#   bash optimize/jobscripts/optimize_EIANN_expanse_cpu_Crossing_CL.sh \
#     optimize/optimize_config/Crossing_CL/20261104_nested_optimize_EIANN_2_hidden_Crossing_CL_van_bp_relu_SGD_config.yaml van_bp_relu_SGD
# With a history file (.hdf5 from a previous run), the optimization resumes from it (--hot-start).
#
# Sizing: pop_size=200 x num_instances=5 seeds (from the optimize config) = 1000 evaluations per generation, one per
# MPI rank, plus the controller. A seed (3 tasks x 2000 episodes) takes ~30 s on a laptop (van_bp_relu_SGD at x0), and up
# to ~2x longer for params that rarely reach a tile (every episode runs to max_steps). 50 iterations x path_length 3
# (150 generations) should then take ~2-5 h on Expanse cores; 24 h leaves margin. RL needs gymnasium and minigrid in the
# cluster environment.
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_EIANN_Crossing_CL_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
export LOG_DIR=$SCRATCH/logs/EIANN/optimize_Crossing_CL
if [ -n "$3" ]; then
  export HOT_START_ARGS="--hot-start --history-file-path=$3"
else
  export HOT_START_ARGS=""
fi
mkdir -p $LOG_DIR

sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o $LOG_DIR/$JOB_NAME.%j.o
#SBATCH -e $LOG_DIR/$JOB_NAME.%j.e
#SBATCH -p compute
#SBATCH -N 16
#SBATCH -n 1001
#SBATCH -t 48:00:00
#SBATCH --mem=249208M
#SBATCH --account=$ACCOUNT_NUMBER
#SBATCH --export=ALL
#SBATCH --mail-user=$MAIL_USER
#SBATCH --mail-type=ALL
#SBATCH --constraint="lustre"
#SBATCH --no-requeue

source $HOME/cpu_py311_intelmpi.sh

# one torch thread per MPI rank, and cap glibc malloc arenas, as in the MNIST jobscripts
export OMP_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

cd $PROJECT/EIANN/EIANN

set -x

srun -n 1001 --mpi=pmi2 python -m mpi4py.futures -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=$SCRATCH/data/EIANN --pop_size=200 --max_iter=50 --path_length=3 --disp \
  --framework=mpi $HOT_START_ARGS
EOT
