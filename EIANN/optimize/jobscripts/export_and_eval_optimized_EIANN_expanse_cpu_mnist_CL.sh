#!/bin/bash -l
# Retrains one optimized CL model (nested.analyze with the best params under its model key), exports the network
# pickle after each phase, then runs scripts/get_cl_metrics.py on those pickles. Submits one job running the seeds in
# parallel (1 controller + num_instances worker ranks).
#   pickles: $SCRATCH/data/EIANN/<network_name>/<seed>/<network_name>_phase<i>_<seed>_<data_seed>_<model_key>.pkl
#   metrics: $SCRATCH/data/EIANN/cl_metrics/<model_key>_{accuracy.csv,activity.h5,fisher.h5,training.h5}
#   logs:    $SCRATCH/logs/EIANN/export_eval_EIANN_mnist_CL_<model_key>_<date>.<jobid>.{o,e}
# Existing pickles with this model key are moved to $SCRATCH/data/EIANN/replaced_pickles/<date>/ first, since
# nested.analyze does not overwrite them (and get_cl_metrics.py would then score the old ones).
#
# usage (from the EIANN/ package directory, after pushing/pulling the current params file):
#   bash optimize/jobscripts/export_and_eval_optimized_EIANN_expanse_cpu_mnist_CL.sh <optimize config> <model key> \
#     [num_instances, default 5]
# e.g.
#   bash optimize/jobscripts/export_and_eval_optimized_EIANN_expanse_cpu_mnist_CL.sh \
#     optimize/optimize_config/mnist_CL/5_tasks/20260203_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BTSP_ELR_config_A.yaml \
#     BTSP_CL_5_tasks_ELR_A
# Set PARAM_FILE_PATH to use another params file. The number of splits, task-incremental mode and network name (the
# pickle folder) are read from the optimize config.
# pull the pickles and metrics down afterwards (from the EIANN/ package directory):
#   rsync -av --include='*/' --include='*_phase*.pkl' --exclude='cl_metrics/' --exclude='replaced_pickles/' \
#     --exclude='*' --prune-empty-dirs <user>@login.expanse.sdsc.edu:<SCRATCH>/data/EIANN/ data/
#   rsync -av <user>@login.expanse.sdsc.edu:<SCRATCH>/data/EIANN/cl_metrics/ data/cl_metrics/
export DATE=$(date +%Y%m%d_%H%M%S)
export CONFIG_FILE_PATH="$1"
export MODEL_KEY="$2"
export NUM_INSTANCES="${3:-5}"
export PARAM_FILE_PATH="${PARAM_FILE_PATH:-optimize/optimize_params/mnist_CL/20260921_v2dev_mnist_CL_params.yaml}"
export OUTPUT_DIR=$SCRATCH/data/EIANN

if [ -z "$CONFIG_FILE_PATH" ] || [ -z "$MODEL_KEY" ]; then
  echo "usage: bash $0 <optimize config> <model key> [num_instances]"
  exit 1
fi
if [ ! -f "$CONFIG_FILE_PATH" ]; then
  echo "$0: optimize config not found: $CONFIG_FILE_PATH"
  exit 1
fi
if ! grep -q "^$MODEL_KEY:" "$PARAM_FILE_PATH"; then
  echo "$0: model key $MODEL_KEY not found in $PARAM_FILE_PATH"
  exit 1
fi

export NETWORK_NAME=$(basename "$(grep -m1 'network_config_file_path:' "$CONFIG_FILE_PATH" | awk '{print $2}')" .yaml)
export NUM_SPLITS=$(grep -m1 '^ *num_splits:' "$CONFIG_FILE_PATH" | awk '{print $2}')
if grep -qE '^ *task_incremental: *[Tt]rue' "$CONFIG_FILE_PATH"; then
  export TASK_INCREMENTAL_FLAG=--task-incremental
else
  export TASK_INCREMENTAL_FLAG=
fi
export NUM_RANKS=$((NUM_INSTANCES + 1))
export JOB_NAME=export_eval_EIANN_mnist_CL_"$MODEL_KEY"_"$DATE"
echo "$MODEL_KEY: $NETWORK_NAME, num_splits=$NUM_SPLITS${TASK_INCREMENTAL_FLAG:+ $TASK_INCREMENTAL_FLAG}, $NUM_INSTANCES seeds"

sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o $SCRATCH/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e $SCRATCH/logs/EIANN/$JOB_NAME.%j.e
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n $NUM_RANKS
#SBATCH -t 24:00:00
#SBATCH --mem=32G
#SBATCH --account=$ACCOUNT_NUMBER
#SBATCH --export=ALL
#SBATCH --mail-user=$MAIL_USER
#SBATCH --mail-type=ALL
#SBATCH --constraint="lustre"
#SBATCH --no-requeue

source $HOME/cpu_py311_intelmpi.sh

# one torch thread per MPI rank, and cap glibc malloc arenas: EWC's per-step penalty tensors otherwise appear to
# fragment memory on Linux until the job is OOM killed
export OMP_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

cd $PROJECT/EIANN/EIANN

set -x

# move aside earlier pickles of this model, which nested.analyze would not overwrite
if [ -d $OUTPUT_DIR/$NETWORK_NAME ]; then
  find $OUTPUT_DIR/$NETWORK_NAME -mindepth 2 -maxdepth 2 -regextype posix-extended \
    -regex ".*_phase[0-9]+_[0-9]+_[0-9]+_${MODEL_KEY}\.pkl" | while read -r f; do
    mkdir -p $OUTPUT_DIR/replaced_pickles/$DATE/$NETWORK_NAME/\$(basename \$(dirname \$f))
    mv \$f $OUTPUT_DIR/replaced_pickles/$DATE/$NETWORK_NAME/\$(basename \$(dirname \$f))/
  done
fi

srun -n $NUM_RANKS --mpi=pmi2 python -m mpi4py.futures -m nested.analyze \
  --config-file-path=$CONFIG_FILE_PATH \
  --param-file-path=$PARAM_FILE_PATH \
  --model-key=$MODEL_KEY \
  --output-dir=$OUTPUT_DIR --disp --label=$MODEL_KEY --export \
  --num_instances=$NUM_INSTANCES --store_history=True --retrain=True --full_analysis=False --status_bar=False \
  --framework=mpi || exit 1

# post-hoc CL metrics, serial, using all of the job's cores
OMP_NUM_THREADS=$NUM_RANKS python scripts/get_cl_metrics.py --model-folder-path=$OUTPUT_DIR/$NETWORK_NAME \
  --data-dir=$OUTPUT_DIR/datasets/MNIST_data/ --num-splits=$NUM_SPLITS --label=$MODEL_KEY --model-key=$MODEL_KEY \
  --output-dir=$OUTPUT_DIR/cl_metrics $TASK_INCREMENTAL_FLAG
EOT
