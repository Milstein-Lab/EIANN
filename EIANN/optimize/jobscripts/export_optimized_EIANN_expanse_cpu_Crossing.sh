#!/bin/bash -l
# Retrains one optimized Crossing (gridworld tasks A, B, C interleaved at random) model with nested.analyze (the best params under
# its model key) and exports one pickle per seed, holding network.rl_history with the greedy validation episodes. The
# features (rewards, success/navigation/choice rates) are printed
# to the .o log. Submits one job running the seeds in parallel (1 controller + num_instances worker ranks).
#   pickles: $SCRATCH/data/EIANN/Crossing/<network_name>/<network_name>_<seed>_<data_seed>_<label>.pkl
#   logs:    $SCRATCH/logs/EIANN/export_Crossing/export_EIANN_Crossing_<label>_<date>.<jobid>.{o,e}
# Existing pickles with this label are moved to $SCRATCH/data/EIANN/Crossing/replaced_pickles/<date>/ first, since
# nested.analyze does not overwrite them. The crossing_plots functions (e.g. animate_learning) work on them.
#
# usage (from the EIANN/ package directory, after pushing/pulling the current params file):
#   bash optimize/jobscripts/export_optimized_EIANN_expanse_cpu_Crossing.sh <optimize config> <model key> \
#     [num_instances, default 5]
# e.g.
#   bash optimize/jobscripts/export_optimized_EIANN_expanse_cpu_Crossing.sh \
#     optimize/optimize_config/Crossing/20261110_nested_optimize_EIANN_2_hidden_Crossing_van_bp_relu_SGD_config.yaml van_bp_relu_SGD
# Set PARAM_FILE_PATH to use another params file.
# Set PLOT=1 to also save the plot_crossing_summary figures (png) to $SCRATCH/data/EIANN/Crossing/plots/.
# pull the pickles down afterwards (from the EIANN/ package directory):
#   rsync -av --exclude='replaced_pickles/' <user>@login.expanse.sdsc.edu:<SCRATCH>/data/EIANN/Crossing/ data/Crossing/
export DATE=$(date +%Y%m%d_%H%M%S)
export CONFIG_FILE_PATH="$1"
export MODEL_KEY="$2"
export NUM_INSTANCES="${3:-5}"
export PARAM_FILE_PATH="${PARAM_FILE_PATH:-optimize/optimize_params/Crossing/20261110_Crossing_params.yaml}"
export OUTPUT_DIR=$SCRATCH/data/EIANN/Crossing

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

export LABEL="$MODEL_KEY"
if [ "${PLOT:-0}" = 1 ]; then
  export PLOT_ARGS="--plot --plot_dir=$OUTPUT_DIR/plots --plot_format=png"
else
  export PLOT_ARGS=""
fi
export NETWORK_NAME=$(basename "$(grep -m1 'network_config_file_path:' "$CONFIG_FILE_PATH" | awk '{print $2}')" .yaml)
export JOB_NAME=export_EIANN_Crossing_"$LABEL"_"$DATE"
export NUM_RANKS=$((NUM_INSTANCES + 1))
echo "$MODEL_KEY: $NETWORK_NAME, label $LABEL, $NUM_INSTANCES seeds${PLOT_ARGS:+, with plots}"

export LOG_DIR=$SCRATCH/logs/EIANN/export_Crossing
mkdir -p $LOG_DIR $OUTPUT_DIR/plots

sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o $LOG_DIR/$JOB_NAME.%j.o
#SBATCH -e $LOG_DIR/$JOB_NAME.%j.e
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n $NUM_RANKS
#SBATCH -t 4:00:00
#SBATCH --mem=16G
#SBATCH --account=$ACCOUNT_NUMBER
#SBATCH --export=ALL
#SBATCH --mail-user=$MAIL_USER
#SBATCH --mail-type=ALL
#SBATCH --constraint="lustre"
#SBATCH --no-requeue

source $HOME/cpu_py311_intelmpi.sh

# one torch thread per MPI rank, as in the MNIST jobscripts; plots are saved, never shown
export OMP_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MPLBACKEND=Agg

cd $PROJECT/EIANN/EIANN

set -x

# move aside earlier pickles with this label, which nested.analyze would not overwrite
if [ -d $OUTPUT_DIR/$NETWORK_NAME ]; then
  find $OUTPUT_DIR/$NETWORK_NAME -maxdepth 1 -regextype posix-extended \
    -regex ".*/${NETWORK_NAME}_[0-9]+_[0-9]+_${LABEL}\.pkl" | while read -r f; do
    mkdir -p $OUTPUT_DIR/replaced_pickles/$DATE/$NETWORK_NAME
    mv \$f $OUTPUT_DIR/replaced_pickles/$DATE/$NETWORK_NAME/
  done
fi

srun -n $NUM_RANKS --mpi=pmi2 python -m mpi4py.futures -m nested.analyze \
  --config-file-path=$CONFIG_FILE_PATH \
  --param-file-path=$PARAM_FILE_PATH \
  --model-key=$MODEL_KEY \
  --output-dir=$OUTPUT_DIR --disp --label=$LABEL --export \
  --num_instances=$NUM_INSTANCES --retrain=True --status_bar=False \
  --framework=mpi $PLOT_ARGS
EOT
