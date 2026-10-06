#!/bin/bash -l
# Retrains a batch of optimized CL models (nested.analyze with the best params under each model key) and exports the
# network pickle after each phase to
#   $SCRATCH/data/EIANN/<network_name>/<seed>/<network_name>_phase<i>_<seed>_<data_seed>_<model_key>.pkl
# Each model runs as its own srun step with NUM_INSTANCES seeds in parallel; all steps share one node.
# Per-model logs go to $SCRATCH/logs/EIANN/<job name>_<model_key>.log
#
# usage (from the EIANN/ package directory, after pushing/pulling the current params file):
#   bash optimize/jobscripts/batch_export_optimized_EIANN_expanse_cpu_mnist_CL.sh [num_instances (default 5)]
# pull the pickles down afterwards (from the EIANN/ package directory):
#   rsync -av --include='*/' --include='*_phase*.pkl' --exclude='*' --prune-empty-dirs \
#     <user>@login.expanse.sdsc.edu:<SCRATCH>/data/EIANN/ data/
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=analyze_EIANN_mnist_CL_batch_"$DATE"
export NUM_INSTANCES=${1:-5}
export PARAM_FILE_PATH=optimize/optimize_params/mnist_CL/20260921_v2dev_mnist_CL_params.yaml
export CONFIG_DIR=optimize/optimize_config/mnist_CL/5_tasks

# <optimize config file in CONFIG_DIR>:<model key in PARAM_FILE_PATH>
declare -a models=(
  20240923_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_relu_SGD_config_G.yaml:van_bp_CL_5_tasks_G
  20231129_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_bpDale_relu_SGD_config_G.yaml:bpDale_CL_5_tasks_G
  20260325_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BP_like_config_5J.yaml:BP_like_CL_5_tasks_5J
  20251229_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BTSP_config_6L.yaml:BTSP_CL_5_tasks_6L
  20260203_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BTSP_ELR_config_A.yaml:BTSP_CL_5_tasks_ELR_A
  20260930_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_SI_relu_SGD_config_A.yaml:van_bp_CL_5_tasks_SI
  20260928_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_EWC_relu_SGD_config_A.yaml:van_bp_CL_5_tasks_EWC
  20261005_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_SI_LR_relu_SGD_config_A.yaml:van_bp_CL_5_tasks_SI_LR
  20261005_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_EWC_LR_relu_SGD_config_A.yaml:van_bp_CL_5_tasks_EWC_LR
)

# one controller rank + NUM_INSTANCES workers per model
export TASKS_PER_MODEL=$((NUM_INSTANCES + 1))
export NUM_TASKS=$((${#models[@]} * TASKS_PER_MODEL))
if [ $NUM_TASKS -gt 128 ]; then
  echo "$NUM_TASKS tasks do not fit on one 128-core node; reduce num_instances or split the model list"
  exit 1
fi

sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o $SCRATCH/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e $SCRATCH/logs/EIANN/$JOB_NAME.%j.e
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n $NUM_TASKS
#SBATCH -t 24:00:00
#SBATCH --mem=249208M
#SBATCH --account=$ACCOUNT_NUMBER
#SBATCH --export=ALL
#SBATCH --mail-user=$MAIL_USER
#SBATCH --mail-type=ALL
#SBATCH --constraint="lustre"
#SBATCH --no-requeue

source $HOME/cpu_py311_intelmpi.sh

cd $PROJECT/EIANN/EIANN

set -x

for model in ${models[*]}; do
  config=\${model%%:*}
  key=\${model##*:}
  srun -n $TASKS_PER_MODEL --exact --mpi=pmi2 python -m mpi4py.futures -m nested.analyze \
    --config-file-path=$CONFIG_DIR/\$config --param-file-path=$PARAM_FILE_PATH --model-key=\$key \
    --output-dir=$SCRATCH/data/EIANN --label=\$key --export --framework=mpi --num_instances=$NUM_INSTANCES \
    --store_history=True --retrain=True --full_analysis=False --status_bar=False \
    > $SCRATCH/logs/EIANN/${JOB_NAME}_\$key.log 2>&1 &
done
wait
EOT
