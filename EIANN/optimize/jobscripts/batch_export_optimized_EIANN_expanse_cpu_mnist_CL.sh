#!/bin/bash -l
# Retrains a batch of optimized CL models (nested.analyze with the best params under each model key) and exports the
# network pickle after each phase to
#   $SCRATCH/data/EIANN/<network_name>/<seed>/<network_name>_phase<i>_<seed>_<data_seed>_<model_key>.pkl
# Submits one job per model, each running 5 seeds in parallel (1 controller + 5 worker ranks).
# Per-model logs go to $SCRATCH/logs/EIANN/export_EIANN_mnist_CL_<model_key>_<date>.<jobid>.{o,e}
#
# usage (from the EIANN/ package directory, after pushing/pulling the current params file):
#   bash optimize/jobscripts/batch_export_optimized_EIANN_expanse_cpu_mnist_CL.sh
# pull the pickles down afterwards (from the EIANN/ package directory):
#   rsync -av --include='*/' --include='*_phase*.pkl' --exclude='*' --prune-empty-dirs \
#     <user>@login.expanse.sdsc.edu:<SCRATCH>/data/EIANN/ data/
export DATE=$(date +%Y%m%d_%H%M%S)
export CONFIG_DIR=optimize/optimize_config/mnist_CL/5_tasks
export PARAM_FILE_PATH=optimize/optimize_params/mnist_CL/20260921_v2dev_mnist_CL_params.yaml

declare -a config_files=(
  20240923_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_relu_SGD_config_G.yaml
  20231129_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_bpDale_relu_SGD_config_G.yaml
  20260325_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BP_like_config_5J.yaml
  20251229_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BTSP_config_6L.yaml
  20260203_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BTSP_ELR_config_A.yaml
  20260930_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_SI_relu_SGD_config_A.yaml
  20260928_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_EWC_relu_SGD_config_A.yaml
  20261005_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_SI_LR_relu_SGD_config_A.yaml
  20261005_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_EWC_LR_relu_SGD_config_A.yaml
)

declare -a model_keys=(
  van_bp_CL_5_tasks_G
  bpDale_CL_5_tasks_G
  BP_like_CL_5_tasks_5J
  BTSP_CL_5_tasks_6L
  BTSP_CL_5_tasks_ELR_A
  van_bp_CL_5_tasks_SI
  van_bp_CL_5_tasks_EWC
  van_bp_CL_5_tasks_SI_LR
  van_bp_CL_5_tasks_EWC_LR
)

for ((i=0; i<${#config_files[@]}; i++))
do
  export JOB_NAME=export_EIANN_mnist_CL_${model_keys[$i]}_"$DATE"
  sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o $SCRATCH/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e $SCRATCH/logs/EIANN/$JOB_NAME.%j.e
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -t 24:00:00
#SBATCH --mem=32G
#SBATCH --account=$ACCOUNT_NUMBER
#SBATCH --export=ALL
#SBATCH --mail-user=$MAIL_USER
#SBATCH --mail-type=ALL
#SBATCH --constraint="lustre"
#SBATCH --no-requeue

source $HOME/cpu_py311_intelmpi.sh

cd $PROJECT/EIANN/EIANN

set -x

srun -n 6 --mpi=pmi2 python -m mpi4py.futures -m nested.analyze \
  --config-file-path=$CONFIG_DIR/${config_files[$i]} \
  --param-file-path=$PARAM_FILE_PATH \
  --model-key=${model_keys[$i]} \
  --output-dir=$SCRATCH/data/EIANN --disp --label=${model_keys[$i]} --export \
  --num_instances=5 --store_history=True --retrain=True --full_analysis=False --status_bar=False \
  --framework=mpi
EOT
done
