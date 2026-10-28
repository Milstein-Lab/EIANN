#!/bin/bash -l
# Retrains a batch of optimized class-incremental CL models and computes their CL metrics: submits one job per model
# with export_and_eval_optimized_EIANN_expanse_cpu_mnist_CL.sh (nested.analyze export of the phase pickles, then
# scripts/get_cl_metrics.py). See that script for where the pickles, metrics and logs go.
#
# usage (from the EIANN/ package directory, after pushing/pulling the current params file):
#   bash optimize/jobscripts/batch_export_optimized_EIANN_expanse_cpu_mnist_CL.sh [num_instances, default 5]
# pull the pickles and metrics down afterwards (from the EIANN/ package directory):
#   rsync -av --include='*/' --include='*_phase*.pkl' --exclude='cl_metrics/' --exclude='replaced_pickles/' \
#     --exclude='*' --prune-empty-dirs <user>@login.expanse.sdsc.edu:<SCRATCH>/data/EIANN/ data/
#   rsync -av <user>@login.expanse.sdsc.edu:<SCRATCH>/data/EIANN/cl_metrics/ data/cl_metrics/
export NUM_INSTANCES="${1:-5}"
export CONFIG_DIR=optimize/optimize_config/mnist_CL/5_tasks
export PARAM_FILE_PATH=optimize/optimize_params/mnist_CL/20260921_v2dev_mnist_CL_params.yaml

# one model per line: <model key> <optimize config in CONFIG_DIR>; delete or comment out a line to skip that model
declare -a models=(
  "van_bp_CL_5_tasks_G          20240923_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_relu_SGD_config_G.yaml"
  "van_bp_CL_5_tasks_EWC        20260928_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_EWC_relu_SGD_config_A.yaml"
  "van_bp_CL_5_tasks_EWC_LR     20261005_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_EWC_LR_relu_SGD_config_A.yaml"
  "van_bp_CL_5_tasks_SI         20260930_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_SI_relu_SGD_config_A.yaml"
  "van_bp_CL_5_tasks_SI_LR      20261005_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_SI_LR_relu_SGD_config_A.yaml"
  "van_bp_CL_5_tasks_SI_ELR     20261014_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_SI_ELR_relu_SGD_config_A.yaml"
  "van_bp_CL_5_tasks_SI_HG      20261014_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_SI_HD_relu_SGD_config_A.yaml"
  "van_bp_CL_5_tasks_SI_HG_v1   20261021_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_van_bp_SI_HD_v1_relu_SGD_config_A.yaml"
  "bpDale_CL_5_tasks_G          20231129_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_bpDale_relu_SGD_config_G.yaml"
  "bpDale_CL_5_tasks_EWC        20260928_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_bpDale_relu_SGD_EWC_config_A.yaml"
  "bpDale_CL_5_tasks_SI         20260930_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_bpDale_relu_SGD_SI_config_A.yaml"
  "bpDale_CL_5_tasks_SI_ELR     20261014_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_bpDale_relu_SGD_SI_ELR_config_A.yaml"
  "BP_like_CL_5_tasks_5J        20260325_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BP_like_config_5J.yaml"
  "dtp_CL_5_tasks_SI            20261002_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_DTP_SI_config_5J.yaml"
  "BTSP_CL_5_tasks_6L           20251229_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BTSP_config_6L.yaml"
  "BTSP_CL_5_tasks_ELR_A        20260203_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BTSP_ELR_config_A.yaml"
  "BTSP_SI_CL_5_tasks           20261013_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BTSP_SI_config_6L.yaml"
  "BTSP_BP_CL_5_tasks           20261012_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BTSP_BP_config_A.yaml"
  "BTSP_BP_ELR_CL_5_tasks       20261012_nested_optimize_EIANN_2_hidden_CL_mnist_5_tasks_BTSP_ELR_BP_config_A.yaml"
)

for model in "${models[@]}"; do
  read -r model_key config_file <<< "$model"
  bash optimize/jobscripts/export_and_eval_optimized_EIANN_expanse_cpu_mnist_CL.sh "$CONFIG_DIR/$config_file" \
    "$model_key" "$NUM_INSTANCES"
done
