#!/bin/bash -l
# Retrains a batch of optimized 2ACDC (random track every episode) models: submits one job per model with
# export_optimized_EIANN_expanse_cpu_2ACDC.sh (nested.analyze export of one pickle per seed). See that script for
# where the pickles and logs go and for the PLOT option.
#
# usage (from the EIANN/ package directory, after pushing/pulling the current params file):
#   bash optimize/jobscripts/batch_export_optimized_EIANN_expanse_cpu_2ACDC.sh [num_instances, default 5]
# pull the pickles down afterwards (from the EIANN/ package directory):
#   rsync -av --exclude='replaced_pickles/' <user>@login.expanse.sdsc.edu:<SCRATCH>/data/EIANN/2ACDC/ data/2ACDC/
export NUM_INSTANCES="${1:-5}"
export CONFIG_DIR=optimize/optimize_config/2ACDC
export PARAM_FILE_PATH="${PARAM_FILE_PATH:-optimize/optimize_params/2ACDC/20261102_2ACDC_params.yaml}"

# one model per line: <model key> <optimize config in CONFIG_DIR>; delete or comment out a line to skip that model
declare -a models=(
  "van_bp_relu_SGD    20261102_nested_optimize_EIANN_2_hidden_2ACDC_van_bp_relu_SGD_config.yaml"
  "eprop_relu_SGD     20261102_nested_optimize_EIANN_2_hidden_2ACDC_eprop_relu_SGD_config.yaml"
)

for model in "${models[@]}"; do
  read -r model_key config_file <<< "$model"
  bash optimize/jobscripts/export_optimized_EIANN_expanse_cpu_2ACDC.sh "$CONFIG_DIR/$config_file" "$model_key" \
    "$NUM_INSTANCES"
done
