#!/bin/bash -l
# Retrains a batch of optimized Crossing_CL (gridworld tasks A, B, C in blocks) models: submits one job per model with
# export_optimized_EIANN_expanse_cpu_Crossing_CL.sh (nested.analyze export of one pickle per seed). See that script for
# where the pickles and logs go and for the PLOT option.
#
# usage (from the EIANN/ package directory, after pushing/pulling the current params file):
#   bash optimize/jobscripts/batch_export_optimized_EIANN_expanse_cpu_Crossing_CL.sh [num_instances, default 5]
# to train for more cycles than were optimized (labels become <model key>_<n>cycles):
#   NUM_CYCLES=4 bash optimize/jobscripts/batch_export_optimized_EIANN_expanse_cpu_Crossing_CL.sh
# pull the pickles down afterwards (from the EIANN/ package directory):
#   rsync -av --exclude='replaced_pickles/' <user>@login.expanse.sdsc.edu:<SCRATCH>/data/EIANN/Crossing_CL/ data/Crossing_CL/
export NUM_INSTANCES="${1:-5}"
export CONFIG_DIR=optimize/optimize_config/Crossing_CL
export PARAM_FILE_PATH="${PARAM_FILE_PATH:-optimize/optimize_params/Crossing_CL/20261104_Crossing_CL_params.yaml}"

# one model per line: <model key> <optimize config in CONFIG_DIR>; delete or comment out a line to skip that model
declare -a models=(
  "van_bp_relu_SGD    20261104_nested_optimize_EIANN_2_hidden_Crossing_CL_van_bp_relu_SGD_config.yaml"
)

for model in "${models[@]}"; do
  read -r model_key config_file <<< "$model"
  bash optimize/jobscripts/export_optimized_EIANN_expanse_cpu_Crossing_CL.sh "$CONFIG_DIR/$config_file" "$model_key" \
    "$NUM_INSTANCES"
done
