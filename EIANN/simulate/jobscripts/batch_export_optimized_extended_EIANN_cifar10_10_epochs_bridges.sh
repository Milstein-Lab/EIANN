#!/bin/bash -l

export CONFIG_DIR=../network_config/cifar10

declare -a config_files=(
  20250827_EIANN_2_hidden_lrf_cifar10_van_bp_relu_SGD_config_G_zero_bias_optimized.yaml
  20250829_EIANN_2_hidden_lrf_cifar10_bpDale_relu_SGD_config_G_zero_bias_optimized.yaml
  20250829_EIANN_2_hidden_lrf_cifar10_DTP_config_5J_zero_bias_optimized.yaml
)

arraylength=${#config_files[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh export_optimized_extended_EIANN_cifar10_10_epochs_bridges.sh $CONFIG_DIR/${config_files[$i]}
done
