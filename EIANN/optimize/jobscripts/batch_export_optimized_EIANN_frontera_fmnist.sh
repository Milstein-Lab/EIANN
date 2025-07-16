#!/bin/bash -l
#SBATCH -J batch_export_optimized_EIANN_fmnist
#SBATCH -o /scratch1/06441/aaronmil/logs/EIANN/batch_export_optimized_EIANN_fmnist.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/EIANN/batch_export_optimized_EIANN_fmnist.%j.e
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 36
#SBATCH -t 2:00:00
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $WORK/EIANN/EIANN

export CONFIG_DIR=optimize/optimize_config/fmnist
export PARAM_FILE_PATH=optimize/optimize_params/fmnist/20250607_nested_optimize_fmnist_params.yaml

export MPI4PY_RC_RECV_MPROBE=false

declare -a config_files=(
  20250606_nested_optimize_EIANN_2_hidden_fmnist_van_bp_relu_SGD_config_G_zero_bias.yaml
  20250606_nested_optimize_EIANN_2_hidden_fmnist_bpDale_relu_SGD_config_G_zero_bias.yaml
  20250607_nested_optimize_EIANN_2_hidden_fmnist_DTP_config_5J_zero_bias.yaml
  20250619_nested_optimize_EIANN_2_hidden_fmnist_DTP_config_5J_learn_TD_HTCWN_2_zero_bias.yaml
  20250619_nested_optimize_EIANN_2_hidden_fmnist_BTSP_config_5L_zero_bias.yaml
  20250619_nested_optimize_EIANN_2_hidden_fmnist_BTSP_config_5L_learn_TD_HTCWN_3_zero_bias.yaml
)

declare -a model_keys=(
  van_bp_zero_bias
  bpDale_G_zero_bias
  DTP_5J_zero_bias
  DTP_5J_learn_TD_HTCWN_2_zero_bias
  BTSP_5L_zero_bias
  BTSP_5L_learn_TD_HTCWN_3_zero_bias
)

arraylength=${#config_files[@]}

declare o=0
for ((i=0; i<${arraylength}; i++))
do
  ibrun -n 6 -o $o python -m mpi4py.futures -m nested.analyze \
    --config-file-path=$CONFIG_DIR/${config_files[$i]} \
    --param-file-path=$PARAM_FILE_PATH \
    --model-key=${model_keys[$i]} \
    --output-dir=$SCRATCH/data/EIANN --disp --label=complete --export \
    --full_analysis --store_history --framework=mpi &
  ((o+=6))
done
wait
