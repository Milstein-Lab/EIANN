#!/bin/bash -l
#SBATCH -J simulate_EIANN_mnist
#SBATCH -o /scratch1/06441/aaronmil/logs/EIANN/simulate_EIANN_mnist.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/EIANN/simulate_EIANN_mnist.%j.e
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -n 102
#SBATCH -t 2:00:00
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $WORK/EIANN/EIANN

export CONFIG_DIR=network_config/mnist

export MPI4PY_RC_RECV_MPROBE=false

declare -a config_files=(
  20250103_EIANN_0_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml
  20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml
  20250108_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_fixed_hidden_complete_optimized.yaml
  20240419_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_F_complete_optimized.yaml
  20231129_EIANN_2_hidden_mnist_bpDale_relu_SGD_config_G_complete_optimized.yaml
  20240919_EIANN_2_hidden_mnist_bpDale_noI_relu_SGD_config_G_complete_optimized.yaml
  20241105_EIANN_2_hidden_mnist_Top_Layer_Supervised_Hebb_WeightNorm_config_7_complete_optimized.yaml
  20240508_EIANN_2_hidden_mnist_BP_like_config_1J_complete_optimized.yaml
  20241113_EIANN_2_hidden_mnist_BP_like_config_5M_complete_optimized.yaml
  20241113_EIANN_2_hidden_mnist_BP_like_config_5K_complete_optimized.yaml
  20241009_EIANN_2_hidden_mnist_BP_like_config_5J_complete_optimized.yaml
  20241114_EIANN_2_hidden_mnist_BP_like_config_5J_fixed_TD_complete_optimized.yaml
  20241120_EIANN_2_hidden_mnist_BP_like_config_5J_learn_TD_HTCWN_2_complete_optimized.yaml
  20241125_EIANN_2_hidden_mnist_Hebb_Temp_Contrast_config_2_complete_optimized.yaml
  20240723_EIANN_2_hidden_mnist_Supervised_BCM_config_4_complete_optimized.yaml
  20241212_EIANN_2_hidden_mnist_BTSP_config_5L_complete_optimized.yaml
  20241216_EIANN_2_hidden_mnist_BTSP_config_5L_fixed_TD_complete_optimized.yaml
  20241216_EIANN_2_hidden_mnist_BTSP_config_5L_learn_TD_HTCWN_3_complete_optimized.yaml
)

arraylength=${#config_files[@]}

declare o=0
for ((i=0; i<${arraylength}; i++))
do
  ibrun -n 6 -o $o python -m mpi4py.futures simulate/simulate_EIANN_2_hidden_mnist.py \
    --config-file-path=simulate/config/mnist/simulate_EIANN_1_hidden_mnist_supervised_config.yaml \
    --network-config-file-path=$CONFIG_DIR/${config_files[$i]} \
    --output-dir=$SCRATCH/data/EIANN --disp --label=extended --export \
    --framework=mpi &
  ((o+=6))
done
wait
