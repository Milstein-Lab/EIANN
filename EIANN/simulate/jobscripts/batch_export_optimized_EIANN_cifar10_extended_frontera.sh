#!/bin/bash -l
#SBATCH -J batch_export_optimized_EIANN_cifar10_extended
#SBATCH -o /scratch1/06441/aaronmil/logs/EIANN/batch_export_optimized_EIANN_cifar10_extended.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/EIANN/batch_export_optimized_EIANN_cifar10_extended.%j.e
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 18
#SBATCH -t 2:00:00
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $WORK/EIANN/EIANN/simulate

export CONFIG_DIR=../network_config/cifar10

export MPI4PY_RC_RECV_MPROBE=false

declare -a config_files=(
  20250827_EIANN_2_hidden_lrf_cifar10_van_bp_relu_SGD_config_G_zero_bias_optimized.yaml
  20250829_EIANN_2_hidden_lrf_cifar10_bpDale_relu_SGD_config_G_zero_bias_optimized.yaml
  20250829_EIANN_2_hidden_lrf_cifar10_DTP_config_5J_zero_bias_optimized.yaml
)

arraylength=${#config_files[@]}

declare o=0
for ((i=0; i<${arraylength}; i++))
do
  ibrun -n 6 -o $o python -m mpi4py.futures simulate_EIANN_cifar10.py \
    --network-config-file-path=$CONFIG_DIR/${config_files[$i]} \
    --output-dir=$SCRATCH/data/EIANN --disp --export \
    --framework=mpi --label=extended --train_steps=40000 &
  ((o+=6))
done
wait
