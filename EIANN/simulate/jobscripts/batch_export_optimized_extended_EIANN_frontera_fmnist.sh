#!/bin/bash -l
#SBATCH -J batch_export_optimized_extended_EIANN_fmnist
#SBATCH -o /scratch1/06441/aaronmil/logs/EIANN/batch_export_optimized_extended_EIANN_fmnist.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/EIANN/batch_export_optimized_extended_EIANN_fmnist.%j.e
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -t 2:00:00
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $WORK/EIANN/EIANN/simulate

export CONFIG_DIR=../network_config/fmnist

export MPI4PY_RC_RECV_MPROBE=false

declare -a config_files=(
  20250819_EIANN_0_hidden_fmnist_van_bp_relu_SGD_config_G_zero_bias_optimized.yaml
  20250819_EIANN_2_fixed_hidden_fmnist_van_bp_relu_SGD_config_G_zero_bias_optimized.yaml
)

arraylength=${#config_files[@]}

declare o=0
for ((i=0; i<${arraylength}; i++))
do
  ibrun -n 6 -o $o python -m mpi4py.futures simulate_EIANN_fashion_mnist.py \
    --network-config-file-path=$CONFIG_DIR/${config_files[$i]} \
    --output-dir=$SCRATCH/data/EIANN --disp --label=extended --export \
    --framework=mpi &
  ((o+=6))
done
wait
