#!/bin/bash -l
#SBATCH -J simulate_EIANN_mnist
#SBATCH -o /scratch1/06441/aaronmil/logs/EIANN/simulate_EIANN_mnist.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/EIANN/simulate_EIANN_mnist.%j.e
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -t 1:00:00
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $WORK/EIANN/EIANN

export CONFIG_DIR=network_config/mnist

export MPI4PY_RC_RECV_MPROBE=false

declare -a config_files=(
  20250103_EIANN_0_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml
  20231129_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G_complete_optimized.yaml
)

arraylength=${#config_files[@]}

declare o=0
for ((i=0; i<${arraylength}; i++))
do
  echo ibrun -n 6 -o $o python -m mpi4py.futures simulate/simulate_EIANN_2_hidden_mnist.py \
    --config-file-path=simulate/config/mnist/simulate_EIANN_1_hidden_mnist_supervised_config.yaml \
    --network-config-file-path=$CONFIG_DIR/${config_files[$i]} \
    --output-dir=$SCRATCH/data/EIANN --disp --label=extended --export \
    --framework=mpi&
  ((o+=6))
done
wait
