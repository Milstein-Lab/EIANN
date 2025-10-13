#!/bin/bash -l
#SBATCH -J batch_export_optimized_EIANN_mnist_extended
#SBATCH -o /scratch1/06441/aaronmil/logs/EIANN/batch_export_optimized_EIANN_mnist_extended.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/EIANN/batch_export_optimized_EIANN_mnist_extended.%j.e
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -t 6:00:00
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $WORK/EIANN/EIANN/simulate

export CONFIG_DIR=../network_config/mnist

export MPI4PY_RC_RECV_MPROBE=false

declare -a config_files=(
  20240714_EIANN_2_hidden_mnist_Supervised_Hebb_WeightNorm_config_4_optimized.yaml
)

arraylength=${#config_files[@]}

declare o=0
for ((i=0; i<${arraylength}; i++))
do
  ibrun -n 6 -o $o python -m mpi4py.futures simulate_EIANN_mnist.py \
    --network-config-file-path=$CONFIG_DIR/${config_files[$i]} \
    --output-dir=$SCRATCH/data/EIANN --disp --export \
    --framework=mpi --label=extended --train_steps=50000 &
  ((o+=6))
done
wait
