#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=export_optimized_EIANN_mnist_"$DATE"
export CONFIG_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/EIANN/batch_export_optimized_EIANN_mnist.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/EIANN/batch_export_optimized_EIANN_mnist.%j.e
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -t 2:00:00
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $WORK/EIANN/EIANN/simulate

export MPI4PY_RC_RECV_MPROBE=false

ibrun -n 6 python -m mpi4py.futures simulate_EIANN_mnist.py \
    --network-config-file-path=$CONFIG_FILE_PATH \
    --output-dir=$SCRATCH/data/EIANN --disp --export \
    --framework=mpi --label=extended --train_steps=50000
EOT