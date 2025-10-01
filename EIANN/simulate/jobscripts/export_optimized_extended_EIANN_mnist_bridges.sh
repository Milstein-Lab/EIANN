#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=export_optimized_extended_EIANN_mnist_"$DATE"
export CONFIG_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /ocean/projects/bio240068p/aaronmil/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e /ocean/projects/bio240068p/aaronmil/logs/EIANN/$JOB_NAME.%j.e
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH -n 6
#SBATCH -t 24:00:00
#SBATCH -A bio250022p
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $PROJECT/EIANN/EIANN/simulate

export MPI4PY_RC_RECV_MPROBE=false

mpirun -n 6 python -m mpi4py.futures simulate_EIANN_mnist.py \
    --network-config-file-path=$CONFIG_FILE_PATH \
    --output-dir=../data/EIANN --disp --export \
    --framework=mpi --label=extended --train_steps=50000
EOT