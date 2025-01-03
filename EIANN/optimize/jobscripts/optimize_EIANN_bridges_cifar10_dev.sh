#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_EIANN_cifar10_dev_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /ocean/projects/bio240068p/aaronmil/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e /ocean/projects/bio240068p/aaronmil/logs/EIANN/$JOB_NAME.%j.e
#SBATCH -p RM
#SBATCH -N 8
#SBATCH --ntasks-per-node=128
#SBATCH -n 1024
#SBATCH -t 1:00:00
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $PROJECT/EIANN/EIANN

export MPI4PY_RC_RECV_MPROBE=false

mpirun -n 1001 python -m mpi4py.futures -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=data/cifar10 --pop_size=200 --max_iter=1 --path_length=1 --disp \
  --framework=mpi
EOT
