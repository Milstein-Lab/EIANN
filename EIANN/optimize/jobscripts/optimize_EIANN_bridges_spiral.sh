#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_EIANN_spiral_"$LABEL"_"$DATE"
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
#SBATCH -t 12:00:00
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $PROJECT/EIANN/EIANN

mpirun -n 1001 python -m mpi4py.futures -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=data/spiral --pop_size=200 --max_iter=50 --path_length=3 --disp \
  --framework=mpi
EOT
