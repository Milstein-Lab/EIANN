#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_EIANN_mnist_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /expanse/lustre/scratch/aaronmil/temp_project/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e /expanse/lustre/scratch/aaronmil/temp_project/logs/EIANN/$JOB_NAME.%j.e
#SBATCH -p compute
#SBATCH -N 8
#SBATCH -n 1001
#SBATCH -t 2:00:00
#SBATCH --mem=0
#SBATCH --account=sua199
#SBATCH --export=ALL
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH --constraint="lustre"

set -x

source $HOME/cpu_py311_intelmpi.sh

cd $PROJECT/EIANN/EIANN

srun -n 1001 --mpi=pmi2 python -m mpi4py.futures -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=$SCRATCH/data/EIANN --pop_size=200 --max_iter=1 --path_length=1 --disp \
  --framework=mpi
EOT
