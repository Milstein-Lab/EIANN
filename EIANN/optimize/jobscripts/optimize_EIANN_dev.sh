#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_EIANN_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/EIANN/$JOB_NAME.%j.e
#SBATCH -p development
#SBATCH -N 18
#SBATCH -n 1008
#SBATCH -t 1:00:00
#SBATCH --mail-user=milstein@cabm.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd $WORK2/EIANN/EIANN

ibrun -n 1008 python -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=$SCRATCH/data/EIANN --pop_size=200 --max_iter=1 --path_length=1 --disp \
  --framework=pc
EOT
