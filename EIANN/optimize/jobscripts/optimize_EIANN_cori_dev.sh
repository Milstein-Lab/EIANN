#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_EIANN_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /global/cscratch1/sd/aaronmil/logs/EIANN/"$JOB_NAME".%j.o
#SBATCH -e /global/cscratch1/sd/aaronmil/logs/EIANN/"$JOB_NAME".%j.e
#SBATCH -q debug
#SBATCH -N 32
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH --mail-user=neurosutras@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $HOME/EIANN/EIANN

export OMP_NUM_THREADS=1

srun -N 32 -n 1001 -c 2 --cpu-bind=cores python -m mpi4py.futures -m nested.optimize \
    --config-file-path=$CONFIG_FILE_PATH --disp --output-dir=$SCRATCH/data/EIANN \
    --pop_size=200 --max_iter=50 --path_length=3 --disp --framework=mpi
EOT
