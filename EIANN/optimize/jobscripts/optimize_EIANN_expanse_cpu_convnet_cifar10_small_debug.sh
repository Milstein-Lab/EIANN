#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_EIANN_convnet_cifar10_debug_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
sbatch <<'EOT'
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o $SCRATCH/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e $SCRATCH/logs/EIANN/$JOB_NAME.%j.e
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=2
#SBATCH -t 0:30:00
#SBATCH --mem=249208M
#SBATCH --account=$ACCOUNT_NUMBER
#SBATCH --export=ALL
#SBATCH --mail-user=$MAIL_USER
#SBATCH --mail-type=ALL
#SBATCH --constraint="lustre"
#SBATCH --no-requeue

source $HOME/cpu_py311_intelmpi.sh

set -x

# one torch thread per MPI rank, and cap glibc malloc arenas: per-step tensor allocations in the CL rules (EWC, SI_HD)
# otherwise appear to bloat and fragment worker memory on Linux
export OMP_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MEM_PER_CPU=1946M

cd $PROJECT/EIANN/EIANN

srun -n 6 --mpi=pmi2 -c 2 --mem-per-cpu=$MEM_PER_CPU python -m mpi4py.futures -m nested.optimize \
  --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=$SCRATCH/data/EIANN --pop_size=1 --max_iter=1 --path_length=1 --disp \
  --framework=mpi --num_instances=5
EOT
