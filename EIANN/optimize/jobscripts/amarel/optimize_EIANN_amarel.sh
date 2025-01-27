#!/bin/bash

export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_EIANN_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"

export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1

mkdir -p /scratch/${USER}/data/EIANN

sbatch <<EOT
#!/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch/${USER}/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e /scratch/${USER}/logs/EIANN/$JOB_NAME.%j.e
#SBATCH --partition=nonpre,p_am2820_1,genetics_1,main                 # Choose the appropriate partition (options: main, gpu, mem, cmain, cgpu, nonpre, graphical)
#SBATCH --requeue                                                     # Keep the job in the queue if it is preempted                   
#SBATCH --ntasks=251                                                  # Total number of cores/tasks across all nodes
#SBATCH --time=24:00:00                                               # Time limit for the job
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu                    # Email notifications
#SBATCH --mail-type=ALL                                               # Get email notifications on job start, end, and failure

set -x

cd $HOME/EIANN/EIANN

srun --mpi=pmi2 -n 251 python -m mpi4py.futures -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=/scratch/${USER}/data/EIANN --pop_size=200 --max_iter=50 --path_length=3 --disp \
  --framework=mpi
EOT


# Run with this:
# cd $HOME/EIANN/EIANN/optimize/jobscripts/amarel
# sbatch optimize_EIANN_amarel.sh optimize/optimize_config/spiral/nested_optimize_EIANN_2_hidden_BP_like_5J_fixed_bias.yaml spiral_BP_like_5J_fixed_bias

# See error and output logs with this:
# cd /scratch/$USER/logs/EIANN

# Delete all .out files in home directory with this:
# find . -maxdepth 1 -type f -name "*.out" -exec rm {} \;




# If you want to specify the number of nodes to run on, add this:
# #SBATCH --nodes=10

# Better way to do use mpi (doesn't work with current nested). Put after srun:
# --mpi=pmix