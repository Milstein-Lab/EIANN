#!/bin/bash

export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_EIANN_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"

mkdir -p /scratch/${USER}/data/EIANN

sbatch <<EOT
#!/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch/${USER}/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e /scratch/${USER}/logs/EIANN/$JOB_NAME.%j.e
#SBATCH -p main                                             # Choose the appropriate partition (options: main, gpu, mem, cmain, cgpu, nonpre, graphical)
#SBATCH --nodes=1                                           # Number of nodes
#SBATCH --ntasks=11                                         # Total number of cores/tasks across all nodes
#SBATCH --time=0:30:00                                      # Time limit for the job
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu          # Email notifications
#SBATCH --mail-type=ALL                                     # Get email notifications on job start, end, and failure

set -x

cd $HOME/EIANN/EIANN

srun --mpi=pmi2 -n 11 python -m mpi4py.futures -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=/scratch/${USER}/data/EIANN --pop_size=2 --max_iter=1 --path_length=1 --disp \
  --framework=mpi
EOT


# Run with this:
# cd $HOME/EIANN/EIANN/optimize/jobscripts/amarel
# sh optimize_EIANN_amarel.sh optimize/optimize_config/spiral/nested_optimize_EIANN_2_hidden_dend_EI_contrast_fixed_bias.yaml spiral_dend_EI_contrast

# See error and output logs with this:
# cd /scratch/$USER/logs/EIANN

# Delete all .out files in home directory with this:
# find . -maxdepth 1 -type f -name "*.out" -exec rm {} \;