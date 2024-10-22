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
#SBATCH -p main                                             # Choose the appropriate partition, e.g., 'main', 'himem', 'gpu'
#SBATCH --nodes=1                                           # Number of nodes
#SBATCH --ntasks=20                                         # Total number of cores/tasks across all nodes
#SBATCH --time=1:00:00                                      # Time limit for the job
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu          # Email notifications
#SBATCH --mail-type=ALL                                     # Get email notifications on job start, end, and failure

set -x

cd /scratch/$USER/EIANN/EIANN

srun --mpi=pmi2 -n 20 python -m mpi4py.futures -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=/scratch/${USER}/data/EIANN --pop_size=19 --max_iter=1 --path_length=1 --disp \
  --framework=mpi
EOT


# Run with this:
# yc1376$ sbatch /home/$USER/EIANN/EIANN/optimize/jobscripts/amarel/optimize_EIANN_amarel.sh /home/$USER/EIANN/EIANN/optimize/optimize_config/spiral/nested_optimize_EIANN_2_hidden_dend_EI_contrast_fixed_bias.yaml spiral_dend_EI_contrast

# Go to scratch with:
# cd /scratch/yc1376