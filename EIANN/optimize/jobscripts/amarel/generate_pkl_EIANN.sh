#!/bin/bash

export DATE=$(date +%Y%m%d_%H%M%S)
export MODEL_KEY="$4"
export JOB_NAME=generate_pkl_EIANN_"$MODEL_KEY"_"$DATE"
export CONFIG_FILE_PATH="$1"
export PARAM_FILE_PATH="$2"
export TASK="$3"

export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1

mkdir -p /scratch/${USER}/data/EIANN
mkdir -p $HOME/EIANN/EIANN/data/${TASK}

sbatch <<EOT
#!/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch/${USER}/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e /scratch/${USER}/logs/EIANN/$JOB_NAME.%j.e
#SBATCH --partition=nonpre,p_am2820_1,genetics_1,main                 # Choose the appropriate partition (options: main, gpu, mem, cmain, cgpu, nonpre, graphical)
#SBATCH --requeue                                                     # Keep the job in the queue if it is preempted                   
#SBATCH --ntasks=6                                                    # Total number of cores/tasks across all nodes
#SBATCH --time=2:00:00                                                # Time limit for the job
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu                    # Email notifications
#SBATCH --mail-type=ALL                                               # Get email notifications on job start, end, and failure

set -x

cd $HOME/EIANN/EIANN

srun --mpi=pmi2 -n 6 python -m mpi4py.futures -m nested.analyze --config-file-path=$CONFIG_FILE_PATH \
  --param-file-path=$PARAM_FILE_PATH --output-dir=data/${TASK} --model-key=$MODEL_KEY \
  --framework=mpi --status_bar=True --full_analysis --label=complete --store_history --export
EOT


# cd $HOME/EIANN/EIANN/optimize/jobscripts/amarel


# EIANN_0_hidden_spiral_vanBP_learned_bias
# sbatch generate_pkl_EIANN.sh optimize/optimize_config/spiral/20250108_nested_optimize_EIANN_0_hidden_spiral_van_bp_relu_learned_bias_config.yaml optimize/optimize_params/spiral/20250112_spiral_params.yaml spiral van_bp_0_hidden_learned_bias

# EIANN_2_hidden_spiral_vanBP_learned_bias
# sbatch generate_pkl_EIANN.sh optimize/optimize_config/spiral/20250108_nested_optimize_EIANN_2_hidden_spiral_van_bp_relu_learned_bias_config.yaml optimize/optimize_params/spiral/20250112_spiral_params.yaml spiral van_bp_learned_bias

# EIANN_2_hidden_spiral_vanBP_zero_bias
# sbatch generate_pkl_EIANN.sh optimize/optimize_config/spiral/20250108_nested_optimize_EIANN_2_hidden_spiral_van_bp_relu_zero_bias_config.yaml optimize/optimize_params/spiral/20250112_spiral_params.yaml spiral van_bp_zero_bias

# EIANN_2_hidden_spiral_bpDale_learned_bias
# sbatch generate_pkl_EIANN.sh optimize/optimize_config/spiral/20250108_nested_optimize_EIANN_2_hidden_spiral_bpDale_fixed_SomaI_learned_bias_config.yaml optimize/optimize_params/spiral/20250112_spiral_params.yaml spiral bpDale_learned_bias

# EIANN_2_hidden_spiral_BP_like_DTC_learned_bias
# sbatch generate_pkl_EIANN.sh optimize/optimize_config/spiral/20250108_nested_optimize_EIANN_2_hidden_spiral_BP_like_1_fixed_SomaI_learned_bias_config.yaml optimize/optimize_params/spiral/20250112_spiral_params.yaml spiral BP_like_1_fixed_SomaI_learned_bias

# EIANN_2_hidden_spiral_DTP_learned_bias
# sbatch generate_pkl_EIANN.sh optimize/optimize_config/spiral/20250108_nested_optimize_EIANN_2_hidden_spiral_DTP_fixed_SomaI_learned_bias_config.yaml optimize/optimize_params/spiral/20250112_spiral_params.yaml spiral DTP_fixed_SomaI_learned_bias


# See error and output logs with this:
# cd /scratch/$USER/logs/EIANN