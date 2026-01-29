#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=export_optimized_EIANN_mnist_"$DATE"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /ocean/projects/bio250022p/chennawa/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e /ocean/projects/bio250022p/chennawa/logs/EIANN/$JOB_NAME.%j.e
#SBATCH -p RM-512
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -n 32
#SBATCH -t 24:00:00
#SBATCH -A bio250022p
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -x

cd ~/EIANN/EIANN

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate eiann5

export MPI4PY_RC_RECV_MPROBE=false

# Use \$CONDA_PREFIX to ensure we use the mpirun from the conda env, not the system.
\$CONDA_PREFIX/bin/mpirun -n 32 python -m mpi4py.futures -m nested.analyze \
    --config-file-path=optimize/optimize_config/mnist/20231129_nested_optimize_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G.yaml \
    --output-dir=data/mnist --disp \
    --num_instances=31 \
    --status_bar --framework=mpi
EOT

# Run with: 
# cd ~/EIANN/EIANN/simulate/jobscripts
# ./run_EIANN_cpu_bridges_mnist_mpi.sh

# See logs:
# cd /ocean/projects/bio250022p/$USER/logs/EIANN

# On interactive:
# interact -p RM -N 1 -t 60:00 --ntasks-per-node=64
# cd ~/EIANN/EIANN
# mpirun -n 64 -genv MPI4PY_RC_RECV_MPROBE false python -m mpi4py.futures -m nested.analyze --config-file-path=optimize/optimize_config/mnist/20231129_nested_optimize_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G.yaml --disp --num_instances=63 --status_bar --output-dir=data/mnist --framework=mpi