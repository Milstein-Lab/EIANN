#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_EIANN_mnist_MPI_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch2/11358/yashchennawar5555/logs/EIANN/$JOB_NAME.%j.o
#SBATCH -e /scratch2/11358/yashchennawar5555/logs/EIANN/$JOB_NAME.%j.e
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 2:00:00
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -x

source /work2/11358/yashchennawar5555/frontera/miniconda3/etc/profile.d/conda.sh
conda activate eiann7
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd $HOME/EIANN/EIANN

export MPI4PY_RC_RECV_MPROBE=false

ibrun -n 8 python -m mpi4py.futures -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=$SCRATCH/data/EIANN --pop_size=15 --max_iter=15 --path_length=3 --disp \
  --framework=mpi
EOT

# ./optimize_MPI_EIANN_frontera_MNIST.sh optimize/optimize_config/mnist/20231129_nested_optimize_EIANN_2_hidden_mnist_van_bp_relu_SGD_config_G.yaml van_bp
