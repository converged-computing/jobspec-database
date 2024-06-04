##!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu -c 10
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH -A nstaff
#SBATCH -J cosmoflow-cgpu
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
## End SLURM directives and begin shell commands


# Run parameters
export BATCHSIZE=8
export DO_PROFILING='false'  # true or false
export DO_NCCL_DEBUG='false' # true or false
                             # only set at most one of DO_PROFILING,
                             # DO_NCCL_DEBUG to True

# Setup software environment
module purge
module load cgpu
module load tensorflow/2.4.1-gpu
export PYTHONPATH=/usr/common/software/tensorflow/2.4.1-gpu/bin/python
export NODES=${SLURM_NNODES}

#XLA environment
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_DIR


if [ "$DO_PROFILING" == "true" ]
then
    module load nsight-systems
    srun -N $SLURM_NNODES -n $((SLURM_NNODES*8)) -c 10 \
         --cpu-bind=cores \
         ./utils/run_with_profiling.sh
else
    srun -N $SLURM_NNODES -n $((SLURM_NNODES*8)) -c 10 \
         --cpu-bind=cores \
         ./utils/run.sh
fi
