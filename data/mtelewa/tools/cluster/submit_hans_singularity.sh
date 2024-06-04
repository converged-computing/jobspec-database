#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=4:00:00
#SBATCH --partition=single
#SBATCH --output=cluster.out
#SBATCH --error=cluster.err
#SBATCH --job-name=hans
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mohamed.hassan@kit.edu

export KMP_AFFINITY=compact,1,0

module load compiler/intel/19.1
module load mpi/openmpi/4.0

mpirun --bind-to core --map-by core singularity exec --bind /scratch --bind /tmp --bind /pfs/work7/workspace/scratch/lr1762-flow --pwd=$PWD $HOME/programs/hans.sif python3 -m hans -i $(pwd)/channel1D_DH.yaml
