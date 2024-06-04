#!/bin/bash

#SBATCH --job-name=LAMMPS
#SBATCH --output=LAMMPS_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<email_address>
#SBATCH --nodes=4                    # Number of nodes
#SBATCH --ntasks=8                   # Number of MPI ranks
#SBATCH --ntasks-per-node=2          # Number of MPI ranks per node
#SBATCH --ntasks-per-socket=1        # Number of tasks per processor socket on the node
#SBATCH --cpus-per-task=8            # Number of OpenMP threads for each MPI process/rank
#SBATCH --distribution=cyclic:cyclic # Distribute tasks cyclically on nodes and sockets
#SBATCH --mem-per-cpu=2000mb         # Per processor memory request
#SBATCH --time=4-00:00:00            # Walltime in hh:mm:ss or d-hh:mm:ss

date
hostname

module load intel/2018 openmpi

export OMP_NUM_THREADS=8

srun --mpi=pmix_v2 /path/to/app/lmp_gator2 < in.Cu.v.24nm.eq_xrd
