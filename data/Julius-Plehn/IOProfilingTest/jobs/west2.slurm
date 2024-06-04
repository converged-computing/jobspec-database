#!/bin/bash
	
# Time limit is one minute. See "man sbatch" for other time formats.
#SBATCH --time=02:00:00
# Run a total of ten tasks on two nodes (that is, five tasks per node).
#SBATCH --nodes=2
#SBATCH --ntasks=2
# Use "west" partition.
#SBATCH --partition=west
# Output goes to "job.out", error messages to "job.err".
#SBATCH --output=west2.out
#SBATCH --error=west2.err

. /etc/profile.d/modules.sh
. /etc/profile.d/wr-spack.sh
spack load --dependencies mpi

if [ "${SLURM_PARTITION}" = 'abu' ]
then
	export MPICH_NEMESIS_NETMOD=ib
fi

mpiexec ../writefile 1000000000 1 3
mpiexec ../writefile 1000000000 2 3