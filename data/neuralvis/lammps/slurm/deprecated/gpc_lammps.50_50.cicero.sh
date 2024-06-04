#!/bin/bash --login

###
# job name
#SBATCH --job-name=gpc_lammps.001
# specify its partition
#SBATCH --partition=iv24
# job stdout file
#SBATCH --output=stdout/gpc_lammps.001.%J.out
# job stderr file
#SBATCH --error=stdout/gpc_lammps.001.%J.err
# On failure, requeue for another try
# maximum job time in HH:MM:SS
#SBATCH --time=06:00:00
#SBATCH --nodes=154
# maximum memory
#SBATCH --mem-per-cpu=1024
# run a single task
#SBATCH --cpus-per-task=1
###

module load PrgEnv-cray
module load cray-mpich
module load perftools-base perftools

# Run gpcnet with 10 ppn
srun --exclusive -N 77 -n 770 \
     /home/users/msrinivasa/develop/GPCNET/network_load_test > gpc.out &

# wait till gpcnet primes up the network
sleep 10

# Run lammps with 4 ppn
srun --exclusive -N 77 -n 308 \
     /home/users/msrinivasa/develop/lammps/build/lmp+trace \
     -i /home/users/msrinivasa/develop/lammps/examples/DIFFUSE/in.msd.2d > lammps.out

# wait till all jobsteps finish
wait
