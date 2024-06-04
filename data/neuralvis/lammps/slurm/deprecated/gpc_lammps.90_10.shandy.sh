#!/bin/bash --login

###
# job name
#SBATCH --job-name=gpc_lammps.001
# specify its partition
#SBATCH --partition=workq
# job stdout file
#SBATCH --output=stdout/gpc_lammps.001.%J.out
# job stderr file
#SBATCH --error=stdout/gpc_lammps.001.%J.err
# On failure, requeue for another try
# maximum job time in HH:MM:SS
#SBATCH --time=06:00:00
#SBATCH --nodes=927
# maximum memory
#SBATCH --mem-per-cpu=512
# run a single task
#SBATCH --cpus-per-task=1
###

module restore PrgEnv-cray
module load cray-mpich/8.0.15

# Run gpcnet on 90% of allocation with 10 ppn
# srun --exclusive -N 834  -n 8340 \
#   /home/users/msrinivasa/develop/GPCNET/network_load_test > gpc.out &

# wait till gpcnet primes up the network
# sleep 60

# Run lammps with 4 ppn
srun --exclusive -N 93 -n 372 \
     /home/users/msrinivasa/develop/lammps/build/lmp+tracing \
     -i /home/users/msrinivasa/develop/lammps/examples/DIFFUSE/in.msd.2d > lammps.out

# wait till all jobsteps finish
# wait

# sbatch cmd on Shandy that works [October 6, 2020]
# sbatch --exclude=nid001421,nid001382,nid001297,nid001260,nid001464 gpc_lammps.90_10.shandy.sh
