#!/bin/bash
#PBS -N 1kx5_1aoi1
#PBS -k oe
#PBS -m be



module load NAMD/2.12-ibverbs
make-namd-nodelist

#charmrun ++nodelist ~/namd.$SLURM_JOBID ++p $SLURM_NTASKS `which namd2` +setcpuaffinity  min_equil.conf > ./output/min_equil.out
charmrun ++nodelist ~/namd.$SLURM_JOBID ++p $SLURM_NTASKS `which namd2` +setcpuaffinity  md_restart.conf > ./output/md6.out

# delete the NAMD-specific node list
rm ~/namd.$SLURM_JOBID

#submit job
#sbatch --partition=multinode --constraint=x2650 --ntasks=48 --ntasks-per-core=1 --time=168:00:00 --exclusive sub_job.sh
