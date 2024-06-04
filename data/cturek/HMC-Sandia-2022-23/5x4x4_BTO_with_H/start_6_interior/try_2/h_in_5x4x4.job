#!/bin/bash
#SBATCH -p RM
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 12:00:00

#load the modules

module load intel/20.4
module load intelmpi/20.4-intel20.4
module load mkl/2020.4.304
module load allocations/1.0

cd /ocean/projects/mat220020p/pangreka/HMC-Sandia-2022-23/5x4x4_BTO_with_H/start_6_interior/try_2
mpirun /ocean/projects/mat220020p/pangreka/SeqQuest/Rbuild/allintel/dynamic_link/quest_268e.x
