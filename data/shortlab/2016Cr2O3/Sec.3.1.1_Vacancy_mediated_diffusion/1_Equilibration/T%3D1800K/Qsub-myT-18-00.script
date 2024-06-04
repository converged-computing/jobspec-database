#!/bin/bash 

    #PBS -l walltime=99:00:00
    #PBS -l nice=19
    #PBS -q default
    #PBS -l nodes=1:ppn=15
    cd $PBS_O_WORKDIR

mpirun ~/Install_Lammps/lammps-28Jun14/src/lmp_linux -i inNPT.18-00.CrO

echo "end"
