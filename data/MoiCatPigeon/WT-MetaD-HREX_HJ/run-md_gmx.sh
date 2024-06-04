#!/bin/bash
#PBS Job
#PBS -V
#PBS -N HJ_isoi_MetaD_i
### #PBS -l nodes=1:ppn=1:gpus=1:exclusive_process
### #PBS -l nodes=x02:ppn=:gpus=1:default
#PBS -l nodes=x08:ppn=24:gpus=3:default
### #PBS -l nodes=1:ppn=16
#

cat $PBS_GPUFILE

export CUDA_VISIBLE_DEVICES=0,1,2,3

module purge

# OpenMPI module load
module load openmpi-3.1.3
#module load plumed-2.5.1


module load plumed-2.5.6-switch
source /usr/local/programs/common/Gromacs2018_Plumed256_switch/gromacs-2018.8/mpi-cuda/bin/GMXRC.bash
source /usr/local/programs/common/intel/compilers_and_libraries_2019.5.281/linux/mkl/bin/mklvars.sh intel64


# Sourcing of the GMXRC for the desired Gromacs version
# (sets up the environment appropriately)
#source /usr/local/programs/common/Gromacs_Plumed-2.5.1/gromacs-2018.6/mpi-cuda/bin/GMXRC

# this is to go in the right directory
cd $PBS_O_WORKDIR

# echo $CUDA_VISIBLE_DEVICES

export PARNODES=`wc -l $PBS_NODEFILE |gawk '{print $1}'`
export HOSTS_FILE=$PBS_NODEFILE

#
# Jobs start here (uncomment and change)
#
cat $HOSTS_FILE>hosts_file

#############################################

JOB=HJ_isoi_MetaD_i
home_dir=$PWD

uname -n
date
echo ""

mkdir /scratch/$USER/$JOB
cp * /scratch/$USER/$JOB
cd /scratch/$USER/$JOB

### Will create /scratch directories but submitting command run from home --all data copied to /home directory ###

# for node in `awk '!array[$1]++ && $1 ~ /^x01$|^x02$|^x03$|^x04$|^x05$|^x06$|^x07$|^x08$|^x09$/' $HOSTS_FILE`
# do
#   ssh $node "mkdir /scratch/${USER}/${JOB}"
#   scp * $node:/scratch/${USER}/${JOB}
# done

# Run of moleclar dynamics simulation.

export OMP_NUM_THREADS=4
# export PLUMED_NUM_THREADS=8

# mpirun -x PLUMED_KERNEL --nooversubscribe $GMXBIN/gmx_omp_mpi_gpu mdrun -v -s topol.tpr -plumed plumed.dat -replex 120 -multi ${NSLOTS} -ntomp 4 -nb gpu -nsteps $NSTEPS -pin on
# mpirun gmx_mpi mdrun -v -s complex-1.tpr -o md.trr -x md.xtc -c npt_box.gro -e md.edr -ntomp 2 -nb gpu -pin on -nsteps 10000 -maxh 0.9
# mpirun -npernode 6 gmx_mpi mdrun -v -s complex-mRNA-BzCN_.tpr -plumed plumed.dat -o md_.trr -x md_.xtc -c npt_box.gro -e md.edr -nb gpu -pin on -multi 18 -replex 500 -ntomp 5 -nsteps 10000 -hrex -gpu_id 001122
# mpirun -np 4 -x PLUMED_KERNEL --nooversubscribe gmx_mpi mdrun -v -s complex-1.tpr -plumed plumed.dat -o md.trr -x md.xtc -c npt_box.gro -e md.edr -pin on -nsteps 20000 -maxh 0.9


# Run 10 rounds, 10x100 ns
START=0
ALL=10

# Test rerun first for 20 ps
while [ $START -lt $ALL ]
do
	if [[ $START -eq 0 ]]
	then
		mpirun -np 6 -x PLUMED_KERNEL --nooversubscribe gmx_mpi mdrun -v -s HJ_isoi -x HJ_isoi -plumed preplumed.dat -multi 6 -nb gpu -ntomp 4 -pin on -replex 2500 -nsteps 25000000 -hrex -dlb no
		mv state0.cpt checkpoint0.cpt
                mv state1.cpt checkpoint1.cpt
                mv state2.cpt checkpoint2.cpt
                mv state3.cpt checkpoint3.cpt
                mv state4.cpt checkpoint4.cpt
                mv state5.cpt checkpoint5.cpt
	else
		mpirun -np 6 -x PLUMED_KERNEL --nooversubscribe gmx_mpi mdrun -v -s HJ_isoi -x HJ_isoi -plumed plumed.dat -cpi checkpoint -noappend -multi 6 -nb gpu -ntomp 4 -pin on -replex 2500 -nsteps 25000000 -hrex -dlb no
		mv state0.cpt checkpoint0.cpt
		mv state1.cpt checkpoint1.cpt
		mv state2.cpt checkpoint2.cpt
		mv state3.cpt checkpoint3.cpt
		mv state4.cpt checkpoint4.cpt
		mv state5.cpt checkpoint5.cpt
	fi
	START=$(( $START + 1 ))
done


