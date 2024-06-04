#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=72:00:00
#PBS -l nodes=10:GPU:ppn=12
#PBS -m e
#PBS -j oe
#PBS -M tekpinar@buffalo.edu
#PBS -o nvt_cMD_r1_to_t_pbs.out
#PBS -N nvt_cMD_r1_to_t
#PBS -q gpu
#
# Change to job submission directory
#

export CONV_RSH=ssh
#export CONV_DAEMON
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/util/namd/NAMD_2.8_Source-CUDA

cd $PBS_O_WORKDIR
#
# load namd via module

. $MODULESHOME/init/bash
module load namd/2.8-MPI-CUDA
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/util/namd/NAMD_2.8_Linux-x86_64-ibverbs-CUDA

#
# Set number of processors to number requested from PBS
#
NP=`cat $PBS_NODEFILE | wc -l`
echo "NAMDHOME = "$NAMDHOME
#
# construct namd nodes file from $PBS_NODEFILE
#
export NODELIST=tmp.$$
echo "group main" > $NODELIST
NODES=`cat $PBS_NODEFILE`
for node in $NODES ; do
   echo "host "$node >> $NODELIST
done

NAME_WORK_DIR=R1-T_transition
PATH_WORK_DIR=/panasas/scratch/tekpinar/hemoglobin/MD/${NAME_WORK_DIR}

#Run preparation perl script
#perl /user/tekpinar/perl_scripts/produce_md_input_files.pl ${PATH_WORK_DIR}

#echo "Minimization step:"
#CURRENTDIR=${PATH_WORK_DIR}/minimization
#date
#INFILE=${CURRENTDIR}/min.conf
#OUTFILE=${CURRENTDIR}/namd.log
#ERRFILE=${CURRENTDIR}/namd.err
##This version is supposed to run NAMD-CUDA on multiple GPU nodes. 
#charmrun ++remote-shell ssh ++verbose $NAMDHOME/namd2 +p$NP ++nodelist $NODELIST +idlepoll $INFILE  1>${OUTFILE} 2>${ERRFILE}
#date

#echo "Heating step:"
#CURRENTDIR=${PATH_WORK_DIR}/heating
#date
#INFILE=${CURRENTDIR}/heat.conf
#OUTFILE=${CURRENTDIR}/namd.log
#ERRFILE=${CURRENTDIR}/namd.err
##This version is supposed to run NAMD-CUDA on multiple GPU nodes. 
#charmrun ++remote-shell ssh ++verbose $NAMDHOME/namd2 +p$NP ++nodelist $NODELIST +idlepoll $INFILE  1>${OUTFILE} 2>${ERRFILE}
#date

echo "Equillibration step:"
CURRENTDIR=${PATH_WORK_DIR}/equillibration
date
INFILE=${CURRENTDIR}/equil.conf
OUTFILE=${CURRENTDIR}/namd.log
ERRFILE=${CURRENTDIR}/namd.err
#This version is supposed to run NAMD-CUDA on multiple GPU nodes. 
charmrun ++remote-shell ssh ++verbose $NAMDHOME/namd2 +p$NP ++nodelist $NODELIST +idlepoll $INFILE  1>${OUTFILE} 2>${ERRFILE}
date

echo "Production step for 'protein+water':"
CURRENTDIR=${PATH_WORK_DIR}/production
date
INFILE=${CURRENTDIR}/prod_nvt_cMD.conf
OUTFILE=${CURRENTDIR}/namd_nvt_cMD_r1_to_t_trj1.log
ERRFILE=${CURRENTDIR}/namd_nvt_cMD_r1_to_t_trj1.err
#This version is supposed to run NAMD-CUDA on multiple GPU nodes. 
charmrun ++remote-shell ssh ++verbose $NAMDHOME/namd2 +p$NP ++nodelist $NODELIST +idlepoll $INFILE  1>${OUTFILE} 2>${ERRFILE}
date

rm $NODELIST