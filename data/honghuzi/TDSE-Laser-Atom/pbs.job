#!/bin/bash -l

##set the job name.
#PBS -N cyb_two


##set the queue on which you will run your job
#PBS -q small

##set the max run time of your job, the max value is 240hours.
#PBS -l walltime=240:00:00

## set memory requirement of your job, max memeory is 64gb
#PBS -l mem=64gb

##set num of nodes your job needs and process num of each node.
##you only need to set the nodes depend on your job
#PBS -l nodes=1:ppn=16

##load modules if your job needs, modify the module name and uncomment the command
#module load /public/software/modules/mpi/intelmpi/5.0.2.044  /public/software/modules/compiler/intel/composer_xe_2015.2.164

#PBS -o po
#PBS -e pe

#ulimit -s unlimited
#ulimit -l unlimited

##export I_MPI_PROCESS_MANAGER=mpd
#export FOR_COARRAY_NUM_IMAGES=8
## enter the pbs work directory
cd $PBS_O_WORKDIR

##set the programm path of your job
#exe=/home/laser/chenyinbo/qtmcdi/qtmcdi.x
exe=julia
#nodes assigned by pbs
##hostfile=$PBS_NODEFILE

#calculate the node num assigned by pbs
##hostnums=`cat $hostfile | wc -l`

##attention modify this command to your programm command
##you can use other command instead of mpirun

#mpirun -np $hostnums -hostfile $hostfile $exe
#$exe --procs auto --check-bounds=no --depwarn=no --math-mode=fast -O3 tdse.jl 1>>output 2>>err
./tdse.x  1>>output 2>>err