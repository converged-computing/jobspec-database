#!/bin/bash
### Set the job name
#PBS -N gen3dlayout

### Request email when job begins and ends
#PBS -m bea

### Specify email address to use for notification.
#PBS -M angl@umd.edu 

### Set the queue for this job as windfall or standard (adjust ### and #)
##PBS -q longnarrow
##PBS -q matlab-long
#PBS -q octaque
##PBS -q matlab
##PBS -q dque
##PBS -q matlab-wide
##PBS -q shortwide

### Set the jobtype for this job (serial, small_mpi, small_smp, large_mpi, large_smp)
### jobtype=serial submits to htc and can be automatically moved to cluster and smp
### Type parameter determines initial queue placement and possible automatic queue moves

### Set the number of cores (cpus) and memory that will be used for this job
### When specifying memory request slightly less than 2GB memory per ncpus for standard node
### Some memory needs to be reserved for the Linux system processes
#PBS -l nodes=8:ppn=8

### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=08:00:00

### Specify total cpu time required for this job, hhh:mm:ss
### total cputime = walltime * ncpus

### Load required modules/libraries if needed (blas example)
### Use "module avail" command to list all available modules
### NOTE: /usr/share/Modules/init/csh -CAPITAL M in Modules

### cd: set directory for job execution, ~netid = home directory path
### executable command with trailing & - "

# This finds out the number of nodes we have

NP=$(wc -l < $PBS_NODEFILE)
echo "Total CPU count = $NP"

export jobname="main_3dsolver_euclid"
shellpath="/gleuclid/angli/layout-retrieval/src/3dsolver/generic-matlab-subjob.sh"

if [ -z "$shellpath" ]; then
	echo "shellpath unassigned"
	exit
fi

if [ -z "$jobname" ]; then
	jobname=unnamed
fi

WORKDIR="/gleuclid/angli/layout-retrieval/src/3dsolver"
if [ ! -d $WORKDIR/pbs ]; then
	mkdir $WORKDIR/pbs
	echo $WORKDIR/pbs created
fi

if [ ! -d $WORKDIR/pbs/$jobname ]; then
	mkdir $WORKDIR/pbs/$jobname
	echo $WORKDIR/pbs/$jobname created
else
	rm -rf $WORKDIR/pbs/$jobname/$jobname.o*
	echo removed all under $WORKDIR/pbs/$jobname
fi

pbsdsh -v bash $shellpath $NP $jobname
