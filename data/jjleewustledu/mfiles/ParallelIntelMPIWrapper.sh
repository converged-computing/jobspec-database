#!/bin/sh
#PBS -v MDCE_DECODE_FUNCTION,MDCE_STORAGE_LOCATION,MDCE_STORAGE_CONSTRUCTOR,MDCE_JOB_LOCATION,MDCE_CMR,MDCE_MATLAB_EXE,MDCE_MATLAB_ARGS,MDCE_TOTAL_TASKS,MDCE_DEBUG
# The previous line ensures that all required environment variables are
# forwarded from the client MATLAB session through the scheduler to the cluster.
# 
# This script uses the following environment variables set by the code in the submit function:
# MDCE_CMR            - the value of ClusterMatlabRoot (may be empty)
# MDCE_MATLAB_EXE     - the MATLAB executable to use
# MDCE_MATLAB_ARGS    - the MATLAB args to use
#
# The following environment variables are forwarded through mpiexec:
# MDCE_DECODE_FUNCTION     - the decode function to use
# MDCE_STORAGE_LOCATION    - used by decode function 
# MDCE_STORAGE_CONSTRUCTOR - used by decode function 
# MDCE_JOB_LOCATION        - used by decode function 

# Copyright 2006-2009 The MathWorks, Inc.
# $Revision: 1.1.6.1 $   $Date: 2010/05/10 17:05:00 $

PBS_O_WORKDIR=$MDCE_STORAGE_LOCATION/$MDCE_JOB_LOCATION
cd $PBS_O_WORKDIR

MPI_PATH=/export/intel/Compiler/12.0/impi/4.0.1.007/bin64

MPIEXEC=/export/src/mpiexec-0.84/mpiexec

UNIQ_HOSTS=hosts.$$
cat $PBS_NODEFILE | uniq > $UNIQ_HOSTS
NNuniq=`cat $UNIQ_HOSTS | wc -l`

MPIEXEC_CODE=0
MDCE_PARALLEL=1
ulimit -s unlimited

# Work out how many processes to launch - set MACHINE_ARG
chooseMachineArg() {
 MACHINE_ARG="-n ${MDCE_TOTAL_TASKS}"
}

runMpiexec() {
    
    echo "export I_MPI_PMI_EXTENSIONS=on"
    eval "export I_MPI_PMI_EXTENSIONS=on"

    echo \"${MPIEXEC}\" -comm=mpich2-pmi \
     \"${MDCE_MATLAB_EXE}\" ${MDCE_MATLAB_ARGS} 

    eval \"${MPIEXEC}\" -comm=mpich2-pmi \
       \"${MDCE_MATLAB_EXE}\" ${MDCE_MATLAB_ARGS} 

    MPIEXEC_CODE=${?}
}


# Define the order in which we execute the stages defined above
MAIN() {
    chooseMachineArg
    runMpiexec
    exit ${MPIEXEC_CODE}
}

# Call the MAIN loop
MAIN
