#!/bin/sh
#PBS -N gtcrun
#PBS -A csc190gtc
#PBS -l nodes=128
#PBS -l walltime=1:00:00
#PBS -q batch

# On titan, use queues 'debug' or 'batch'.

# Set number of jobs to be chained.  (To comply with
# ORNL policy, this should always be '1' for debug
# queue.)
job_number_max=1

: ${job_number:="1"}
: ${JOBDIR:=$PBS_O_WORKDIR}
cd $JOBDIR

if [[ ${job_number} -lt ${job_number_max} ]] 
then
  (( job_number++ ))
  qsub -v job_number=${job_number},JOBDIR -W depend=afterany:${PBS_JOBID} job-titan
fi

mkdir -p restart_dir1
mkdir -p restart_dir2
mkdir -p restart_dir

mkdir -p phi_dir
mkdir -p trackp_dir


# Set the number of OpenMP threads using
# system variable OMP_NUM_THREADS.  
export OMP_NUM_THREADS=2

aprun -n 1024 -N 8 -d 2 ./gtc -pc_type hypre -pc_hypre_type boomeramg
# aprun arguments:
#   -n  total number of MPI tasks
#   -N  number of MPI tasks per node
#   -d  number of OpenMP threads per MPI task
# notes:
#   (1) The number specified by the '-d' argument
#       should be the same as the value of system
#       variable 'OMP_NUM_THREADS'.
#   (2) For titan, there are 16 cpus per node.
#       The most efficient allocation of resources
#       is to use '-N 2 -d 8' for a total of 2*8=16
#       cores.
#   (3) The total number of nodes used is the number
#       of total MPI task / MPI tasks per node. 
#       For example, for 
#       
#           aprun -n 32 -N 2 -d 8 ./gtc
#         
#       the total number of nodes = 32/2 = 16.  
#       Once the '-n' and '-N' argments are fixed, 
#       you can fix the 'nodes' parameter at the 
#       top of the job file, which sends a request 
#       for that number of nodes, e.g.
#       
#           #PBS -l nodes=16
#       
#   (4) On titan, when using 8 OpenMP threads, 
#       '-n' argument should be a multiple of 2 so 
#       that the total number of cores is a multiple 
#       of 16.  This ensures that whole nodes are 
#       requested.

# increments 'irun' variable in gtc.in by 1, for chained jobs.
irun_n=`sed -n 's/^ *irun=\([0-9]*\).*/\1/p' gtc.in`
(( irun_m = irun_n + 1)) 
sed -i "s/^\( *\)irun=${irun_n}/\1irun=${irun_m}/" gtc.in

# uncomment the following line to rename gtc.out after every run to avoid overwriting
mv gtc.out gtc.out${irun_n}

# short delay 
sleep 3
