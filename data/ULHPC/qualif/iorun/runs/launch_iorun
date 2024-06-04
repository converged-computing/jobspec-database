#! /bin/bash
################################################################################
# launch_iorun - launch IORUN test on the UL HPC platform
# Time-stamp: <Thu 2014-01-28 19:42 hcartiaux>
#
# Copyright (c) 2014 Hyacinthe Cartiaux <Hyacinthe.Cartiaux@uni.lu>
################################################################################
#
# Submit this job in passive mode by
#
#   oarsub [options] -S ./launcher_iorun


##########################
#                        #
#   The OAR  directives  #
#                        #
##########################
#
#          Set number of resources
#

#OAR -l nodes=2,walltime=2

#          Set the name of the job (up to 15 characters,
#          no blank spaces, start with alphanumeric character)

#OAR -n IORUN

#          By default, the standard output and error streams are sent
#          to files in the current working directory with names:
#              OAR.%jobid%.stdout  <-  output stream
#              OAR.%jobid%.stderr  <-  error stream
#          where %job_id% is the job number assigned when the job is submitted.
#          Use the directives below to change the files to which the
#          standard output and error streams are sent, typically to a common file

#OAR -O IORUN-%jobid%.log
#OAR -E IORUN-%jobid%.log

#####################################

if [ -f  /etc/profile ]; then
    .  /etc/profile
fi

### Modules

module load mpi/OpenMPI

### Local variables

FILESIZE=20g
FILESYSTEM=$SCRATCH/

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCH_DATADIR="${SCRIPTDIR}/data"
OUTPUT_FILE="${BENCH_DATADIR}/iorun_`date +%Y-%m-%d-%H-%M`"

MACHINEFILE="/tmp/nodes_${OAR_JOB_ID}"
cat $OAR_NODEFILE | uniq > $MACHINEFILE

NODENUMBER=`wc -l $MACHINEFILE | cut -d' ' -f1`

mkdir -p $BENCH_DATADIR

for i in `seq 1 $NODENUMBER` ;do
  mpirun --map-by ppr:1:node  --mca plm_rsh_agent "oarsh"          \
         -machinefile $MACHINEFILE -n $i ./runs/ior -a POSIX -N $i \
         -b $FILESIZE -d 1 -t 128k -o $FILESYSTEM/iorun_bigfile    \
         -e -g -r -w -s 1 -i 1 -vv -F -C -k 2>&1 | tee -a $OUTPUT_FILE ;
done

