#!/bin/bash

NAME=lammps
upper_NAME=`echo ${NAME} | tr '[a-z]' '[A-Z]'`
NAME_MPI_EXE=/home/shormich/projects/fission/lammps/src/lammps-16Feb16/src/lammps
VERSION=28Jun14
VERSIONSUFFIX=
TOOLCHAIN=gmvolf-3.5.0
EXTRA_MODULE_NAME1=use.projects
EXTRA_MODULE_NAME2=use.easybuild
MODULE_NAME=${NAME}/${VERSION}-${TOOLCHAIN}${VERSIONSUFFIX}
#mvapich2
MPI_ENV_VAR='-genvlist PATH,LD_LIBRARY_PATH'
#openmpi
#MPI_ENV_VAR='-x PATH -x LD_LIBRARY_PATH -x DATAPATH'
#
#export DATAPATH=${DATAPATH:-"/hpc-common/data/mcnp/mcnpdata-1.0/MCNP_DATA"}
#
source /hpc-common/software/sub_scripts/common_sub_fns.sh

function usage ()
{
     echo
     echo "sub_${NAME}_${VERSION}"
     echo      "usage: sub_${NAME}_${VERSION} <-i inputfile> <-w HH:MM:SS> [<-n mpiprocs>] [<-q queue>] [<-N mpipernode>] [<-j PBS_job_name>] [<-m PBS_memory>] [<-h>]"
     echo      "Submits an ${NAME} version ${VERSION} job to the cluster via PBS Pro."
     echo      " Where:"
     echo      " -i inputfile  : Selects the Input file.( Default: none)"
     echo      " -w HH:MM:SS   : Selects the PBS Job Walltime.( Default: none)"
     echo      " -n mpiprocs   : Selects the number of parallel MPI processes to use.( Default: 1)"
     echo      " -q queue      : Selects the PBS queue to use.( Default: general)"
     echo      " -N mpipernode : Selects the maximum number of parallel mpi processes to use in each node.( Default: ${MPI_PER_NODE})"
     echo      " -j jobname    : Selects the name of PBS job.( Default: input file name)"
     echo      " -m PBS_memory : Selects the PBS Memory per Node.( Default: PBS defaults)"
     echo      " -h            : Selects help.( Default: none)"
     echo
     echo "NOTE: To add extra PBS options (e.g -M email_address -m be), use the environmental variable PBS_EXTRA_ARGS"
     echo 'e.g. export PBS_EXTRA_ARGS="-M cormac.garvey@inl.gov -m be"'
     echo '      To add extra lammps args/options, use the environmental variable LAMMPS_EXTRA_ARGS".'
     echo 'e.g. export LAMPS_EXTRA_ARGS="-l my.log"'
     echo
}


while getopts "i:w:n:q:N:j:m:hX" opt; do
  case $opt in
    i)
      INPUT=$OPTARG
      ;;
    w)
      WALLTIME=$OPTARG
      ;;
    n)
      MPIPROCS=$OPTARG
      ;;
    q)
      QUEUE=$OPTARG
      ;;
    N)
      MPI_PER_NODE=$OPTARG
      ;;
    j)
      JOBNAME=$OPTARG
      ;;
    m)
      PBS_MEM=$OPTARG
      ;;
    h)
      HELP=1
      usage
      exit 0
      ;;
    X)
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit
      ;;
  esac
done

if [ $HELP -ne 1 ]
then
   if [ -z "$INPUT" -o -z "$WALLTIME" ]
   then
      echo "Error: Make you specify the input file and PBS Walltime."
      usage
      exit 1
   fi
fi

if [ -n "$JOBNAME" ]
then
   JOBNAME=$JOBNAME
else
   JOBNAME=$INPUT
fi

calc_chunk

#echo "2 NNODES = $NNODES RPROCS = $RPROCS"
#echo "DATAPATH=" $DATAPATH

echo  "#!/bin/bash"                                                                                          > $INPUT.pbs
#echo  "#PBS -V"                                                                                             >> $INPUT.pbs
echo  "#PBS -N $JOBNAME"                                                                                    >> $INPUT.pbs
echo  "#PBS -j oe"                                                                                          >> $INPUT.pbs
if [ -n "$QUEUE" ]
then 
echo  "#PBS -q $QUEUE"                                                                                      >> $INPUT.pbs
fi
echo  "#PBS -ko"                                                                                            >> $INPUT.pbs
echo  "#PBS -l walltime=$WALLTIME"                                                                          >> $INPUT.pbs
if [  -n "$RPROCS2" ]
then
if [  -n "$PBS_MEM" ]
then
echo  "#PBS -l select=1:ncpus=$CORES_PER_NODE:mem=$PBS_MEM:mpiprocs=$RPROCS2+$NNODES:ncpus=$CORES_PER_NODE:mpiprocs=$RPROCS1:mem=$PBS_MEM" >> $INPUT.pbs
else
echo  "#PBS -l select=1:ncpus=$CORES_PER_NODE:mpiprocs=$RPROCS2+$NNODES:ncpus=$CORES_PER_NODE:mpiprocs=$RPROCS1" >> $INPUT.pbs
fi
else
if [  -n "$PBS_MEM" ]
then
echo  "#PBS -l select=$NNODES:ncpus=$CORES_PER_NODE:mpiprocs=$RPROCS1:mem=$PBS_MEM"                                      >> $INPUT.pbs
else
echo  "#PBS -l select=$NNODES:ncpus=$CORES_PER_NODE:mpiprocs=$RPROCS1"                                      >> $INPUT.pbs
fi
fi
echo  "#PBS -l place=excl"                                                                                  >> $INPUT.pbs
echo  "#"                                                                                                   >> $INPUT.pbs
echo  "#"                                                                                                   >> $INPUT.pbs
echo  "cat \$PBS_NODEFILE"                                                                                  >> $INPUT.pbs
echo  "source /etc/profile.d/modules.sh"                                                                    >> $INPUT.pbs
if [ -n "$EXTRA_MODULE_NAME1" ]
then
echo  "module load $EXTRA_MODULE_NAME1"                                                                     >> $INPUT.pbs
fi
if [ -n "$EXTRA_MODULE_NAME2" ]
then
echo  "module load $EXTRA_MODULE_NAME2"                                                                     >> $INPUT.pbs
fi
echo  "module load $MODULE_NAME"                                                                            >> $INPUT.pbs
echo  "#"                                                                                                   >> $INPUT.pbs
echo  "echo \"$STR1\""                                                                                      >> $INPUT.pbs
echo  "echo \"$STR2\""                                                                                      >> $INPUT.pbs
echo  "echo \"The following PBS extra args (PBS_EXTRA_ARGS) will be used: $PBS_EXTRA_ARGS\""                >> $INPUT.pbs
echo  "echo \"The following lammps extra args (LAMMPS_EXTRA_ARGS) will be used: $LAMMPS_EXTRA_ARGS\""       >> $INPUT.pbs
echo  "NPROCS=\`cat \$PBS_NODEFILE | wc -l\`"                                                               >> $INPUT.pbs
echo  "cd \$PBS_O_WORKDIR"                                                                                  >> $INPUT.pbs
echo  "JOB_NUM=\${PBS_JOBID%\.*}"                                                                           >> $INPUT.pbs
echo  "if [ \$PBS_O_WORKDIR != \$HOME ]"                                                                    >> $INPUT.pbs
echo  "then"                                                                                                >> $INPUT.pbs
echo  "ln -s \$HOME/\$PBS_JOBNAME.o\$JOB_NUM \$PBS_JOBNAME.o\$JOB_NUM"                                      >> $INPUT.pbs
echo  "fi"                                                                                                  >> $INPUT.pbs
echo  "#"                                                                                                   >> $INPUT.pbs
echo  "mpirun ${MPI_ENV_VAR} ${NAME_MPI_EXE} ${LAMPS_EXTRA_ARGS} -in \$PBS_JOBNAME "                        >> $INPUT.pbs
echo  "#"                                                                                                   >> $INPUT.pbs
echo  "if [ \$PBS_O_WORKDIR != \$HOME ]"                                                                    >> $INPUT.pbs
echo  "then"                                                                                                >> $INPUT.pbs
echo  "rm \$PBS_JOBNAME.o\$JOB_NUM"                                                                         >> $INPUT.pbs
echo  "mv \$HOME/\$PBS_JOBNAME.o\$JOB_NUM \$PBS_JOBNAME.o\$JOB_NUM"                                         >> $INPUT.pbs
echo  "fi"                                                                                                  >> $INPUT.pbs

qsub $PBS_EXTRA_ARGS $INPUT.pbs
