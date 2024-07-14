#!/bin/bash
#
#-----------------------------------------------------------------------------
# Ethan Ho 4/13/2021
#
# Like Maverick2's /share/doc/slurm/job.jupyter, but runs jupyter-notebook in a
# Singularity container. The container must be saved as a SIF file, and its
# POSIX path should be specified by setting the SIF variable below.
#
# To submit the job, issue: sbatch scripts/jupyter-mav2.sh -i ./my_image.sif
#
# You can also provide a path to an env file that is sourced OUTSIDE the
# the container. For instance, to set the env variable SINGULARITYENV_SHELL, which
# is set as env variable SHELL inside the container, write an env file my_env.env:
#
# SINGULARITYENV_SHELL="/bin/bash"
#
# and pass the path to env file using the -e option:
#
# sbatch scripts/jupyter-mav2.sh -i ./my_image.sif -e my_env.env
#
# If the .sif file does not exist, the script will attempt to pull it from
# DockerHub, e.g.
#
# sbatch scripts/jupyter-mav2.sh -i ./my_image.sif -u docker://docker/whalesay:latest

# For more information, please consult the User Guide at:
# http://www.tacc.utexas.edu/user-services/user-guides/maverick2-user-guide
#-----------------------------------------------------------------------------
#
#SBATCH -J tvp_jupyter_sing           # Job name
#SBATCH -o jupyter.out                # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                          # Total number of nodes requested (56 cores/node)
#SBATCH -n 1                          # Total number of mpi tasks requested

# module configuration
echo "TACC: unloading xalt"
module unload xalt
module load cuda/10.1
module load cudnn nccl tacc-singularity
module list

#--------------------------------------------------------------------------
# ---- You normally should not need to edit anything below this point -----
#--------------------------------------------------------------------------

# getopts
OPTIND=1
SIF=
DOTENV=
URL=
while getopts "i:e:u:d:" opt; do
    case "$opt" in
    i)  SIF=$OPTARG
        ;;
    e)  DOTENV=$OPTARG
        ;;
    u)  URL=$OPTARG
        ;;
    d)  WD=$OPTARG
        ;;
    esac
done
shift $((OPTIND-1))
[ "${1:-}" = "--" ] && shift

# our node name
echo job $SLURM_JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "TACC: running on node $NODE_HOSTNAME"

# configure singularity runtime
if [ ! -z ${DOTENV} ] ; then
    if [ -f ${DOTENV} ]; then
        echo "using DOTENV=${DOTENV}"
        export $(grep -v '^#' ${DOTENV} | xargs | envsubst)
    else
        echo "could not find env file at DOTENV=${DOTENV}"
        exit 1
    fi
fi
[ -d $WD ] || WD=$PWD
SING_OPTS="--nv --home ${WD} --bind /work2"

# pull image if does not exist
echo "TACC: using SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR}"
if [ ! -f ${SIF} ]; then
    echo "TACC: pulling Singularity image to ${SIF}"
    singularity pull ${SIF} ${URL}
fi
[ ! -f ${SIF} ] && [ -z $URL ] && echo "Could not find image at ${SIF}, and no DockerHub URL passed" && exit 1

# entrypoint
echo "TACC: using singularity version $(singularity version)"
IPYTHON_BIN="singularity exec ${SING_OPTS} ${SIF} $@"
echo "TACC: using IPYTHON_BIN ${IPYTHON_BIN}"

NB_SERVERDIR=$HOME/.jupyter
IP_CONFIG=$NB_SERVERDIR/jupyter_notebook_config.py

# make .ipython dir for logs
mkdir -p $NB_SERVERDIR

rm -f $NB_SERVERDIR/.jupyter_address $NB_SERVERDIR/.jupyter_port $NB_SERVERDIR/.jupyter_status $NB_SERVERDIR/.jupyter_job_id $NB_SERVERDIR/.jupyter_job_start $NB_SERVERDIR/.jupyter_job_duration

# launch ipython
JUPYTER_LOGFILE=$NB_SERVERDIR/$NODE_HOSTNAME.log
JUP_CONFIG_PY="/home1/00832/envision/tacc-tvp/server/scripts/maverick2/jupyter.tvp.config.py"
IPYTHON_ARGS="--config=${JUP_CONFIG_PY}"
echo "TACC: using jupyter command: $IPYTHON_BIN $IPYTHON_ARGS"
nohup $IPYTHON_BIN $IPYTHON_ARGS &> $JUPYTER_LOGFILE && rm $NB_SERVERDIR/.jupyter_lock &
IPYTHON_PID=$!
echo "$NODE_HOSTNAME $IPYTHON_PID" > $NB_SERVERDIR/.jupyter_lock
echo "TACC: sleeping for 60 seconds..."
sleep 60
JUPYTER_TOKEN=`grep -m 1 'token=' $JUPYTER_LOGFILE | cut -d'?' -f 2`
LOCAL_IPY_PORT=5902
IPY_PORT_PREFIX=2
#echo "TACC: remote ipython port prefix is $IPY_PORT_PREFIX"

#  queues have row-node numbering, put login port in 50K block to avoid collisions with other machine ports
LOGIN_IPY_PORT="50`echo $NODE_HOSTNAME | perl -ne 'print $1.$2 if /c\d\d(\d)-\d(\d\d)/;'`"

echo "TACC: got login node jupyter port $LOGIN_IPY_PORT"

# create reverse tunnel port to login nodes.  Make one tunnel for each login so the user can just
# connect to maverick1.tacc
for i in `seq 1`; do
    ssh -q -f -g -N -R $LOGIN_IPY_PORT:$NODE_HOSTNAME:$LOCAL_IPY_PORT login$i
done
echo "TACC: created reverse ports on Maverick2 logins"

echo "Your jupyter notebook server is now running!"
echo "Please point your favorite web browser to https://vis.tacc.utexas.edu:$LOGIN_IPY_PORT/?$JUPYTER_TOKEN"

# Warn the user when their session is about to close
# see if the user set their own runtime
#TACC_RUNTIME=`qstat -j $JOB_ID | grep h_rt | perl -ne 'print $1 if /h_rt=(\d+)/'`  # qstat returns seconds
TACC_RUNTIME=`squeue -l -j $SLURM_JOB_ID | grep $SLURM_QUEUE | awk '{print $7}'` # squeue returns HH:MM:SS
if [ x"$TACC_RUNTIME" == "x" ]; then
	TACC_Q_RUNTIME=`sinfo -p $SLURM_QUEUE | grep -m 1 $SLURM_QUEUE | awk '{print $3}'`
	if [ x"$TACC_Q_RUNTIME" != "x" ]; then
		# pnav: this assumes format hh:dd:ss, will convert to seconds below
		#       if days are specified, this won't work
		TACC_RUNTIME=$TACC_Q_RUNTIME
	fi
fi

if [ "x$TACC_RUNTIME" != "x" ]; then
  # there's a runtime limit, so warn the user when the session will die
  # give 5 minute warning for runtimes > 5 minutes
	H=`echo $TACC_RUNTIME | awk -F: '{print $1}'`
        M=`echo $TACC_RUNTIME | awk -F: '{print $2}'`
        S=`echo $TACC_RUNTIME | awk -F: '{print $3}'`
        if $(echo $H | grep -q '\-'); then
            D=`echo $H | cut -d '-' -f 1`
            H=`echo $H | cut -d '-' -f 2`
        fi
        if [ "x$S" != "x" ]; then
            # full HH:MM:SS present
            H=$(($H * 3600))
            M=$(($M * 60))
            TACC_RUNTIME_SEC=$(($H + $M + $S))
        elif [ "x$M" != "x" ]; then
            # only HH:MM present, treat as MM:SS
            H=$(($H * 60))
            TACC_RUNTIME_SEC=$(($H + $M))
        else
            TACC_RUNTIME_SEC=$S
        fi
fi

# info for TACC Visualization Portal
echo "vis.tacc.utexas.edu" > $NB_SERVERDIR/.jupyter_address
# pnav: abuse ipython_port file for now so that the URL gets built correctly in webserver/website/job/job.php and entered into the "go" button in resources/resources.php
echo "$LOGIN_IPY_PORT/?$JUPYTER_TOKEN" > $NB_SERVERDIR/.jupyter_port
echo "$SLURM_JOB_ID" > $NB_SERVERDIR/.jupyter_job_id
# write job start time and duration (in seconds) to file
date +%s > $NB_SERVERDIR/.jupyter_job_start
echo "$TACC_RUNTIME_SEC" > $NB_SERVERDIR/.jupyter_job_duration

sleep 1
echo "success" > $NB_SERVERDIR/.jupyter_status

# spin on .ipython.lock file to keep job alive
while [ -f $NB_SERVERDIR/.jupyter_lock ]; do
  sleep 30
done


# job is done!

# wait a brief moment so ipython can clean up after itself
sleep 1

echo "TACC: job $SLURM_JOB_ID execution finished at: `date`"
