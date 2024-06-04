#!/bin/bash
#PBS -N app
#PBS -q cheminf
#PBS -l select=1:ncpus=8:ngpus=1:mem=8gb
#PBS -l walltime=720:00:00
#PBS -m ae

[ -z "$WORKDIR" ] && echo "\$WORKDIR is empty. Exiting..." && exit 1
[ -z "$CONDA_ENV_PATH" ] && echo "\$CONDA_ENV_PATH is empty. Exiting..." && exit 1
[ -z "$REMOTE_HOST" ] && echo "\$REMOTE_HOST is empty. Exiting..." && exit 1

export SCRATCHDIR=/scratch/$USER/ # this might be useful -> we can get this variable from python and fetch big data there

# go to the working directory
cd $WORKDIR

# append a line to a file "jobs_info.txt" containing the ID of the job and the current worker hostname
echo "$PBS_JOBID is running on node `hostname -f`." >> $WORKDIR/jobs_info.txt

# activate the conda environment
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate $CONDA_ENV_PATH
nvidia-smi > nvidia_inf.txt
nvcc --version > cuda_inf.txt

# run the server and bind it to our remote host
if [ -z "$REMOTE_HOST" ]
then
    echo "\$REMOTE_HOST not specified. Exiting..." && exit 1
else
    PORT="${PORT:-8000}" # port for the service on both remote and local host

    # run our jupyter service locally in the background
    stdbuf -oL nohup streamlit run webapp.py > $WORKDIR/${PBS_JOBID}_app.log 2>&1 &
    SERVER_PID=$! # get the background process id for later
    echo "streamlit server started with process ID: $SERVER_PID"


    # if a terminating signal is received (https://www.shellscript.sh/trap.html), kill the server
    trap "kill -SIGKILL ${SERVER_PID} && echo 'Session interrupted. Service was killed.'" 1 2 3 6 15 # if a terminating signal received (https://www.shellscript.sh/trap.html), kill the running service 

    # tunnel our localhost service to the REMOTE_HOST and keep the connection up for as long as it lasts
    ssh -R $PORT:localhost:$PORT $REMOTE_HOST 'sleep infinity' # bind our localhost to the REMOTE_HOST and keep the connection up for as long as it lasts

    # kill the local service if the connection is lost or interrupted
    # we might want to skip this if we need the local session to outlive the ssh tunnel
    # kill -SIGKILL ${SERVER_PID} && echo 'We lost connection to the remote host. Jupyter service was killed.'
fi