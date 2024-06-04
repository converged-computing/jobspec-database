#!/bin/bash
#PBS -N JupyterLabPyTorch_Job
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=48gb:scratch_local=100gb:gpu_cap=^cuda80:cluster=^glados:cl_zubat=False
#PBS -l walltime=9:00:00
#PBS -m ae
# The 4 lines above are options for scheduling system: job will run 4 hours at maximum, 1 machine with 2 processors + 4gb RAM memory + 10gb scratch memory  are requested, email notification will be sent when the job aborts (a) or ends (e) 

## TODO: predefinovat ^^


# define variables
SING_IMAGE="/storage/brno2/home/mpicek/img.img"
HOMEDIR=/storage/praha1/home/$USER # substitute username and path to to your real username and path
HOSTNAME=`hostname -f`
PORT="8888"
IMAGE_BASE=`basename $SING_IMAGE`
export PYTHONUSERBASE=$HOMEDIR/.local-${IMAGE_BASE}

mkdir -p ${PYTHONUSERBASE}/lib/python3.6/site-packages 

#find nearest free port to listen
isfree=$(netstat -taln | grep $PORT)
while [[ -n "$isfree" ]]; do
    PORT=$[PORT+1]
    isfree=$(netstat -taln | grep $PORT)
done


# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

#set SINGULARITY variables for runtime data
export SINGULARITY_CACHEDIR=$HOMEDIR
export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR
export SINGULARITY_TMPDIR=$SCRATCHDIR
export SINGULARITYENV_PREPEND_PATH=$PYTHONUSERBASE/bin:$PATH


# move into $HOME directory
cd $HOMEDIR 
if [ ! -f ./.jupyter/jupyter_notebook_config.json ]; then
   echo "jupyter passwd reset!"
   mkdir -p .jupyter/
   #here you can commem=nt randomly generated password and set your password
   pass=`dd if=/dev/urandom count=1 2> /dev/null | uuencode -m - | sed -ne 2p | cut -c-12` ; echo $pass
   #pass="SecretPassWord" 
   hash=`singularity exec $SING_IMAGE python -c "from notebook.auth import passwd ; hash = passwd('$pass') ; print(hash)" 2>/dev/null`
   cat > .jupyter/jupyter_notebook_config.json << EOJson
{
  "NotebookApp": {
      "password": "$hash"
    }
}
EOJson
  PASS_MESSAGE="Your password was set to '$pass' (without ticks)."
else
  PASS_MESSAGE="Your password was already set before."
fi

# MAIL to user HOSTNAME
# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
#echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

EXECMAIL=`which mail`
$EXECMAIL -s "JupyterLab with PyTorch job is running on $HOSTNAME:$PORT" martinpicek4@gmail.com << EOFmail
Job with JupiterLab with PyTorch was started.

Use URL  http://$HOSTNAME:$PORT 

$PASS_MESSAGE

You can reset password by deleting file $HOMEDIR/.jupyter/jupyter_notebook_config.json and run job again with this script.
EOFmail

echo "http://$HOSTNAME:$PORT" > "/storage/praha1/home/mpicek/$(date +"%Y_%m_%d_%I_%M_%p").log" 

ls /cvmfs/singularity.metacentrum.cz/ > /dev/null #automount cvmfs

singularity exec --nv -H $HOMEDIR \
		 --bind /auto \
		 --bind /storage \
		 --bind $SCRATCHDIR \
		 $SING_IMAGE jupyter-lab --port $PORT --ip '*' --notebook-dir=/storage/budejovice1/home/mpicek/reCNN_visual_prosthesis\

# setting token with parameter  --NotebookApp.token=abcd123456
#singularity exec --nv -H $HOMEDIR $SING_IMAGE jupyter-lab --port $PORT --NotebookApp.token=abcd123456


# clean the SCRATCH directory
clean_scratch
