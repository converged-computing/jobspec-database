#!/bin/bash -l

#SBATCH --partition=workq
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=3:00:00
#SBATCH --account=pawsey0315
#SBATCH --export=NONE
#SBATCH -J jupyter   # name
#SBATCH -o jupyter-%J.out

module load singularity

# Create trap to kill notebook when user is done
kill_server() {
    if [[ $JNPID != -1 ]]; then
        echo -en "\nKilling Jupyter Notebook Server with PID=$JNPID ... "
        kill $JNPID
        echo -e "Done\n"
        exit 0
    else
        exit 1
    fi
}

# Group Singularity image repository -----
# Let's put shared Singularity images here
groupRepository=/group/$PAWSEY_PROJECT/singularity/groupRepository

# Users working directory -----
# This is the directory we'll mount to /home/jovyan in the container
# Must be under /group or /scratch
userDirectory="${MYSCRATCH}/sandpit"
mkdir -p $userDirectory
cd $userDirectory

# Set the image and tag we want to use ----
# For now, using Richard's mod of docker://pangeo-notebook-onbuild
# image="docker://pangeo-notebook-onbuild:mod1"
image="docker://matear/pangeo-mac:mod1"

# Get the image filename -----
imagename=${image##*/}
imagename=$groupRepository/${imagename/:/_}.sif

# Pull/build the container if it doesn't already exist in the groupRepository -----
# When building singularity images, many files and a copy of the images themselves are saved in the cache. 
# Singularity modules at Pawsey define the Singularity image cache at $MYGROUP/.singularity/cache, contrary 
# to the Singularity default that defines the cache in /home. This avoids problems with the restricted 
# quota of /home. On other systems, e.g. Pearcey, it may be necessary to symlink $HOME/.singularity to a 
# (non-NFS) location with more space, e.g. /scratch1 or /flush# on Pearcey.
# Once you have finished building/pulling containers, you can clean up the cache using:
#     singularity cache clean -a
if [ -f $imagename ]; then
    echo -e "Using existing build of ${image} at ${imagename}\n"
else
    echo "Pulling and building ${image}..."
    singularity pull $imagename $image
    echo -e "Done\n"
fi

# Start the dask-scheduler -----
# The scheduler process and worker process(es) communicate know to communicate through
# $MYSCRATCH/scheduler.json
echo "Starting dask scheduler..."
#srun --export=ALL -n 1 -N 1 -c $SLURM_CPUS_PER_TASK \
singularity exec -C -B ${userDirectory}:/home/joyvan -B ${userDirectory}:$HOME ${imagename} \
dask-scheduler --scheduler-file $MYSCRATCH/scheduler.json --idle-timeout 0 &
sleep 20
echo -e "Done\n"

# Search for available ports on the node -----
# This will allow multiple instances of GUI servers to be run from the same host node
let DASK_PORT=8787
PORT=8888
pfound="0"
while [ $PORT -lt 65535 ] ; do
    check=$( netstat -tuna | awk '{print $4}' | grep ":$PORT *" )
    if [ "$check" == "" ] ; then
      pfound="1"
      break
    fi
    : $((++PORT))
done
if [ $pfound -eq 0 ] ; then
  echo "No available communication port found to establish the SSH tunnel."
  echo "Try again later. Exiting."
  exit
fi
let LOCALHOST_PORT=$PORT

# Start the Jupyter notebook -----
HOST=$(hostname)
HOSTIP=$(hostname -i)
logDirectory="$MYSCRATCH/logs"
mkdir -p $logDirectory
LOGFILE=$logDirectory/pangeo_jupyter_log.$(date +%Y%m%dT%H%M%S)
echo -e "Starting jupyter notebook (logging jupyter notebook session on ${HOST} to ${LOGFILE})...\n"
#srun --export=ALL -n 1 -N 1 -c $SLURM_CPUS_PER_TASK \
singularity exec -C -B ${userDirectory}:/home/joyvan -B ${userDirectory}:$HOME ${imagename} \
jupyter lab --no-browser --ip=$HOST --notebook-dir=${userDirectory} \
>& $LOGFILE &
JNPID=$!

# Wait for notebook to start
ELAPSED=0
ADDRESS=
while [[ $ADDRESS != *"${HOST}"* ]]; do
    sleep 1
    ELAPSED=$(($ELAPSED+1))
    ADDRESS=$(grep -e '^\[.*\]\s*http://.*:.*/\?token=.*' $LOGFILE | head -n 1 | awk -F'//' '{print $NF}')
    if [[ $ELAPSED -gt 360 ]]; then
        echo -e "Something went wrong:\n-----"
        cat $LOGFILE
        echo "-----"
        kill_server
    fi
done

# Print jupyter port forwarding info -----i
PORT=$(echo $ADDRESS | awk -F':' ' { print $2 } ' | awk -F'/' ' { print $1 } ')
TOKEN=$(echo $ADDRESS | awk -F'=' ' { print $NF } ')
cat << EOF
  Run the following command on your local computer:
  
    ssh -N -l $USER -L ${LOCALHOST_PORT}:${HOST}:$PORT zeus.pawsey.org.au

  Log in with your Username/Password or SSH keys.
  Then open a browser and go to:

    Jupyter notebook : http://localhost:${LOCALHOST_PORT} 
    Dask dashboard : http://localhost:${LOCALHOST_PORT}/proxy/${DASK_PORT}/status

  The Jupyter web interface will ask you for a token. Use the following:

    $TOKEN

  Note that anyone to whom you give the token can access (and modify/delete)
  files in your PAWSEY spaces, regardless of the file permissions you
  have set. SHARE TOKENS RARELY AND WISELY!
  To stop the server, press Ctrl-C.
EOF

# Wait for user kill command -----
sleep inf
