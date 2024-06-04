#!/bin/bash -l
# Allocate slurm resources, edit as necessary
#SBATCH --account=pawsey0219
#SBATCH --partition=work
#SBATCH --ntasks=8
#SBATCH --time=08:00:00
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=LOGS/jupyter-%j.out
#SBATCH --export=NONE
 
# Set our working directory
# This is the directory we'll mount to /home/jovyan in the container
# Should be in a writable path with some space, like /scratch
dir=$1 #"${MYSCRATCH}/"
 
# Set the image and tag we want to use
#image="docker://jupyter/datascience-notebook:latest"
image="docker://mrayson/jupyter-sfoda:latest"
 
# You should not need to edit the lines below
 
# Prepare the working directory
mkdir -p ${dir}
cd ${dir}
 
# Get the image filename
imagename=${image##*/}
imagename=${imagename/:/_}.sif
 
# Get the hostname of the Zeus node
# We'll set up an SSH tunnel to connect to the Juypter notebook server
host=$(hostname)
 
# Set the port for the SSH tunnel
# This part of the script uses a loop to search for available ports on the node;
# this will allow multiple instances of GUI servers to be run from the same host node
port="8888"
pfound="0"
while [ $port -lt 65535 ] ; do
  check=$( netstat -tuna | awk '{print $4}' | grep ":$port *" )
  if [ "$check" == "" ] ; then
    pfound="1"
    break
  fi
  : $((++port))
done
if [ $pfound -eq 0 ] ; then
  echo "No available communication port found to establish the SSH tunnel."
  echo "Try again later. Exiting."
  exit
fi
 
# Load Singularity
module load singularity/3.8.6
 
# Pull our image in a folder
singularity pull $imagename $image
echo $imagename $image
 
echo "*****************************************************"
echo "Setup - from your laptop do:"
echo "ssh -N -f -L ${port}:${host}:${port} $USER@$PAWSEY_CLUSTER.pawsey.org.au"
echo "*****"
echo "The launch directory is: $dir"
echo "*****************************************************"
echo ""
echo "*****************************************************"
echo "Terminate - from your laptop do:"
echo "kill \$( ps x | grep 'ssh.*-L *${port}:${host}:${port}' | awk '{print \$1}' )"
echo "*****************************************************"
echo ""
  
#singularity exec $imagename python -c "import sfoda.utils.timeseries;print('done')"
## Launch our container
## and mount our working directory to /home/jovyan in the container
## and bind the run time directory to our home directory
singularity exec -C \
  -B ${dir}:/home/joyvan \
  -B ${dir}:${HOME} \
  ${imagename} \
  jupyter notebook \
  --no-browser \
  --port=${port} --ip=0.0.0.0 \
  --notebook-dir=${dir}
