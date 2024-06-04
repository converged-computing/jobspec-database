#!/bin/sh

### The following requests a single GPU
#PBS -l select=1:ncpus=5:ngpus=1

### The "select=1" specifies the number of nodes
### To ask for a whle node add "ncpus=40:ngpus=8"
### If you request less than 8 GPU then make the ncpus value
###   five times the ngpus value, e.g. select=1:ncpus=5:ngpus=1

### Specify amount of time required (maximum 24 hours)
#PBS -l walltime=0:05:00

### Select correct queue
#PBS -q dgx

### Specify project code
#PBS -P 99999999

### Specify name for job
#PBS -N pip

### Standard output by default goes to file $PBS_JOBNAME.o$PBS_JOBID
### Standard error by default goes to file $PBS_JOBNAME.e$PBS_JOBID
### To merge standard output and error use the following
#PBS -j oe

### Start of commands to be run

# Docker image to use for container
#   To see available images run command: nscc-docker images
#   If image is not present, email help@nscc.sg to request pulling image into repository on all DGX nodes
image="nvcr.io/nvidia/tensorflow:latest"

# Change to directory where job was submitted
cd "$PBS_O_WORKDIR" || exit $?

# The "nscc-docker run $image" command runs the following Docker command: 
#    nvidia-docker run -u UID:GID --group-add PROJECT \
#      -v /home:/home -v /scratch:/scratch -v /raid:/raid \
#      --rm -i --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 $image /bin/sh
# See full list of options with "nscc-docker run -h"
# Pass the commands that you wish to run inside the container on the standard input
nscc-docker run $image << EOF
hostname
nvidia-smi
pwd
# If using the <<EOF syntax then variables are expanded immediately before passing to docker.
# Therefore PBS_O_WORKDIR in the following line will be taken from the setting in the job shell
cd "$PBS_O_WORKDIR"
pwd
echo should fail:
python -c "import keras;print('Imported keras')"
# set environment variable to install packages to world-writable location inside container
export PYTHONUSERBASE=/workspace/.local
# Escaping PYTHONUSERBASE in the next line so that it is expanded inside docker and take the value set inside the container
mkdir -p \$PYTHONUSERBASE/bin
export PATH=\$PATH:\$PYTHONUSERBASE/bin
echo PATH in job shell
echo PATH=$PATH
echo PATH inside container
echo PATH=\$PATH
echo install keras inside container
pip install -U -q --no-cache-dir --user keras
echo should work:
python -c "import keras;print('Imported keras')"

### options to read requirements from file and disable the message warning about pip version:
## pip install -U -q --no-cache-dir --disable-pip-version-check --user -r reqs.txt
EOF
