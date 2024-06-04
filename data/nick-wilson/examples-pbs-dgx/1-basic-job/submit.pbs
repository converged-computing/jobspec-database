#!/bin/sh

### The following requests all resources on 1 DGX-1 node
#PBS -l select=1:ncpus=40:ngpus=8

### The "select=1" specifies the number of nodes
### The "ncpus=40:ngpus=8" asks for acccess to all 8 GPU cards
### If you request less than 8 GPU then make the ncpus value
###   five times the ngpus value, e.g. select=1:ncpus=5:ngpus=1

### Specify amount of time required
###  Values less than 4 hours go into a higher priority queue
###  This job only needs 5 minute to run:
#PBS -l walltime=0:05:00

### Specify DGX queue
#PBS -q dgx

### Specify project code
#PBS -P 99999999

### Specify name for job
#PBS -N container

### Standard output by default goes to file $PBS_JOBNAME.o$PBS_JOBID
### Standard error by default goes to file $PBS_JOBNAME.e$PBS_JOBID
### To merge standard output and error use the following
#PBS -j oe

### Start of commands to be run

# Docker image to use for container
#   To see available images run command: nscc-docker images
#   If image is not present, email help@nscc.sg to request pulling image into repository on all DGX nodes
image="nvcr.io/nvidia/cuda:latest"

# Job should start in your home directory:
pwd
# Change to directory where job was submitted:
cd "$PBS_O_WORKDIR" || exit $?
pwd
# Please note that when you start a Docker container then it will start in a directory defined by the image
# You will also need to change to the correct directory inside the container

# See which node job is running on and GPU status
hostname
nvidia-smi

# The "nscc-docker run $image" command runs the following Docker command: 
#    nvidia-docker run -u UID:GID --group-add PROJECT \
#      -v /home:/home -v /scratch:/scratch -v /raid:/raid \
#      --rm -i --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 $image /bin/sh
# See full list of options with "nscc-docker run -h"
# Pass the commands that you wish to run inside the container on the standard input
nscc-docker run $image << EOF
echo Container starts in the directory:
pwd
echo Change directory to where job was submitted:
# You can do the following if using "<<EOF" to pass commands as it is expanded by the job shell
cd $PBS_O_WORKDIR
pwd
echo PBS_O_WORKDIR is set in shell:
echo PBS_O_WORKDIR=$PBS_O_WORKDIR
echo PBS_O_WORKDIR is not set in the container:
echo PBS_O_WORKDIR=\$PBS_O_WORKDIR
echo
echo By default Docker starts in a private network with hostname different to that of host operating system:
hostname
echo
nvidia-smi
EOF
