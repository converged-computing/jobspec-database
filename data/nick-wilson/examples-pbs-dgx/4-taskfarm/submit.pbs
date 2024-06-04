#!/bin/bash

### Select 1 GPU
#PBS -l select=1:ncpus=5:ngpus=1

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

# Job should start in your home directory:
pwd
# Change to directory where job was submitted:
cd "$PBS_O_WORKDIR" || exit $?
pwd

# See which node job is running on and GPU status
hostname
nvidia-smi

# watch GPU processes in the background
nvidia-smi pmon -o DT -d 10 > nvidia-smi.pmon.$PBS_JOBID 2>&1 &

# GNU parallel can be used to run multiple simultaneous commands
# Various usage patterns are available
## Documentation: https://www.gnu.org/software/parallel/parallel_tutorial.html


# bash function to launch instance
launch(){
 instance=${1:-unknown}
 echo Start instance $instance at `date`
 image="nvcr.io/nvidia/tensorflow:latest-tf2"
 nscc-docker run --ipc=host $image << EOF > output.$PBS_JOBID.$instance 2>&1
cd $PBS_O_WORKDIR
echo running instance $instance
python trivial.py
EOF
 rc=$?
 echo End instance $instance at `date`
 return $?
}
export -f launch

# Number of instances to run simultaneously
nslots=10

# Example: Launch 100 instances, running $nslots at a time
parallel --jobs=$nslots launch {1} ::: {1..100}

## ## Alternative: pass a list of commands on standard input, running 10 at a time
## parallel --jobs=$nslots < commands

## ## Alternative: pass files which match input*.py files as arguments to the launch function, running 10 at a time
## parallel --jobs=$nslots launch {1} ::: input*.py

# terminate the nvidia-smi pmon command
kill %1
