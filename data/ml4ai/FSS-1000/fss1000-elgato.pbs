#!/bin/bash
#
## --------------------------------------------------------------
#### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Set the job name (needed for job lookup)
#PBS -N _fss1000
### Specify the PI group for this job (always claytonm)
#PBS -W group_list=claytonm
### Set the queue for your job (options: windfall, standard, high_priority, debug -- only on Ocelote)
#PBS -q windfall
### Set the number of nodes, cores and memory that will be used for this job
### pcmem is optional as it defaults to 6gb per core. Note: mem=ncpus x pcmem
###PBS -l select=1:ncpus=4:mem=32gb:np100s=1:os7=True
#PBS -l select=1:ncpus=16:mem=250gb:ngpus=1:pcmem=16gb
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=0:30:0
### Optional. cput = time x ncpus. If not included, default := cput = walltime x ncpus.
###PBS -l cput=0:30:0

# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------
### Load required modules/libraries if needed
# If using on elgato
module load singularity/3.6.4
# If using on ocelote
#module load singularity/3/3.6.4

### Define path to singularity image that we will use for running commands
CONTAINER=/groups/claytonm/arete-realsim/containers/ubuntu18_torch_torchvision_opencv_cuda10.sif

### NOTE: You must update the SCRIPT_PATH to your context
# SCRIPT_ROOT=/home/u32/usg/FSS-1000
# SCRIPT_ROOT=/home/u25/claytonm/projects/FSS-1000
SCRIPT_ROOT=/home/u11/seanmhendryx/realsim/FSS-1000

DATA_ROOT=/xdisk/claytonm/project/arete-realsim/data
OUTPUT_ROOT=/xdisk/claytonm/projects/arete-realsim/results

### Run commands in the singularity container using exec.
cd $SCRIPT_ROOT
singularity exec $CONTAINER python3 autolabel_hpc.py --output_root=$OUTPUT_ROOT
# If not specifying path to input data, this example will just run on the example images shipped with the repo.
