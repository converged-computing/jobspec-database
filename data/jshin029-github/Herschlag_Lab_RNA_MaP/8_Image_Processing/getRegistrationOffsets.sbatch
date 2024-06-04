#!/bin/bash
#
# bash script to call getRegistrationOffsets.py using SLURM scheduler on Sherlock
#
# Usage: ex_getRegistrationOffset_V2.sbatch # Edit this sbatch file before running!!!
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#set a job name
#SBATCH --job-name=reg_offsets
#################
#a file for job output, you can check job progress
#SBATCH --output=reg_offsets.out
#################
# a file for errors from the job
#SBATCH --error=reg_offsets.err
#################
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=1:00:00
#################
#quality of service; think of it as job priority
#SBATCH --partition=biochem,normal
#SBATCH --qos=normal
#SBATCH --mem=32G
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#tasks to run per node; a "task" is usually mapped to a MPI processes.
# for local parallelism (OpenMP or threads), use "--ntasks-per-node=1 --cpus-per-task=16" instead
# Sherlock nodes have 16 cpus. For some reason, you can request more than that on 'owners' partition, but not on others.
# It can't give you more obviously, but it's interesting that it lets you ask on one partion but not another.
# Note: On Sherlock2, most of the nodes we have access to have 20 cores.
#################


module load python/3.6.1
source $py3env/bin/activate

# Load matlab
module load matlab/R2017b

MATLABPATH="$IMAGING_DIR/CPlibs:/$IMAGING_DIR/CPscripts"

export MATLABPATH

####################################
### Variables to change
FID_image_dir="<path to images>/09_imageTilesBothColors_green"
FID_tile_dir="<path to split_tile fid>/FID"
out_dir="<output path where log files and registration offsets will go>"



####################################

python $SCRIPT_DIR/getRegistrationOffsets.py -id $FID_image_dir -sd $FID_tile_dir -gv $IMAGING_DIR -f FID -od $out_dir 1> $out_dir/logfile.txt 2> $out_dir/errfile.txt
