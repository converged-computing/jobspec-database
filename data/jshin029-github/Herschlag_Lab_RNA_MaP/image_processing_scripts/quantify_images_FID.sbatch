#!/bin/bash 
#
# bash script to call quantifyTilesDownstream.py using SLURM scheduler on Sherlock
#
# Usage: quantify_images.sbatch image_dir seq_dir roff_dir fluor_dir gv_path
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=quantify_images
#################  
#a file for job output, you can check job progress
#SBATCH --output=quantify_tiles.out
#################
# a file for errors from the job
#SBATCH --error=quantify_tiles.err
#################
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=15:00:00
#################
#quality of service; think of it as job priority
#SBATCH --partition=biochem,owners,normal
#SBATCH --qos=normal
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#tasks to run per node; a "task" is usually mapped to a MPI processes.
# for local parallelism (OpenMP or threads), use "--ntasks-per-node=1 --cpus-per-task=16" instead
# Sherlock nodes have 16 cpus. For some reason, you can request more than that on 'owners' partition, but not on others. 
# It can't give you more obviously, but it's interesting that it lets you ask on one partion but not another.
# Note: On Sherlock2, most of the nodes we have access to have 20 cores. 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#################


module load python/2.7.13
source $SCRIPT_DIR/env/bin/activate

# Load matlab
module load matlab/R2017b

MATLABPATH="$IMAGING_DIR/CPlibs:$IMAGING_DIR/CPscripts"

export MATLABPATH

# Define paths
image_dir=$1
seq_dir=$2
roff_dir=$3
fluor_dir=$4
gv_path=$5
log_dir=$6

# Define other script parameters
num_cores="20"
data_scaling="MiSeq_to_TIRFStation1"

# Define filter subsets to use for registration
reg_subset0="FID"
reg_subset1="anyRNA"
#reg_subset1="FIDUCIAL"

# Set outputs appropriately 
script=$(basename $0)
script_name=${script%.*}
#log_dir=$image_dir/$script_name"Logs"
log_file_suffix=".log"
err_file_suffix=".err"

mkdir -p $log_dir

# Quantification using SLURM scheduler
echo "Submitting jobs via SLURM..."

for d in $image_dir/*
do

    
    d_base=$(basename $d)
    log_file=$log_dir/$d_base$log_file_suffix
    err_file=$log_dir/$d_base$err_file_suffix
    echo "Image: $d"
    echo "Generating Log Files:"
    echo $log_file
    echo $err_file

    start_time=$SECONDS
    echo "Starting quantification for $image_dir at timepoint $d_base..."

    srun python $IMAGING_DIR/CPscripts/quantifyTilesDownstream.py \
        -id $image_dir -ftd $seq_dir -rod $roff_dir -fd $fluor_dir -n $num_cores \
        -rs $reg_subset0 \
        -sf $data_scaling -gv $gv_path 1> $log_file 2> $err_file

    sleep 1  # Pause for 1 second to avoid overloading scheduler
    duration=$(( SECONDS - start_time))
    echo "Done with quantification for $image_dir at timepoint $d_base. Duration: $duration" | tee -a $log_file

done	

