#!/bin/bash
#
# NAMD GPU script for MD

# Job name:
#----------------
#SBATCH --job-name="a178-vanilla"
#SBATCH --partition=mf_titanx
#SBATCH --nodes=1
#SBATCH --mem=8gb
#SBATCH --time=72:00:00
#SBATCH --distribution=block:cyclic
#SBATCH --gres=gpu:4
#SBATCH --ntasks=8
#SBATCH --depend=afterany:3755380
#--------------



#########################################
# Set copy_local to "yes" to copy contents of submission directory to node-local workspace.
# After the calculation, results will be copied back and the temporary work directory
# /work/$USER/$SLURM_JOB_ID will be deleted.
#
# If time runs out before the job is done, the copyback will be done by a post-execution
# script, but only if the SLURM_WORK_DIR-$SLURM_JOB_ID file (created in this script) exists in the
# submission directory. 

copy_local="no"

#########################################

# Define some simpler local evironment variables:

slurm_startjob(){
#----------------- Actual calculation command goes here: ---------------------------
export NAMD="/beegfs/DATA/mobley/limvt/local/namd/2.12-multicore-CUDA"
export LD_LIBRARY_PATH="${NAMD}/libcudart.so.6.5"
i=$SLURM_ARRAY_TASK_ID
N=$SLURM_NTASKS

$NAMD/namd2 +idlepoll +isomalloc_sync +p$((N*2)) equil1.inp > out01.log
#$NAMD/namd2 +idlepoll +isomalloc_sync +p$((N*2)) step7.${i}_production.inp > step7.${i}_production.log

echo Job Done
#-----------------------------------------------------------------------------------
}

#########################################################################
#########    Never change anything in this section              #########
#########    unless you know what you are doing of course !!!!  #########
#########################################################################

# SLURM environment variables:
# SLURM_JOB_USER             The user who started the job
# SLURM_ARRAY_TASK_ID        Job array ID (index) number.
# SLURM_ARRAY_JOB_ID         Job array’s master job ID number.
# SLURM_JOB_CPUS_PER_NODE    Count of processors available to the job on this node.
# SLURM_JOB_ID                The ID of the job allocation.
# SLURM_JOB_NAME             Name of the job.
# SLURM_JOB_NODELIST         List of nodes allocated to the job.
# SLURM_JOB_NUM_NODES        Total number of nodes in the job’s resource allocation.
# SLURM_JOB_PARTITION        Name of the partition in which the job is running.
# SLURM_NODEID               ID of the nodes allocated.
# SLURMD_NODENAME            Names of all the allocated nodes.
# SLURM_NTASKS               Same as -n, --ntasks
# SLURM_NTASKS_PER_CORE      Number of tasks requested per core. [If specified]
# SLURM_NTASKS_PER_NODE      Number of tasks requested per node. [If specified]
# SLURM_NTASKS_PER_SOCKET    Number of tasks requested per socket. [If specified]
# SLURM_PROCID               The MPI rank (or relative process ID) of the current process.
# SLURM_SUBMIT_DIR           The directory from which sbatch was invoked.
# SLURM_SUBMIT_HOST          The hostname of the computer from which sbatch was invoked.
# SLURM_TASKS_PER_NODE       Number of tasks to be initiated on each node. In the same order as SLURM_JOB_NODELIST.

# Function to echo informational output
slurm_info_out(){
# Informational output
echo "=================================== SLURM JOB ==================================="
date
echo
echo "The job will be started on the following node(s):"
echo $SLURM_JOB_NODELIST
echo
echo "Slurm User:         $SLURM_JOB_USER"
echo "Run Directory:      $(pwd)"
echo "Job ID:             $SLURM_JOB_ID"
echo "Job Name:           $SLURM_JOB_NAME"
echo "Partition:          $SLURM_JOB_PARTITION"
echo "Number of nodes:    $SLURM_JOB_NUM_NODES"
echo "Number of tasks:    $SLURM_NTASKS"
echo "Submitted From:     $SLURM_SUBMIT_HOST:$SLURM_SUBMIT_DIR"
echo "=================================== SLURM JOB ==================================="
echo
echo "--- SLURM job-script output ---"
}


# Copy data to a local work directory:
if [ "$copy_local" = "yes" ]; then
  echo $HOSTNAME > $SLURM_SUBMIT_DIR/SLURM_WORK_NODE-$SLURM_JOB_ID
  if [ "$?" -ne "0" ]; then
    echo "Unable to write $SLURM_SUBMIT_DIR/SLURM_WORK_NODE-$SLURM_JOB_ID"
    echo "$SLURM_JOB_ID on node $HOSTNAME failed to write $SLURM_SUBMIT_DIR/SLURM_WORK_NODE-$SLURM_JOB_ID " >> $HOME/SURM_WARNINGS
    echo "$SLURM_SUBMIT_DIR/SLURM_WORK_NODE-$SLURM_JOB_ID should contain:" >> $HOME/SLURM_WARNINGS
    echo "$HOSTNAME" >> $HOME/SLURM_WARNINGS
  fi
  if (( $SLURM_JOB_NUM_NODES > 1 )); then
    work_dir="/lustre/SCRATCH/$(id -gn)/$USER/$SLURM_JOB_ID"
  else
    work_dir="/work/$USER/$SLURM_JOB_ID"
  fi
  
  mkdir -p $work_dir
  rsync -a --exclude=slurm-${SLURM_JOB_ID}.* $SLURM_SUBMIT_DIR/ $work_dir/
  if (( $? != 0)); then
    echo "FAIL: rsync to local execution directory had problems. Aborting job."
    exit 1
  else
    echo "$work_dir" > $SLURM_SUBMIT_DIR/SLURM_WORK_DIR-$SLURM_JOB_ID
    if [ "$?" -ne "0" ]; then
      echo "Unable to write $SLURM_SUBMIT_DIR/SLURM_WORK_DIR-$SLURM_JOB_ID"
      echo "$SLURM_JOB_ID on node $HOSTNAME failed to write $SLURM_SUBMIT_DIR/SLURM_WORK_DIR-$SLURM_JOB_ID" >> $HOME/SLURM_WARNINGS
      echo "$SLURM_SUBMIT_DIR/SLURM_WORK_DIR-$SLURM_JOB_ID should contain:" >> $HOME/SLURM_WARNINGS
      echo "$work_dir" >> $HOME/SLURM_WARNINGS
    fi
  fi
  cd $work_dir
fi

slurm_info_out

slurm_startjob

# Copy data back to the submission directory:
if [ "$copy_local" = "yes" ]; then
  rsync -a $work_dir/ $SLURM_SUBMIT_DIR/
  if (( $? == 0)); then
    cd $SLURM_SUBMIT_DIR
    rm -rf $work_dir
    # Since the copyback worked, delete the file that triggers the post-execution script
    rm $SLURM_SUBMIT_DIR/SLURM_WORK_DIR-$SLURM_JOB_ID
    rm $SLURM_SUBMIT_DIR/SLURM_WORK_NODE-$SLURM_JOB_ID
  else
    echo "FAIL: rsync back to submission directory had problems. Execution directory not removed."
    echo "$SLURM_JOB_ID on node $HOSTNAME had problems on final rsync" >> $HOME/SLURM_WARNINGS
    cd $SLURM_SUBMIT_DIR
    exit 1
  fi
fi

#
# End file.
#

