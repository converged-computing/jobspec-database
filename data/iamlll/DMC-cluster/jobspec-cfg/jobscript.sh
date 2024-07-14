#!/bin/bash
#SBATCH --job-name=giant
#SBATCH -n 1
#SBATCH --mem-per-cpu 7gb #memory per processor to be used for each task
#SBATCH -o giant-%A-%a.out #A: job ID, a: task ID
#SBATCH --array 1-10 # Max number allowed jobs is 40, try not to overload stack memory

cd $SLURM_SUBMIT_DIR

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated = $SLURM_NTASKS"

source /etc/profile #commands in the file will be executed as they were executed in command line instead of in a new subshell

# ORDER OF STEPS:
# 1. Generate simulation parameters with generateparams.py

# 2. Simulate system using these parameters - make sure that [total number of jobs being submitted (--array)] x [number of tasks per job (per_task)] = total number of different parameter combos. You can check the file length of the parameter file with "wc -l [parameter file path]". Subtract 1 from the final count to account for the header row.
# e.g. sbatch jobscript.sh

#run simulation function
start=`date +%s.%N`
joutdir='/mnt/home/llin1/scratch/test'
paramfile=$joutdir/data_eta_l_params_lin.csv
i=$SLURM_ARRAY_TASK_ID
echo $i
# maybe use this if below method doesn't work
n=$(wc -l < $paramfile)
echo $n
#P=`awk "NR==$i+1" $paramfile`
#./compute $P
#echo $P
per_task=5
ID=data

start_num=$(( ($i-1)*$per_task ))
end_num=$(( $i*$per_task ))

if [ $end_num -gt $(( $n-1 )) ]; then
    end_num=$(( $n-1 )) #-1 since not counting the header row
fi

echo $start_num
echo $end_num

python -u jobarray.py -outdir $joutdir -startnum $start_num -endnum $end_num -paramfile $paramfile > $joutdir/$ID-$i.out & 

wait
end=`date +%s.%N`

runtime=$(echo "$end - $start" | bc -l)
seconds=$(bc <<< "scale=2; $runtime"); 
echo "Runtime: $seconds seconds"
