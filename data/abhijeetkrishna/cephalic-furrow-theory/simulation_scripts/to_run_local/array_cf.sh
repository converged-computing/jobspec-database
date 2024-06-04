#!/bin/bash

#--- NOT EXECUTED --- #SBATCH -J wd
#--- NOT EXECUTED --- #SBATCH -o log_files/wd_out_%A_%a
#--- NOT EXECUTED --- #SBATCH -e log_files/wd_err_%A_%a
#--- NOT EXECUTED --- #SBATCH -t 00:10:00
#--- NOT EXECUTED --- #SBATCH -a 0-9

job_id=0
py_map_creator=map_index_cf.py
initiator=initialize.py
simulation=gradDescent.jl
map_file=map_index_$job_id.csv
postprocess=postprocess.py

python3 $py_map_creator $job_id
nbfiles=$(wc -l < $map_file)
nbfiles=$((nbfiles-2)) #because bash reads header line
echo "Number of files $nbfiles"

for i in {0..8}
do
	echo "Running task number : $i"
	python3 $initiator $map_file $i #$SLURM_ARRAY_TASK_ID
	julia $simulation $map_file $i #$SLURM_ARRAY_TASK_ID
	python3 $postprocess $map_file $i #$SLURM_ARRAY_TASK_ID
done