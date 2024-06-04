#!/bin/bash
 
#PBS -l walltime=0:20:00,select=1:ncpus=10:mem=32gb
#PBS -J 11-101:10
#PBS -N kdd_sockeye_demo                           
#PBS -A ex-kdd-1                               
#PBS -m abe                                   
#PBS -M rtkushner@alumni.ubc.ca              
#PBS -o /scratch/ex-kdd-1/rt/kddlab_sockeye/output.txt
#PBS -e /scratch/ex-kdd-1/rt/kddlab_sockeye/error.txt
 
################################################################################
 
# load your modules
module load python3
module load parallel

cd /arc/project/ex-kdd-1/rt/kddlab_sockeye

#consume your envs, activate environments
source testvenv/bin/activate


#run your code here
python3 experiment_parser.py --outdir /scratch/ex-kdd-1/rt/kddlab_sockeye/ -it 10000 -i $PBS_ARRAY_INDEX

# or with gnu-parallel
# parallel "python3 experiment_parser.py --outdir /scratch/ex-kdd-1/rt/kddlab_sockeye/ -it 100 -i {1}" ::: $(seq  $PBS_ARRAY_INDEX $(($PBS_ARRAY_INDEX+10)))
