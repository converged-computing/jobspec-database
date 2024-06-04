#!/bin/bash

#SBATCH --job-name=test_MPI
#SBATCH --array=20

##SBATCH --mail-user=pengzhai@umich.edu
##SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=5gb
#SBATCH --partition=standard
#SBATCH --account=yiheh1
## remember to change the FILE name and array number
#SBATCH --export=ALL,MPI_MODULE=openmpi/4.1.6,EXECUTABLE=julia,FILE=run.jl,TEST=quartz_test.txt

# mpiexec -n 8 julia --project testMPI.jl

#SBATCH --output=log/%x-%a.log
#SBATCH --error=log/error-%x-%a.log

date
echo "$SLURM_ARRAY_TASK_ID"
#domain=$(sed "${SLURM_ARRAY_TASK_ID}!d" ${TEST} | awk -F',' {'print $1}')
#res=$(sed "${SLURM_ARRAY_TASK_ID}!d" ${TEST} | awk -F',' {'print $2}')
#FZ_length=$(sed "${SLURM_ARRAY_TASK_ID}!d" ${TEST} | awk -F',' {'print $3}')
#half_width=$(sed "${SLURM_ARRAY_TASK_ID}!d" ${TEST} | awk -F',' {'print $4}')
#multiple=$(sed "${SLURM_ARRAY_TASK_ID}!d" ${TEST} | awk -F',' {'print $5}')
#a_over_b=$(sed "${SLURM_ARRAY_TASK_ID}!d" ${TEST} | awk -F',' {'print $6}')
#Lc=$(sed "${SLURM_ARRAY_TASK_ID}!d" ${TEST} | awk -F',' {'print $7}')
#echo $domain
#echo $res
#echo $FZ_length
#echo $half_width
#echo $multiple
#echo $a_over_b
#echo $Lc

#export TURBO="/nfs/turbo/lsa-yiheh/yiheh-mistorage/pengz/data/damage_test"
#cd ${TURBO}/$domain"_"$res"_"$FZ_length"_"$half_width"_"$multiple"_"$a_over_b"_"$Lc
#cd ${TURBO}
# cd damage_test_2

# # Use when a defined module environment related to OpenMPI is wished
module load gcc/10.3.0
module load ${MPI_MODULE}
module list

export MPIRUN_OPTIONS="--bind-to core --map-by node:PE=${SLURM_CPUS_PER_TASK} -report-bindings"
# export MPIRUN_OPTIONS="--bind-to core"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUM_CORES=${SLURM_NTASKS}*${SLURM_CPUS_PER_TASK}

echo "${EXECUTABLE} running on ${NUM_CORES} cores with ${SLURM_NTASKS} MPI-tasks and ${OMP_NUM_THREADS} threads"
startexe="mpirun -n ${SLURM_NTASKS} ${MPIRUN_OPTIONS} ${EXECUTABLE} ${FILE} ${SLURM_ARRAY_TASK_ID} ${TEST}"
echo $startexe
exec $startexe


