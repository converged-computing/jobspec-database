#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=36GB
#SBATCH --time=6-23:00:00
#SBATCH --job-name=2Lray
#SBATCH --array=1-2

module purge

export NUM_JULIA_THREADS=`nproc`

config=parameters.txt

initial_condition_file=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

rundir=$SCRATCH/twolayer_raytracing/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
mkdir -p $rundir
cp array-driver.sbatch Driver.jl TwoLayerRaytracing.jl Raytracing.jl $rundir
cp $initial_condition_file $rundir/initial_condition.jld2
cp ArrayParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK Driver.jl > run.log

exit
