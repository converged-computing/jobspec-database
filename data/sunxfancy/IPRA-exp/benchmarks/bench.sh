#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=0-02:00:00     # 2 hours
#SBATCH --job-name="ipra-bench"
#SBATCH -p short
#SBATCH --constraint=intel

##### //SBATCH --mail-user=xsun042@ucr.edu
##### //SBATCH --mail-type=ALL

module load singularity
module load parallel

# Define srun arguments:
# srun="srun -n1 -N1 --exclusive"
srun=""
# --exclusive     ensures srun uses distinct CPUs for each job step
# -N1 -n1         allocates a single core to each task

# Define parallel arguments:
parallel="parallel -k -N 1 --delay .2 -j 1 --joblog $1.parallel_joblog --resume"
# -N 1              is number of arguments to pass to each job
# --delay .2        prevents overloading the controlling node on short jobs
# -j $SLURM_NTASKS  is the number of concurrent tasks parallel runs, so number of CPUs allocated
# --joblog name     parallel's log file of tasks it has run
# --resume          parallel can use a joblog and this to continue an interrupted run (job resubmitted)

echo $SLURM_PROCID-$SLURM_JOBID
echo Benchmark: $1

shopt -s extglob
rm -vrf /scratch/!(xsun042)
rm -vrf /dev/shm/xsun042

arguments=()

# $srun singularity exec singularity/image.sif make BUILD_PATH=/scratch/xsun042/$SLURM_JOBID/$2/0

function set_bench() {
    arguments+=("benchmarks/$1/$2.$3")
    if [[ "$2" == *"fdoipra"* ]]; then
        arguments+=("benchmarks/$1/$2.1-10.$3")
        arguments+=("benchmarks/$1/$2.1-20.$3")
        arguments+=("benchmarks/$1/$2.3-10.$3")
        arguments+=("benchmarks/$1/$2.3-20.$3")
        arguments+=("benchmarks/$1/$2.5-10.$3")
        arguments+=("benchmarks/$1/$2.5-20.$3")
        arguments+=("benchmarks/$1/$2.10-10.$3")
        arguments+=("benchmarks/$1/$2.10-20.$3")
    fi
}

set_bench $1 pgo-thin bench
set_bench $1 pgo-thin-fdoipra bench
set_bench $1 pgo-thin-fdoipra2 bench
set_bench $1 pgo-thin-fdoipra3 bench
set_bench $1 pgo-thin-bfdoipra bench
set_bench $1 pgo-thin-bfdoipra2 bench
set_bench $1 pgo-thin-bfdoipra3 bench
set_bench $1 pgo-thin-ipra bench
set_bench $1 pgo-thin-fdoipra4 bench
set_bench $1 pgo-thin-fdoipra5 bench
set_bench $1 pgo-thin-fdoipra6 bench
set_bench $1 pgo-thin-bfdoipra4 bench
set_bench $1 pgo-thin-bfdoipra5 bench
set_bench $1 pgo-thin-bfdoipra6 bench

# Run the tasks:
$parallel "mkdir -p /dev/shm/xsun042/$SLURM_JOBID/{%}; $srun singularity exec singularity/image.sif make BUILD_PATH=/dev/shm/xsun042/$SLURM_JOBID/{%} {}" ::: "${arguments[@]}"
# in this case, we are running a script named runtask, and passing it a single argument
# {1} is the first argument
# parallel uses ::: to separate options. Here {1..64} is a shell expansion defining the values for
#    the first argument, but could be any shell command
#
# so parallel will run the runtask script for the numbers 1 through 64, with a max of 40 running 
#    at any one time
#
# as an example, the first job will be run like this:
#    srun -N1 -n1 --exclusive ./runtask arg1:1

