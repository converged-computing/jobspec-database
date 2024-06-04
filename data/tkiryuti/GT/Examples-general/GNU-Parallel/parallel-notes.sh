#!/bin/bash

## GNU Parallel tutorial link:
# https://www.gnu.org/software/parallel/parallel_tutorial.html

## In PACE:
module load parallel

## Informational

# Print summary of the most important options 
parallel --help

# Generate the citation
parallel --citation

# Maximum size of the command line
parallel --max-line-length-allowed

# Determine the number of CPUs and CPU cores on the system
parallel --number-of-cpus
parallel --number-of-cores

## Additional information (I may need for testing)
# Check current prosesses (ps) in full format (-F)
# PSR: processor that process is currently assigned to.
ps -F
# Specify output columns
ps -o psr,cmd

# see the names of all nodes that have been assigned to your job
cat "$PBS_NODEFILE"

## Intro and simplest commands

# Print the inputs
parallel echo ::: A B C

# Generate file with A B C on a separate line each
# -k keep sequence of output same as the order of input
parallel -k echo ::: A B C > abc-file

# Use input file (-a)
parallel -a abc-file echo

# The command can be a script, a binary or a Bash function if the function is exported using export -f:
# Only works in Bash
my_func() {
  echo in my_func $1
}
# Export the function definitions to sub-shell. The function becomes like an environment variable.
	# export sets a function or variable for every program that will be run in the terminal session.
# Export function to the environment
export -f my_func
# Run test
parallel my_func ::: 1 2 3

# The above doesn't work when submitting a job. Add function to environment manually:
# --env Copy environment variable. It can also be a Bash function ("export -f" required). 
parallel --env my_func my_func ::: 1 2 3
# --env '_' Copies all EXPORTED environment variables. Run 'parallel --record-env' in a clean environment first.
parallel --record-env
parallel --env '_' my_func ::: 1 2 3

# To copy the full environment (both exported and not exported variables, arrays, and functions) use env_parallel.
env_parallel ...

See also: --record-env, --session.

## More about environment variables (supplementary info)
# See all environment variables (NAME=VALUE)
	# Environment variables are uppercase (by convention). They communicate to programs how the machine is set up.
env

## Replacement strings
... {}   # default
... {.}  # removes the extension
... {/}  # removes the path
... {//} # keeps only the path
... {/.} # removes the path and extension
... {#}  # gives the job number

# Running 4 jobs with 10 arguments
# -m is multiple arguments. If multiple jobs are being run, distributes arguments evenly among jobs
# --jobs/-j run this many jobs in parallel PER MACHINE (NODE)
parallel --jobs 4 -m echo ::: 1 2 3 4 5 6 7 8 9 10

# -X to repeat the context for each input (repeat the pre- and -post)
parallel --jobs 4 -X echo pre-{}-post ::: A B C D E F G

# -N to limit the number of arguments
parallel -N3 echo ::: A B C D E F G H

# -q to quote the command
parallel -q perl -e 'print "@ARGV\n"' ::: A B

## Controlling the output
# --dryrun to see what commands will be run without running them
# Great for testing!
parallel --dryrun echo {} ::: A B C

# --keep-order/-k to force output in the same order as the arguments
parallel -k echo ::: A B C

## Controlling the execution
# By default --jobs is the same as the number of CPU cores

# Instead of basing the percentage on the number of CPU cores GNU parallel can base it on the number of CPUs:
# --use-cpus-instead-of-cores

## Shuffle job order
# Gets all combinations but different order per run
parallel --shuf echo ::: 1 2 3 ::: a b c ::: A B C

## Time
# Show date and time
date
# --delay t makes sure there are at least t seconds between each start
parallel --delay 0.2 echo Starting {}\;date ::: 1 2 3

## Progress information
# --joblog generates a log file with jobs completed so far
parallel --joblog joblog.txt echo ::: A B C

# --resume to pickup where left off based on job log
parallel --resume --joblog joblog.txt echo ::: A B C D E F

# --resume-failed re-run the jobs that failed
# --retry-failed works almost the same (but I don't understand and might not need it)

## Termination
# --halt now,fail=1
# exit when the first job fails (has an exit code different from 0). Kill running jobs.
parallel -j2 --halt now,fail=1 echo {}\; exit {} ::: 0 0 1 2 3

# --retries retries the command (useful if command fails for unknown reasons now and then)

## Limiting the resources
# --memfree check if there is enough memory free

## Ignore hosts that are down
# --filter-hosts
# -S server

## Header
# --header repeats header from input data for each job

## Shebang
#!/usr/bin/parallel --shebang -r echo
# The parallel command is the first line of the script and the arguments can be listed in the file

## Parallelizing existing scripts
# parallel can be combined into a single file to run the script in parallel
# there is a different shebang line variation with parallel + each language

#___________________________________________________________________________________________#
### Additional reading/finds outside of the tutorial

# Using env_parallel to export environmental variables to all threads 
# https://www.msi.umn.edu/support/faq/how-can-i-use-gnu-parallel-run-lot-commands-parallel
# A recent update to GNU Parallel has added the new command env_parallel, which works like the parallel command, but also 
# exports the current environmental variables to the parallel worker threads, including threads sent to separate nodes.  
# The env_parallel command is new, and is still in a Beta testing phase. 
# The env_parallel command allows simpler exporting of module environment variables.  
# Below is an example of how to load a module and execute parallel software commands which depend on environmental variables on multiple nodes:
env_parallel --jobs 2 --sshloginfile $PBS_NODEFILE --workdir $PWD < commands.txt

# GNU parallel --jobs option using multiple nodes on cluster with multiple cpus per node
# https://stackoverflow.com/questions/22236337/gnu-parallel-jobs-option-using-multiple-nodes-on-cluster-with-multiple-cpus-pe
# -j is the number of jobs per node

### Parallel Serial Jobs Using GNU PARALLEL
## http://www.hpc.lsu.edu/training/weekly-materials/2017-Spring/gnuparallel-Feb2017.pdf

## Multi-Node Considerations
	# The mother superior node is the only one holds all the job information, like environment variables, list of node names, etc.
	# Start programs on other nodes with remote shell commands, like ssh.
#-----------------------------------------------------------------------#
## blast-2nodes.qsub
#! /bin/bash 
#PBS -A hpc_hpcadmin3
#PBS -l nodes=2:ppn=16
#PBS -l walltime=00:20:00
#PBS –q workq
#PBS –N blastp-2nodes
export DIR=/project/$USER/distribution_workload/blast
cd $DIR; mkdir output
# on compute node1 (mother superior)
blastp –query data/input1.faa –db db/xxx –out output/input1.out &
…
blastp –query data/input8.faa –db db/xxx –out output/input8.out &
# start tasks on compute node2
ssh -n $HOST2 “cd $DIR; blastp –query data/input9.faa –db db/xxx –out output/input9.out” &
…
ssh -n $HOST2 “cd $DIR; blastp –query data/input16.faa –db db/xxx –out output/input16.out” &
wait #all the child processes to finish before terminating the parent process
#-----------------------------------------------------------------------#

## GNU Parallel
	# A shell tool to execute independent jobs in parallel using one/more computers
	# Under the hood: the mother superior spawns ssh connections to each remote node/core, where an independent task is conducted
#-----------------------------------------------------------------------#
#!/bin/bash	
#PBS	-l nodes=2:ppn=16	
#PBS	-l	walltime=1:00:00	
#PBS	-A hpc_hpcadmin3	
#PBS	-q workq	
#PBS	–N blast	
#PBS	–j	oe	
export	WDIR=/project/$USER/GNU_PARALLEL/serial	
cd "$WDIR";		
export	JOBS_PER_NODE=16
#	parallel command	flags
PARALLEL="parallel	-j	$JOBS_PER_NODE	--slf	$PBS_NODEFILE	--wd $WDIR		
	 	--joblog	logs/runtask.log	--resume"
#	gnu-parallel	launch	serial	tasks	
$PARALLEL	–a	input.lst sh run_blast.sh	{}	output/{/.}.out
#-----------------------------------------------------------------------#
# --slf filename
#           File with sshlogins. The file consists of sshlogins on separate lines. Empty lines and lines starting with '#' are
#           ignored.

## GNU Parallel Syntax

# Reading command arguments on the command line
parallel	[OPTIONS]	COMMAND	{} ::: TASKLIST

# Reading command arguments from an input file
parallel	[OPTIONS]	COMMAND	{} :::: TASKLIST.LST
parallel	–a	TASKLIST.LST	[OPTIONS]	COMMAND	{}

## GNU Parallel Options/Flags

# Link input sources - one argument read from each of the input sources
... --link
# All 9 combinations:
parallel echo hi {1} {2} ::: 1 2 3 ::: a b c
# Not all combinations, only 3
parallel --link echo hi {1} {2} ::: 1 2 3 ::: a b c

# Separate input columns
... --colsep ‘\,’
... -C ‘\,’

# Manipulate input string
... {} 	 "path/input.txt"	# default string
... {.}  "path/input"		# remove extension
... {/}  "input.txt"		# remove path
... {/.} "input"			# remove path and extension

# Number of jobs/per machine (node)
... --jobs N
... -j N

# USED WHEN MORE THAN ONE NODE IS REQUESTED FOR A JOB
... --slf nodefile
... --sshloginfile $PBS_NODEFILE

# Change working directory (default is login ~)
... --wd $PBS_O_WORKDIR
... --workdir $PBS_O_WORKDIR

# Generate a log file of each completed sub-tasks
# Used as checkpoints to resume unfinished tasks (--resume)
# Identify failed jobs
... --joblog

# Show overall computational progress
... --progress

## Steps for testing
# 1) Test serial task
# 2) Test parallel job interactively

## Testing GNU parallel
# Request an interactive node
qsub –I –A xxx –l nodes=1:ppn=16 –l walltime=2:00:00

## PACE tutorial on interactive jobs:
# https://docs.pace.gatech.edu/interactiveJobs/interactive_cmd/
qsub -A GT-ktk3 -l walltime=02:00:00 -l nodes=2:ppn=4 -l pmem=1gb -q embers -I

# Interactively run commands line by line
#-----------------------------------------------------------------------#
#PBS	…
export	WDIR=/project/$USER/GNU_PARALLEL/serial	
cd $WDIR;		
export	JOBS_PER_NODE=16
#	parallel command	options	
PARALLEL="parallel	-j	$JOBS_PER_NODE	--slf	$PBS_NODEFILE	--wd $WDIR		
	 	--joblog	logs/task.log	--resume"
#	gnu-parallel	launch	serial	tasks	
$PARALLEL	–a	input.lst sh run_blast.sh	{}	output/{/.}.out	
#-----------------------------------------------------------------------#

## Distribute multi-thread tasks
# (multi_threads/blast_job.pbs)
#-----------------------------------------------------------------------#
#!/bin/bash	
#PBS	-l nodes=2:ppn=16	
#PBS	-l	walltime=1:00:00	
#PBS	-A hpc_hpcadmin3	
#PBS	-q workq	
#PBS	–N blast-mt-parallel
#PBS	–j	oe	
export	WDIR=/project/$USER/GNU_PARALLEL/mul@_threads
cd $WDIR;		
export	JOBS_PER_NODE=8	
export	NTHREADS=2
#	parallel command	options
PARALLEL="parallel	-j	$JOBS_PER_NODE	--slf	$PBS_NODEFILE	--wd
	 	$WDIR	--joblog	logs/runtask.log	--resume"
#	gnu-parallel	launch	serial	tasks	
$PARALLEL	–a	input.lst sh run_blast.sh	{}	output/{/.}.out	$NTHREADS\
#-----------------------------------------------------------------------#

