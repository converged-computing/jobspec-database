#!/bin/sh

# Job name to be reported by qstat
#PBS -N concurrent_lammps

# Declare Job, non-rerunable
#PBS -r n

# Specify name for output log file
#PBS -o concurrent_lammps_log

# Join standard output and error so we only get one logfile
#PBS -j oe

# Mail to user on a=abort, b=begin, e=end
#PBS -m aeb

# set the email address where job-related notifications will be sent
#PBS -M jmlarson@anl.gov

# Number of nodes 
#PBS -l nodes=100:ppn=16

# Specify CPU time needed
#PBS -l walltime=00:05:00

# Select queue 
##PBS -q shared

cd $PBS_O_WORKDIR

export PYTHONPATH="${PYTHONPATH}:/home/jlarson/research/superchwm/codes/pyflamestk_20161019/" # For Eugene's multi-objective optimization problem

# A little useful information for the log file...
echo Master process running on: $HOSTNAME
echo Directory is:  $PWD
echo PBS has allocated the following nodes:
cat $PBS_NODEFILE
NPROCS="$(wc -l < $PBS_NODEFILE)"
echo This job has allocated $NPROCS cores

rm machinefile
cat $PBS_NODEFILE | sort | uniq >> machinefile

# Put in a timestamp
echo Starting executation at: `date`

pwd
#cmd="mpirun -np 16 /soft/OPAL/mvapich2-intel17/opal-1.4/bin/opal optgun.in"
#cmd="mpirun -np 16 opal optgun.in"
#cmd="mpiexec -np 4 -machinefile machinefile python call_opal_concurrently.py"
cmd="mpiexec -np 100 -machinefile machinefile python3 call_LAMMPS_concurrently.py"

echo The command is: $cmd
echo End PBS script information. 
echo All further output is from the process being run and not the pbs script.\n\n
$cmd

# Print the date again -- when finished
echo Finished at: `date`
