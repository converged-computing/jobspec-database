#!/usr/bin/env bash


###########################################################################
## Author:  Panagiotis Velissariou <panagiotis.velissariou@noaa.gov>
##
## Version: 1.0
##
## Version - 1.0 Fri Feb 10 2023
###########################################################################


###########################################################################
## Can run this slurm script by suppling command line options.
## Example: sbatch [options] this_script [options to this_script]
#####

##### Options below explained.
## -N, --nodes=<minnodes[-maxnodes]>
##     Request that a minimum of minnodes nodes be allocated to this job.
##     Number of nodes on which to run (uppercase).
## -n, --ntasks=<number>
##     Total number of tasks to run (lowercase).
## -c, --cpus-per-task=<ncpus>
##     Request that ncpus be allocated per process. Number of cpus per task (for hyperthreading).
## --cores-per-socket=<cores>
##     Restrict  node  selection to nodes with at least the specified
##     number of cores per socket.
## --sockets-per-node=<sockets>
##     Restrict  node selection to nodes with at least the specified
##     number of sockets.
## -C, --constraint=<list>
##     Nodes can have features assigned to them by the Slurm administrator.
##     Set which nodes to use depending on node features (eg. intel, amd, YEAR2014)
## -t, --time=<time>
##     Set a limit on the total run time of the job or job step.
##     Format: "minutes", "minutes:seconds", "hours:minutes:seconds"
## --mail-type=<types>
##     Send email notification of <types> for job state changes
##     Valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
## --mail-user=<user>
##     User to receive email notification of state changes as defined by --mail-type.
##     The default value is the submitting user.
## -e, --error=<mode>
##     Specify how stderr is to be redirected. File for batch script's
##     standard error (default same as the stdout).
## -o, --output=<mode>
##     Specify the mode for stdout redirection. File for batch script's
##     standard output (default name is slum-$jobid.out).
## -J, --job-name=<jobname>
##     Specify  a  name for the job.
## -A, --account=<account_name>
##     Charge resources used by this job to specified account.  The account is an arbitrary string.
##     The account name may be changed after job submission using the scontrol command.
##     Account to charge the job, e.g., coastal, ohd.
## -p, --partition=<partition_names>
##     Request a specific partition for the resource allocation.
## -q, --qos=<qos>
##     Request a quality of service for the job.
##     QOS to submit job, e.g., batch, debug.
## --exclusive
##     Run job exclusively, i.e., do not share nodes with other jobs.
###########################################################################


#SBATCH -A coastal
#SBATCH -J FLOFS_atm2adc_RUN
#SBATCH -t 08:00:00
##SBATCH -p hera
##SBATCH -q batch
#SBATCH -n 150
#SBATCH -c 1
#SBATCH -e FLOFS_atm2adc_RUN.err.log
#SBATCH -o FLOFS_atm2adc_RUN.out.log
##SBATCH --mem=3000
##SBATCH --mail-user=USER@noaa.gov
##SBATCH --mail-type=ALL

set -e

if [ -e "${MOD_FILE}" ]; then
  source ${MOD_FILE}
  module list
else
  echo "The module file: <${MOD_FILE}> is not found"
  echo "Will continue without loading any OS defined modules"
fi

${BATCH_RUNEXE} ${BIN_DIR:+${BIN_DIR}/}NEMS.x
