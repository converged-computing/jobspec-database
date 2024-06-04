#! /bin/bash
# USAGE: oarsub -S ./start-oar.sh

##########################
#                        #
#   The OAR  directives  #
#                        #
##########################

#          Set the name of the job
#OAR -n ApkWorkers

#          Besteffort and idempotent
#OAR -t besteffort
#OAR -t idempotent

#          Activate the checkpoint mode
#OAR --checkpoint 1

#          Set number of cpu/time resources
#OAR -l core=BEST,walltime=9000:00:00


#####################################
#                                   #
#   The UL HPC specific directives  #
#                                   #
#####################################

if [ -f  /etc/profile ]; then
    .  /etc/profile
fi

#####################################
#
# Job settings
#
#####################################

# UNIX signal sent by OAR
CHKPNT_SIGNAL=12

# exit value for job resubmission
EXIT_UNFINISHED=99

#####################################
#
# Other functions
#
#####################################

function checkpoint {
    pkill -9 -f 'celery worker'

    exit $EXIT_UNFINISHED
}

##########################################
# Run the job
#

DIR="$(dirname "$0")/../"

# directory where to run
cd $DIR

# trap the checkpoint signal
trap checkpoint $CHKPNT_SIGNAL

# load the right module environment
module load lang/Python/2.7.13-foss-2017a

# execute the command in foreground
source venv/bin/activate
celery -A apkworkers worker -Q analysis --loglevel=warn | tee -a worker.log
