#!/bin/bash
#
#
# TEMPLATE FOR IC PRODUCTIONS AT FT2
#
#SBATCH --time {jobtime}
#SBATCH --job-name {jobname}
#SBATCH --output {logfilename}
#SBATCH --error  {errfilename}
#SBATCH --qos shared_short
#SBATCH --partition shared
##SBATCH --qos amd-shared
##SBATCH --partition amd-shared 
##SBATCH --qos default
##SBATCH --partition thin-shared
##SBATCH --qos cl-intel-shared
##SBATCH --partition cl-intel-shared
#SBATCH -n 1
#SBATCH -N 1

start=`date +%s`

#################################
#########  JOB CORE ############
################################
# set up IC in the CESGA machine
source $STORE/ic_setup.sh
source $HOME/data_labelling/setup.sh
pwd

{commands} #aqui iran los comandos a correr, tipo city beersheba beersheba.conf


end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds
