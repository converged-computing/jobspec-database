#!/bin/bash
#PBS -l nodes=11274,walltime=24:00:00
#PBS -j oe
##PBS -q debug
##PBS -l gres=
#PBS -V
#PBS -M sku@pppl.gov
#PBS -m be
#PBS -A env003

WINDOW=30 # 120 min  window of no activity/progress, while loop checks the file every minute
USER=`whoami`

#BINDIR="/lustre/atlas1/stf008/scratch/${USER}/apkill"
#cd $BINDIR
cd $PBS_O_WORKDIR
#export CRAY_CUDA_PROXY=1
export OMP_NUM_THREADS=16
export MPI_NUM=11264
export XGC_BIN=../XGC1_pushe/xgc2
export OUTFILE_PREFIX=output.out.

# Load Cray hugepage module
#source /opt/modules/default/etc/modules.sh
#module load craype-hugepages2M

# initial steup
date
mkdir restart_dir


for i in `seq 11 19`;
do
    ./run_bpmeta.sh restart_dir
    source /lustre/atlas/proj-shared/env003/camtimers/pre_aprun_env.titan
    
    # Following 3 lines are from old job script
    #/lustre/atlas/proj-shared/env003/camtimers/pre_aprun.titan -stdout output.out.1
    #aprun -n 16384 -d $OMP_NUM_THREADS ../XGC1/xgc2 >& output.out.1
    #/lustre/atlas/proj-shared/env003/camtimers/post_aprun.titan -stdout output.out.1
    
    OUTFILE=$OUTFILE_PREFIX$i  # does it work?
    /lustre/atlas/proj-shared/env003/camtimers/pre_aprun.titan -stdout $OUTFILE   #for timing data
    aprun -n $MPI_NUM -d $OMP_NUM_THREADS $XGC_BIN >& $OUTFILE &  # this "&" at the end is essential so that the code below executed, the code below monitors the temporary output file
    #echo "loop = $i"

#####################################################################################################################
    # Snippet to be moved to application PBS/qsub script:
    # Make sure to set the variable USER and WINDOW same as above or appropriately
    # Flow: store the status (number of lines) of temporary output file and keep checking every minute for updates,
    # if it is being updated, keep within while loop, if not updating for a long duration (2 hours), do apkill.
#####################################################################################################################
    #OUTFILE="$PBS_JOBID.OU"
    OUTLEN=`wc -l $OUTFILE | awk '{print $1}'`
    #echo "outlen = $OUTLEN"
    TIME=0;
    while true
    do
        sleep 60; # sleep in number of seconds
        OUTLEN_NEW=`wc -l $OUTFILE | awk '{print $1}'`
        #echo "len = $OUTLEN and $OUTLEN_NEW"
        if [ $OUTLEN -eq $OUTLEN_NEW ]; then
            TIME=`expr $TIME + 1`
        else
            TIME=0;
            OUTLEN=$OUTLEN_NEW
        fi

        #echo "len after = $OUTLEN and $OUTLEN_NEW"

        APID=`apstat | grep $USER | tail -n 1 | awk '{print $1}'`
        #echo "apid = $APID"
        if [ -n "$APID" ]; then
            if [ $TIME -gt $WINDOW ]; then
                apkill -9 $APID
                #echo "apkill = $APID"
            fi
        else
            break # break the while loop if there is no APID found
        fi
    done
#####################################################################################################################
    #end of snippet to be moved to application pbs script
#####################################################################################################################
    /lustre/atlas/proj-shared/env003/camtimers/post_aprun.titan -stdout $OUTFILE  #finish up timing data

done
wait
