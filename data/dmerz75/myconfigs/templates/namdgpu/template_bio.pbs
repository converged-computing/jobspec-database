#!/bin/bash
# #PBS -l walltime=250:00:00
# #PBS -l nodes=1:ppn=16:gpu          # original
#PBS -l nodes=1:ppn=xxprocsxx:xxnodexx
#PBS -j oe
#PBS -N xxsubnamexx

# cd $PBS_O_WORKDIR


#  ---------------------------------------------------------  #
#  Establish environmental variables.                         #
#  ---------------------------------------------------------  #
USER=xxuserxx # self.user
MY_DIR=xxmy_dirxx # self.cwd
JOB_DIR_NAME=xxjob_dir_namexx # self.job_dir_name
JOB_PATH_LOCAL=xxjob_path_localxx # self.job_path_local
JOB_PATH_RUN=xxjob_path_runxx # self.job_path_run # /scratch/user/$JOB_SUB
JOB_COMPLETED_DIR=xxcompleted_dirxx # self.completed_dir # /data4/user/completed/$JOB_SUB


#  ---------------------------------------------------------  #
#  Prepare directories.                                       #
#  ---------------------------------------------------------  #
rm -rf $JOB_PATH_RUN
mkdir -p $JOB_PATH_RUN
cp $JOB_PATH_LOCAL/* $JOB_PATH_RUN
cd $JOB_PATH_RUN


#  ---------------------------------------------------------  #
#  Locate Job executables and libraries.                      #
#  ---------------------------------------------------------  #
NAMD_DIR=NAMD_2.9_Linux-x86_64-multicore-CUDA
NAMD_HOME=${HOME}/opt/${NAMD_DIR}
export NAMD_HOME
export LD_LIBRARY_PATH=/usr/local/lib64:${NAMD_HOME}


#  ---------------------------------------------------------  #
#  Count the nodes.                                           #
#  ---------------------------------------------------------  #
# NODES=`cat $PBS_NODEFILE`
# NODELIST=~/nodelist.1.namd2
# echo "group main" > $NODELIST

# ncount=0
# for node in $NODES
# do		
#     echo $node
#     echo "host $node" >> $NODELIST
#     ncount=$((ncount+1))
# done		
# cp $NODELIST ~/nodelist.1.namd2.counted

# OR
ncount=xxprocsxx


#  ---------------------------------------------------------  #
#  Begin timing for namd2                                     #
#  ---------------------------------------------------------  #
STARTTIME=$(date +%s)


#  ---------------------------------------------------------  #
#  Running namd2                                              #
#  ---------------------------------------------------------  #
${NAMD_HOME}/charmrun ${NAMD_HOME}/namd2 +p${ncount} +idlepoll +devices xxdevicesxx xxnamdconfigxx >& gpu.log


#  ---------------------------------------------------------  #
#  End timing for namd2                                       #
#  ---------------------------------------------------------  #
# wait
ENDTIME=$(date +%s)
TOTALTIME=$(($ENDTIME-$STARTTIME))
echo $TOTALTIME >> ${JOB_PATH_RUN}/time.dat


#  ---------------------------------------------------------  #
#  Clean up!                                                  #
#  ---------------------------------------------------------  #
rm -f $JOB_COMPLETED_DIR     # rm -rf $JOB_DIR
mkdir -p $JOB_COMPLETED_DIR  # mkdir -p $JOB_DIR
rsync -auvz $JOB_PATH_RUN/* $JOB_COMPLETED_DIR
# cp $JOB_PATH_LOCAL/* $JOB_COMPLETED_DIR
# cd $JOB_COMPLETED_DIR        # cd $MY_DIR
# cd $MY_DIR
# cp * $JOB_COMPLETED_DIR      # cp * $JOB_DIR
# rm -rf $JOB_DIR
