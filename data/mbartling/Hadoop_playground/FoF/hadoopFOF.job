#!/bin/bash
#
# Sample script for running Hadoop on eos.  
#
# This currently operates in two modes:
# 
# 1) interactive mode (INTERACTIVE=1) - start a vncserver with
#    a Firefox session pointing at some useful Hadoop urls.
#    Added to that is an xterm window for spawning new Hadoop jobs.
#    Finally, a xterm window which, when closed, will terminate the  
#    interactive session and stop Hadoop is started
#    (after running the examples below).
#
# 2) batch mode (INTERACTIVE=0) - if no vncserver command available or 
#    no VNC password set, then start HDFS/Hadoop, copy files to HDFS, 
#    run Hadoop app (e.g. MapReduce), copy files from HDFS, 
#    stop Hadoop/HDFS. 
#
#      IMPORTANT NOTE: ONLY ONE HADOOP JOB CAN BE ACTIVE AT A TIME!!!
#
#    If you run more than one Hadoop job at a time you can expect to 
#    have problems with at least one of the jobs.
#
# 
# PBS settings
#
#  - nodes must be >= 2 (2 is best for interactive, larger jobs need batch mode)
#  - ppn must be >= 8
#  - mem should be == nodes X 20gb
#  - walltime should be adjusted as needed
#  - Hadoop requires the use of -W x=NACCESSPOLICY:SINGLEJOB

#PBS -l nodes=2:ppn=8,mem=40gb,walltime=1:00:00
#PBS -W x=NACCESSPOLICY:SINGLEJOB
#PBS -N Hadoop_CFoF
#PBS -S /bin/bash
#PBS -m abe
#PBS -j oe
#PBS -M michael.bartling15+PBS@gmail.com

# module load python
export INTERACTIVE=0  # 0 = batch mode ; 1 = interactive VNC session

echo "##### BEGIN JOB FILE ###################################################"
cat $0 # remember the job run
echo "##### END JOB FILE #####################################################"
echo -e "\n\n\n\n\n" 
echo "##### BEGIN RUNNING JOB ################################################"
cd $PBS_O_WORKDIR # change directory to where we started
# output the current directory so it is clear where it ran from
echo "The current directory is `pwd`"
## cp the geninputs to the scratch drive generate a lot of inputs and put them on root
#cp ./src/geninputs $SCRATCH/geninputs
#cd $SCRATCH
#mkdir -p inputs
#./geninputs 10000000 >> ./inputs/file01
#
#cd $PBS_O_WORKDIR # change directory to where we started

. /g/software/hadoop/tamusc/conf/hadoop-env.sh # set up environment (to be moved to module file)

echo "##### Initialize Hadoop configuratin in $HADOOP_CONF_DIR"
init-hadoop-conf.sh

echo "##### Start Cluster"
start-hadoop-cluster.sh 

if [ $INTERACTIVE -eq 1 ] ; then # start an interactive VNC session
  start-hadoop-vnc.sh
  if [ $? -ne 0 ] ; then
    echo "Failed to start interactive VNC session... falling back to batch mode."
    let INTERACTIVE=0
  fi
fi

# get HDFS URL (needed to copy files to/from HDFS)
HDFS_URL="`grep hdfs: $HADOOP_CONF_DIR/core-site.xml | cut -f 2 -d '>' | cut -f 1 -d '<'`/"

# example of copying to HDFS
echo "##### Copying files to $HDFS_URL ($TAMUSC_USER_WORK_DIR)"
# use $HDFS_URL as a reference to copy files to hadoop file system
hadoop fs -put $0 $HDFS_URL # copy job file to HDFS root
#hadoop fs -put $0 / # this would accomplish the same thing

# example of running Hadoop (replace these with your Hadoop job(s))
# (from Fedora: https://fedoraproject.org/wiki/Changes/Hadoop#How_To_Test )
echo "##### Running MapReduce examples"
#MREXAMPLES=$TAMUSC_HADOOP_HOME/$TAMUSC_HADOOP_VERSION/hadoop-examples-$TAMUSC_HADOOP_VERSION.jar 
# output the current directory so it is clear where it ran from
echo "The current directory is `pwd`"
HADOOP_STREAMING=$TAMUSC_HADOOP_HOME/$TAMUSC_HADOOP_VERSION/contrib/streaming/hadoop-*streaming-*.jar
#CMINMAX= contrib/streaming/hadoop-*streaming-*.jar
#/g/software/hadoop/2.2.0/share/hadoop/tools/lib/hadoop-streaming-2.2.0.jar
#$TAMUSC_HADOOP_HOME/$TAMUSC_HADOOP_VERSION/share/hadoop/tools/lib/hadoop-*streaming-*.jar

#/g/software/hadoop/2.20.0
#hadoop jar /usr/share/hadoop/mapreduce/hadoop-*streaming-*.jar \
#hadoop jar $CMINMAX
#	-input inputs \
#	-output outputs \
#	-mapper src/myMap \
#	-combiner src/myCollect \
#	-reducer src/myReduce 
echo $TAMUSC_USER_WORK_DIR
echo "##### Generating MinMax inputs"
hadoop dfs -mkdir $TAMUSC_USER_WORK_DIR/inputs
# TODO
# cd $SCRATCH
cd $PBS_O_WORKDIR # change directory to where we started
#hadoop fs -put ./inputs/file01 $HDFS_URL/inputs/.
hadoop fs -copyFromLocal ./inputs/*.* $TAMUSC_USER_WORK_DIR/inputs

echo "##### Printing Environment variables"
env

echo "##### Printing Hadoop FS"
hadoop fs -lsr $HDFS_URL

echo "##### Running C Min Max"
# Note, the  -file command lets us package the executables and send 
# them to the hadoop cluster (can also do this with files

# Verify that the Map is working correctly, outputs should have multiple parts with the desired map output
# Success
cp -f ./src/*.out .
hadoop jar $HADOOP_STREAMING \
	-mapper "./map.out" \
	-reducer NONE \
    -file map.out \
    -file reduce.out \
    -input $TAMUSC_USER_WORK_DIR/inputs/ \
	-output $TAMUSC_USER_WORK_DIR/outputs/maponly
echo "##### Finished running MapReduce job"

echo "##### Copying Map outputs from $HDFS_URL ($TAMUSC_USER_WORK_DIR) to $PBS_O_WORKDIR"
hadoop fs -get $TAMUSC_USER_WORK_DIR/outputs/maponly $PBS_O_WORKDIR/outputs/$PBS_JOBID/map_outputs


hadoop jar $HADOOP_STREAMING \
	-mapper "./map.out" \
	-reducer "./reduce.out" \
    -file map.out \
    -file reduce.out \
    -verbose \
    -input $TAMUSC_USER_WORK_DIR/inputs/ \
	-output $TAMUSC_USER_WORK_DIR/outputs/reduceonly

# example of examining files on HDFS
echo "##### Files on $HDFS_URL"
hadoop fs -lsr $HDFS_URL
# hadoop fs -lsr / # this would accomplish the same thing

echo "#### Output of the Part"
hadoop fs -cat $HDFS/outputs/maponly/*
# example of copying files off of HDFS
echo "##### Copying files from $HDFS_URL ($TAMUSC_USER_WORK_DIR) to $PBS_O_WORKDIR"
mkdir -p outputs
hadoop fs -get /`basename $0` $PBS_O_WORKDIR/outputs/$PBS_JOBID # get job script we copied earlier
hadoop fs -get $TAMUSC_USER_WORK_DIR/outputs/reduceonly $PBS_O_WORKDIR/outputs/$PBS_JOBID/reduce_outputs
if [ $INTERACTIVE -eq 1 ] ; then # pause here until user ends VNC
  hold-hadoop-vnc-here.sh
fi

echo "##### Collect Debug Info"
./get-hadoop-debug.sh

echo "##### Stop Cluster"
stop-hadoop-cluster.sh

echo "##### END RUNNING JOB ##################################################"
# EOF
