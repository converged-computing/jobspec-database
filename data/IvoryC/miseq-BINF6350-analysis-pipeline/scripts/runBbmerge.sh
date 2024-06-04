#! /bin/bash

# $1 - first arg is a sample name, should match first column in MatchPairs/output/fileList.txt
# $2 - Path to bbmerge.sh that is ued IFF bbmerge.sh is not on the PATH.

# In case cluster job detached script from intended context
[ ${#PBS_O_WORKDIR} -gt 0 ] && cd $PBS_O_WORKDIR

# Find the MatchPairs module in this pipeline
PIPE=$(dirname $(dirname $PWD))
INDIR=$(ls -d $PIPE/*_MatchPairs/output)

# If bbmerge.sh is on the path, use it. 
# Surprisingly, "which" did not work in this docker container.
# Otherwise, take arg 2 as the path to the executable
bbmerge.sh --version && bbmergeExe=bbmerge.sh || bbmergeExe=$2 && $bbmergeExe --version

echo "Given sample: $1"
OUT=../output/$1.fastq.gz

REF=$INDIR/fileList.txt
echo "Referenceing table: $REF"

while read line; do
  ID=$(echo $line | awk '{print $1;}')
  R1=$(echo $line | awk '{print $2;}')
  R2=$(echo $line | awk '{print $3;}')
  if [ $ID == $1 ] 
  then
  	echo "R1: $R1"
  	echo "R2: $R2"
  	$bbmergeExe in1=$R1 in2=$R2 out=$OUT
  	# for faster tests add: reads=30
  fi
done < $REF

echo "done."
