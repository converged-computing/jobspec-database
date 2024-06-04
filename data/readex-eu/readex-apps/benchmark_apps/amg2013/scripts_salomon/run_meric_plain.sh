#!/bin/bash

#PBS -A OPEN-12-63
#PBS -q qprod
#PBS -N amg2013_meric_plain
#PBS -o amg2013_meric_plain.out
#PBS -e amg2013_meric_plain.err
#PBS -l select=4:ncpus=24:mpiprocs=2:ompthreads=12:accelerator=true
#PBS -l x86_adapt=true
#PBS -l walltime=01:00:00
###############################################################################

if [ "$PBS_ENVIRONMENT" == "PBS_BATCH" ]; then
  LOC=$PBS_O_WORKDIR
else
  LOC=$(pwd)
fi

cd $LOC/..

TMPFILE=amg2013_meric_plain.tmp
rm -f $TMPFILE

# SCRIPT-SPECIFIC PART #########################################################
Evaluate (){
	TIME=$(grep -oP "(?<=Runtime \[s\] = )\d+\.\d+"  <<< $OUTPUT | head -1)
	SUMTIME=$( awk "BEGIN {print $SUMTIME + $TIME}" )
	CPUs=$(grep -oP "(?<=Overall CPUs energy consumption \[J\] = )\d+"  <<< $OUTPUT )
	NODEs=$(grep -oP "(?<=Overall nodes energy consumption \[J\] = )\d+"  <<< $OUTPUT )
	
	for VAL in $CPUs
	do
		SUMCPUs=$( awk "BEGIN {print $SUMCPUs + $VAL}" )
	done
	for VAL in $NODEs
	do
		SUMNODEs=$( awk "BEGIN {print $SUMNODEs + $VAL}" )
	done
}

# MERIC SETTINGS ###############################################################
export MERIC_MODE=3
export MERIC_FREQUENCY=0
export MERIC_UNCORE_FREQUENCY=0
export MERIC_NUM_THREADS=0

# RUN PLAIN APP ################################################################
SUMTIME=0.0
SUMCPUs=0.0
SUMNODEs=0.0
i=0

cd $LOC
./compile_for_plain.sh
cd ..
module purge
source ./readex_env/set_env_meric.source

REPEAT_COUNT=5
while [ $i -lt $REPEAT_COUNT ]; do
	$MERIC_ROOT/tools/staticMERICtool/multiNodeStaticMeasureStart.sh --rapl
  mpirun $BIND_TO_SOCKETS ./test/amg2013_plain -P 2 2 2 -r 40 40 40 2>&1 | tee -a $TMPFILE
	OUTPUT=$($MERIC_ROOT/tools/staticMERICtool/multiNodeStaticMeasureStop.sh --rapl)
	Evaluate
	i=$[ $i + 1 ]
done

echo "PLAIN time:" $( awk "BEGIN {print $SUMTIME / $REPEAT_COUNT}" )
echo "PLAIN CPUs energy:" $( awk "BEGIN {print $SUMCPUs / $REPEAT_COUNT}" )
echo "PLAIN NODEs energy:" $( awk "BEGIN {print $SUMNODEs / $REPEAT_COUNT}" )


# RUN TUNED APP ################################################################
SUMTIME=0.0
SUMCPUs=0.0
SUMNODEs=0.0
i=0

cd $LOC
./compile_for_meric.sh
cd ..
module purge
source ./readex_env/set_env_meric.source

export MERIC_REGION_OPTIONS=energy.opts

while [ $i -lt $REPEAT_COUNT ]; do
	$MERIC_ROOT/tools/staticMERICtool/multiNodeStaticMeasureStart.sh --rapl
  mpirun $BIND_TO_SOCKETS ./test/amg2013_meric -P 2 2 2 -r 40 40 40 2>&1 | tee -a $TMPFILE
	OUTPUT=$($MERIC_ROOT/tools/staticMERICtool/multiNodeStaticMeasureStop.sh --rapl)
	Evaluate
	i=$[ $i + 1 ]
done

echo "MERIC time:" $( awk "BEGIN {print $SUMTIME / $REPEAT_COUNT}" )
echo "MERIC CPUs energy:" $( awk "BEGIN {print $SUMCPUs / $REPEAT_COUNT}" )
echo "MERIC NODEs energy:" $( awk "BEGIN {print $SUMNODEs / $REPEAT_COUNT}" )

rm -f $TMPFILE

