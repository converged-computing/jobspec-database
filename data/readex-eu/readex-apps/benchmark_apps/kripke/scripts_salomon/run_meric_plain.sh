#!/bin/bash

#PBS -A OPEN-12-63
#PBS -q qprod
##PBS -q qexp
#PBS -N compare_kripke
#PBS -l select=1:ncpus=24:mpiprocs=24:accelerator=false
#PBS -l x86_adapt=true
#PBS -l walltime=02:00:00
#PBS -m be
###############################################################################

if [ "$PBS_ENVIRONMENT" == "PBS_BATCH" ]; then
	export FM_DIR=$PBS_O_WORKDIR
else
	export FM_DIR=$(pwd)
fi
cd $FM_DIR/../build

module purge
. ../readex_env/set_env_meric.source
. ../environment.sh

TMPFILE=kripke_meric_plain.tmp
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

REPEAT_COUNT=5
while [ $i -lt $REPEAT_COUNT ]; do
	$MERIC_ROOT/tools/staticMERICtool/multiNodeStaticMeasureStart.sh --rapl
	mpirun -n 24 ./kripke $KRIPKE_COMMAND | tee -a $TMPFILE
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

export MERIC_REGION_OPTIONS=meric.opts

while [ $i -lt $REPEAT_COUNT ]; do
	$MERIC_ROOT/tools/staticMERICtool/multiNodeStaticMeasureStart.sh --rapl
	mpirun -n 24 ./kripke $KRIPKE_COMMAND | tee -a $TMPFILE
	OUTPUT=$($MERIC_ROOT/tools/staticMERICtool/multiNodeStaticMeasureStop.sh --rapl)
	Evaluate
	i=$[ $i + 1 ]
done

echo "MERIC time:" $( awk "BEGIN {print $SUMTIME / $REPEAT_COUNT}" )
echo "MERIC CPUs energy:" $( awk "BEGIN {print $SUMCPUs / $REPEAT_COUNT}" )
echo "MERIC NODEs energy:" $( awk "BEGIN {print $SUMNODEs / $REPEAT_COUNT}" )

rm -f $TMPFILE

