#!/bin/bash 

#PBS -A OPEN-12-63
#PBS -q qprod
#PBS -N KRIPKE_meric
#PBS -l select=1:ncpus=24:mpiprocs=24:accelerator=false
#PBS -l x86_adapt=true
#PBS -l walltime=24:00:00
#PBS -m be


hostname
if [ "$PBS_ENVIRONMENT" == "PBS_BATCH" ]; then
	export FM_DIR=$PBS_O_WORKDIR
else
	export FM_DIR=$(pwd)
fi
cd $FM_DIR/../build

# load necessary modules
. ../readex_env/set_env_meric.source
. ../environment.sh

################################################################################

# SET MERIC OUTPUT FORMAT
export MERIC_MODE=1
export MERIC_COUNTERS=papi
export MERIC_CONTINUAL=1
export MERIC_DETAILED=1
export MERIC_OUTPUT_DIR=$SCRATCH/DELETE

export MERIC_FREQUENCY=25
export MERIC_UNCORE_FREQUENCY=25
export MERIC_NUM_THREADS=0
export MERIC_OUTPUT_FILENAME=$MERIC_FREQUENCY"_"$MERIC_UNCORE_FREQUENCY"_CONFIG"


# RUN ONE TEST
mpirun -n 24 ./kripke $KRIPKE_COMMAND

export MERIC_OUTPUT_DIR=$SCRATCH/KRIPKEiccFIN


# FOR EACH SETTINGS
for proc in 24
do
	for thread in 0
	do
		for cpu_freq in 25 {24..12..2} 
		do
			for uncore_freq in {30..12..2} 
			do
				# OUTPUT NAMES
				export MERIC_OUTPUT_FILENAME=$cpu_freq"_"$uncore_freq"_CONFIG"

				# TEST SETTINGS
				export MERIC_FREQUENCY=$cpu_freq
				export MERIC_UNCORE_FREQUENCY=$uncore_freq

				echo "Output file: "  $MERIC_OUTPUT_FILENAME >> LOGmeric
				echo 
				
				mpirun -n 24 ./kripke $KRIPKE_COMMAND | tee -a LOGmeric

			done
		done
	done
done



