#!/bin/sh

#PBS -A OPEN-12-63
#PBS -q qexp
#PBS -N KRIPKE_SAF
#PBS -l select=1:ncpus=24:mpiprocs=24:accelerator=false
#PBS -l x86_adapt=true
#PBS -l walltime=01:00:00
#PBS -m be

if [ "$PBS_ENVIRONMENT" == "PBS_BATCH" ]; then
	export FM_DIR=$PBS_O_WORKDIR
else
	export FM_DIR=$(pwd)
fi
cd $FM_DIR/../build

. ../readex_env/set_env_saf.source
. ../environment.sh

export SCOREP_FILTERING_FILE=scorep.filt

rm -rf scorep-*
rm -f old_scorep.filt
echo "" > scorep.filt

if [ "$READEX_INTEL" == "1" ]; then
  rm -rf old_scorep_icc.filt
  echo "" > scorep_icc.filt
fi


iter=0
result=1
while [ $result != 0 ]; do
  iter=$[ $iter +1 ]
  echo "result = "$result
  # run the application.. update this for different applications
  mpirun -n 24 ./kripke $KRIPKE_COMMAND
  echo "kripke done."
  ./do_scorep_autofilter_single.sh 0.1 #$1
  result=$?
  echo "scorep_autofilter_singe done ($result)."
done

if [ "$READEX_INTEL" == "1" ]
then
	echo ".*:*__gnu_cxx::* off" >> scorep_icc.filt
	echo ".*:*std::* off" >> scorep_icc.filt
fi

echo "end." 

#cp scorep.filt ../RESULTS/
