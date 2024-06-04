#15/03/2023 version, Francesco Mambretti

#!/bin/bash -l
#PBS -l select=1:ncpus=4:mpiprocs=1:ngpus=1:ompthreads=2
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -N bias384_$N_NH2_$NUMBER
#PBS -q gpu

source /projects/atomisticsimulations/Manyi/Miniconda3-DeepMD-2.1-test.env
cd $PBS_O_WORKDIR

#variables
bias=200
dump_freq=2000

#run multiple times
nlims=3
times='times'

if [[ -f  $times ]]; then
   iters=$(tail -n 1 $times)
else
   iters=0
	echo $iters
	echo "first round"
fi

if [[ $iters -lt $nlims ]] ;then
	if [ "$iters" -eq 0 ]; then
		lmp -i in.bias.surf.lammps -v rest 1 -v ITER 0 1>> model_devi.log 2>> model_devi.log  #rest 1 (no restart), o (yes restart)
		sed -i 's/$BIAS/'"$bias"'/' plumed.dat
		sed -i 's/V_DUMP/'"$dump"'/' plumed.dat 
	else
		lmp -i in.bias.surf.lammps -v rest 0 -v ITER $iters 1>> model_devi.log 2>> model_devi.log  #rest 1 (no restart), o (yes restart)
	fi
	iters=$(($iters+1))
  echo $iters >>$times
else
  exit
fi

qsub chain_sub_dplmp_franklin.bias.surf.sh
