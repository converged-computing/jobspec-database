#!/bin/bash -l
#PBS -l select=1:ncpus=20:mpiprocs=4:ngpus=4:ompthreads=5:mem=384
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N biasNN-r6 
#PBS -q gpu


source /projects/atomisticsimulations/Manyi/Miniconda3-DeepMD-2.1.5.env
cd $PBS_O_WORKDIR

nlims=15
times='times'
if [[ -f  $times ]]; then
   iters=$(tail -n 1 $times)
else
   iters=0
fi
if [[ $iters -lt $nlims ]] ;then
	sed -i s/ITER/$iters/g plumed.0.dat
	sed -i s/ITER/$iters/g plumed.1.dat
	sed -i s/ITER/$iters/g plumed.2.dat
	sed -i s/ITER/$iters/g plumed.3.dat

	if [ "$iters" -eq 0 ]; then
	mpirun -np 4 lmp -partition 4x1  -in in.bias.multi.lammps -v rest 1 -v ITER 0 1>> model_devi.log 2>> model_devi.log #rest 1 (no restart), o (yes restart)
  else
    if [ "$iters" -eq 1 ]; then
	    sed -i s/RESTART=NO/RESTART=YES/g plumed.0.dat
	    sed -i s/RESTART=NO/RESTART=YES/g plumed.1.dat
	    sed -i s/RESTART=NO/RESTART=YES/g plumed.2.dat
	    sed -i s/RESTART=NO/RESTART=YES/g plumed.3.dat
    fi
    mpirun -np 4 lmp -partition 4x1 -in in.bias.multi.lammps -v rest 0 -v ITER $iters 1>> model_devi.log 2>> model_devi.log  #rest 1 (no restart), o (yes restart)
		bash bck.meup.sh -i log.out
		bash bck.meup.sh -v log.lammps* |& tee -a log.out
		bash bck.meup.sh -v model_devi* |& tee -a log.out
	fi
	iters=$[iters+1]
	echo $iters >>$times
else
  exit
fi
qsub <./run-dplmp-franklin-multi-time-walks.sh


