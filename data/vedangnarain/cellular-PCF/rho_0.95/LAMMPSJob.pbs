#!	/bin/bash
#PBS	-l	nodes=1:ppn=1,walltime=1:00:00
#PBS	-q	gpu
#PBS	-m	ae
#PBS	-o	out.log
#PBS	-e	err.log
#PBS	-N	density_USERINPUTRHO

module load PrgEnv-gnu
module load boost/1.65.0
module load gsl
module load cudatoolkit/7.0.28-1.0502.10742.5.1
module swap PrgEnv-cray/5.2.82 PrgEnv-gnu/5.2.82
module load lammps/gnu/gpu/15May15

cd	$PBS_O_WORKDIR

aprun -n 1 lmp_xe6 < in.lj.fluid
