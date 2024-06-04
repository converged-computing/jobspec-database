#PBS -q tng
#PBS -N Soret
#PBS -l select=1:ncpus=8
#PBS -l walltime=06:00:00

module load intel-suite mpi lammps

cp $PBS_O_WORKDIR/data.soret $TMPDIR
cp $PBS_O_WORKDIR/soret.in $TMPDIR

mpiexec lammps -in soret.in

mv $TMPDIR $PBS_O_WORKDIR/run
