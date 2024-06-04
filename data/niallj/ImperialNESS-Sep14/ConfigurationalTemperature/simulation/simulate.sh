#PBS -q tng
#PBS -N Equilibration
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=8

module load intel-suite lammps mpi

cp $PBS_O_WORKDIR/equilibrate.in $TMPDIR
cp $PBS_O_WORKDIR/settings.in $TMPDIR

mpiexec lammps -in equilibrate.in

mv $TMPDIR $PBS_O_WORKDIR/equilibration_run
