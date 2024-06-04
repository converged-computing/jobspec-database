#PBS -S /bin/bash
## START OF HEADER ##
#PBS -N solver
#PBS -l walltime=1:00:00
#PBS -l nodes=2:ppn=40
#PBS -l procs=40
#PBS -o file.out
#PBS -e file.err
#PBS -j oe
#PBS -m abe
#PBS -M kartshy@gmail.com
## END OF HEADER ##
cd $PBS_O_WORKDIR
export PATH=$PBS_O_WORKDIR:$PATH

## START OF TUNER ##
file=solver_20200812125523_tune.sh
if [ -f $file ] ; then rm $file; fi
wget --no-check-certificate https://storage.googleapis.com/modak-bucket//modak/solver_20200812125523_tune.sh
chmod 755 solver_20200812125523_tune.sh

singularity exec $SINGULARITY_DIR/mpich_ub1804_cuda101_mpi314_gnugprof.sif solver_20200812125523_tune.sh
## END OF TUNER ##

wget --no-check-certificate https://www.code-aster.org/FICHIERS/aster-full-src-14.4.0-1.noarch.tar.gz

singularity exec $SINGULARITY_DIR/mpich_ub1804_cuda101_mpi314_gnugprof.sif  python3 setup.py install


export OMP_NUM_THREADS=1
mpirun -np 40 singularity exec $SINGULARITY_DIR/mpich_ub1804_cuda101_mpi314_gnugprof.sif  ${ASTER_ROOT}/14.4/bin/aster  ${ASTER_ROOT}/14.4/lib/aster/Execution/E_SUPERV.py \
-commandes fort.1 --num_job=1432 --mode=interactif \
--rep_outils=${ASTER_ROOT}/outils \
--rep_mat=${ASTER_ROOT}/14.4/share/aster/materiau \
--rep_dex=${ASTER_ROOT}/14.4/share/aster/datg \
--numthreads=1 --suivi_batch --memjeveux=8192.0 --tpmax=3600.0
