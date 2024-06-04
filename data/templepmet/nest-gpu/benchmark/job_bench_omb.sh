#!/bin/sh
#----- qsub option -----
#PBS -q SQUID
#PBS --group=${GROUP_NAME}
#PBS -l elapstim_req=00:10:00
#PBS -o ./log/job_%s_out.txt
#PBS -e ./log/job_%s_err.txt
#PBS -b 2  # ノード数
#PBS -l cpunum_job=2  # 1ノードあたりのCPUコア数 <=76
#PBS -v OMP_NUM_THREADS=1
#PBS -T openmpi
#PBS -v NQSV_MPI_MODULE=BaseGPU/2022
#----- Program execution -----

cd $PBS_O_WORKDIR

module load BaseGPU/2022

SINGULARITY_PWD=`mpirun -np 1 pwd`
# SINGULARITY_IMAGE=../singularity/nestgpu.sif
SINGULARITY_IMAGE=../singularity/nestgpu_sandbox
RESULT_FILE=./log/result.txt

if ls log/*.txt >/dev/null 2>&1
then
	mv log/*.txt log/old/
fi

date > $RESULT_FILE
echo "PBS_JOBID: $PBS_JOBID" >> $RESULT_FILE

# singularity exec --nv --bind $SINGULARITY_PWD $SINGULARITY_IMAGE \
# 	./install_omb.sh

mpirun $NQSV_MPIOPTS -np 2 -npernode 2 --map-by core --bind-to core --display-devel-map \
	singularity exec --nv --bind $SINGULARITY_PWD $SINGULARITY_IMAGE \
	./osu-micro-benchmarks/build/mpi/one-sided/osu_get_bw \
	>> $RESULT_FILE

mpirun $NQSV_MPIOPTS -np 2 -npernode 1 --map-by core --bind-to core --display-devel-map \
	singularity exec --nv --bind $SINGULARITY_PWD $SINGULARITY_IMAGE \
	./osu-micro-benchmarks/build/mpi/one-sided/osu_get_bw \
	>> $RESULT_FILE

