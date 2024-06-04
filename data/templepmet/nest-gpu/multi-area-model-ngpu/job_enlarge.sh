#!/bin/sh
#----- qsub option -----
#PBS -q SQUID
#PBS --group=${GROUP_NAME}
#PBS -l elapstim_req=01:00:00
#PBS -o ./log/job_%s_out.txt
#PBS -e ./log/job_%s_err.txt
#PBS -b 4  # ノード数
#PBS -l cpunum_job=76  # 1ノードあたりのCPUコア数 <=76
#PBS -l gpunum_job=8   # 1ノードあたりのGPU使用台数 <=8
#PBS -v OMP_NUM_THREADS=8
#PBS -T openmpi
#PBS -v NQSV_MPI_MODULE=BaseGPU/2022
#----- Program execution -----

cd $PBS_O_WORKDIR

module load BaseGPU/2022

SINGULARITY_PWD=`mpirun -np 1 pwd`
SINGULARITY_IMAGE=../singularity/nestgpu.sif
# SINGULARITY_IMAGE=../singularity/nestgpu_sandbox
RESULT_FILE=./log/result.txt

if ls log/*.txt >/dev/null 2>&1
then
	mv log/*.txt log/old/
fi

date > $RESULT_FILE
echo "PBS_JOBID: $PBS_JOBID" >> $RESULT_FILE

N_SCALE=0.2
K_SCALE=0.117
T_SCALE=1.0
PBS_NNODES=4
PBS_NGPUS=8
LABEL=${PBS_NNODES}nodes_${PBS_NGPUS}gpus_N${N_SCALE}_K${K_SCALE}_T${T_SCALE}_${PBS_JOBID}
echo "{\"label\": \"$LABEL\", \"N_scale\": $N_SCALE, \"K_scale\": $K_SCALE, \"T_scale\": $T_SCALE}" > sim_info.json

time \
	singularity exec --nv --bind $SINGULARITY_PWD $SINGULARITY_IMAGE \
	python run_theory.py \
	>> $RESULT_FILE

time \
	mpirun $NQSV_MPIOPTS -np 32 -npernode 8 --map-by core --bind-to core --display-devel-map \
	./wrap_cuda.sh singularity exec --nv --bind $SINGULARITY_PWD $SINGULARITY_IMAGE \
	 python run_simulation.py \
	>> $RESULT_FILE

# time \
# 	mpirun $NQSV_MPIOPTS -np 32 -npernode 8 --map-by core --bind-to core --display-devel-map \
# 	./wrap_cuda.sh singularity exec --nv --bind $SINGULARITY_PWD $SINGULARITY_IMAGE \
# 	./wrap_nsys.sh python run_simulation.py \
# 	>> $RESULT_FILE
# mv *.nsys-rep simulation_result/$LABEL/

# REF_LABEL="1nodes_8gpus_0.01scale_ref"
# diff -sq simulation_result/$REF_LABEL/recordings simulation_result/$LABEL/recordings >> $RESULT_FILE

cp ./log/result.txt simulation_result/$LABEL/

# below "cp result.txt"
python analysis/each_proc/time.py simulation_result/$LABEL
python analysis/each_proc/memory.py simulation_result/$LABEL
python analysis/each_proc/neuron.py simulation_result/$LABEL
python analysis/each_proc/synapse.py simulation_result/$LABEL
python analysis/each_proc/spike.py simulation_result/$LABEL
python analysis/distributions/delay.py simulation_result/$LABEL
