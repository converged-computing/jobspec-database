#!/bin/bash
jtag=${1:-aurora}
ppn=${2:-12}
omp=${3:-8}
resq=R9376
ncpus=16

BIND8_S0_OPTIONS="--cpu-bind list:2-9,106-113:10-17,114-121:18-25,122-129:26-33,130-137:34-41,138-145:42-49,146-153"
BIND8_S1_OPTIONS="--cpu-bind list:54-61,158-165:62-69,166-173:70-77,174-181:78-85,182-189:86-93,190-197:94-101,198-205"
BIND8_S2_OPTIONS="--cpu-bind list:2-9,106-113:10-17,114-121:18-25,122-129:26-33,130-137:34-41,138-145:42-49,146-153:54-61,158-165:62-69,166-173:70-77,174-181:78-85,182-189:86-93,190-197:94-101,198-205"

input=NiO-fcc-S128-dmc.xml
input=input.batch.xml

mtag=`date "+%Y%m%d.%H%M"`
A21_JOBNAME="${mtag}.${jtag}"
qmcpack=/home/jeongnim/qmcpack.workspace/build_hpcm_c0308/bin/qmcpack


export MPI_BIND_OPTIONS=${BIND8_S2_OPTIONS}
export GPU_AFFINITY=/home/jeongnim/qmcpack.workspace/hpcm/gpu_mapper.sh


run_dir=${A21_JOBNAME}.N${ncpus}.p${ppn}x${omp}

mkdir -p ${run_dir}
cp ${input} ${run_dir}/
cd ${run_dir}
ln -s ../einspline.tile_2-2626-22-2-2.spin_0.tw_0.l0u3072.g112x66x66.h5 .
ln -s ../einspline.tile_2-2626-22-2-2.spin_1.tw_0.l0u3072.g112x66x66.h5 .

sdir=`pwd`
mpi=$(( ppn * ncpus ))

cat << EOF > job.sh
#PBS -S /bin/bash
#PBS -V
#PBS -l nodes=${ncpus}
#PBS -N ${jtag}

export KMP_BLOCKTIME=0
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

export MPIR_CVAR_ENABLE_GPU=0
export HYDRA_TOPO_DEBUG=1
export PALS_PMI=pmix

export LIBOMPTARGET_PLUGIN=LEVEL0
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1

export LIBOMP_USE_HIDDEN_HELPER_TASK=0
export LIBOMP_NUM_HIDDEN_HELPER_THREADS=0

export ZEX_NUMBER_OF_CCS=0:1,1:1,2:1,3:1,4:1,5:1

export LIBOMPTARGET_LEVEL0_COMPILATION_OPTIONS="-ze-opt-large-register-file"
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"

export LIBOMPTARGET_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LIST=all
export LIBOMPTARGET_LEVEL_ZERO_INTEROP_USE_IMMEDIATE_COMMAND_LIST=1
export LIBOMPTARGET_LEVEL0_USE_COPY_ENGINE=main

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=0
export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0:0

cd ${sdir}

function print_env()
{
  module list
  icpx -V
  sycl-ls
  numactl -H
  numastat -m
  ucs bios options get ${SLURM_NODELIST} memory_mode
  ucs bios options get ${SLURM_NODELIST} snc_enabled

  echo 'MKLROOT='${MKLROOT}
  echo 'Binary='${qmcpack}
  env
  ldd ${qmcpack}
  echo
}

print_env ${app} 2>&1 | tee -a env.out
cat \$PBS_NODEFILE >> env.out

mpiexec -np ${ncpus} -ppn 1 numactl -H 2>&1 | tee -a env.out

export OMP_NUM_THREADS=${omp} 
mpiexec ${MPI_BIND_OPTIONS} -np ${mpi} -ppn ${ppn} \
${GPU_AFFINITY} ${qmcpack}  ${input} --enable-timers=fine  2>&1 | tee -a qmcpack.out

rm -rf *.h5
rm -rf *bandinfo.dat

EOF

chmod +x job.sh

qsub -q ${resq} -l walltime=00:12:00 ./job.sh

cd -
