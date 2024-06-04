#!/bin/sh -x
#PBS -l select=256
#PBS -j oe
#PBS -l walltime=3:00:00
#PBS -A CSC249ADSE16_CNDA
#PBS -q EarlyAppAccess


module use /soft/modulefiles
module load oneapi/release/2023.12.15.001
cd ${PBS_O_WORKDIR}
JOBID=`echo ${PBS_JOBID} | cut -d'.' -f1`
export LD_PRELOAD=/opt/cray/pe/gcc/11.2.0/snos/lib64/libstdc++.so.6:/lib64/libc.so.6:/opt/cray/pe/gcc/11.2.0/snos/lib64/libgcc_s.so.1:/lib64/libz.so.1:/lib64/libm.so.6:/lib64/libdl.so.2:/lib64/librt.so.1:/soft/compilers/oneapi/2023.12.15.001/oneapi/compiler/2024.0/lib/libonnxruntime.1.12.22.721.so:/lib64/libpthread.so.0:/usr/lib64/libgmp.so.10:/lib64/libtinfo.so.6:/lib64/libreadline.so.7:/soft/libraries/intel-gpu-umd/stable_736_25_20231031/driver/lib64/libocloc.so:/usr/lib64/libctf.so.0:/usr/lib64/libbfd-2.39.0.20220810-150100.7.40.so

echo Jobid: $JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

NNODES=`wc -l < $PBS_NODEFILE`
export RANKS_PER_NODE=24           # Number of MPI ranks per node
NRANKS=$(( NNODES * RANKS_PER_NODE ))

echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"

#mpiexec -n ${NNODES} --ppn 1 ./check_script.sh
#sleep 5m
#tail output.* | grep "All 6 GPUs are okay" > good_nodes.txt
#awk NF=1 FS='[,:]' good_nodes.txt 2>&1 | tee new_hostfile.txt
#NNODES=`wc -l new_hostfile.txt | awk '{ print $1 }'`
#NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
#echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

 #mpiexec -n ${NNODES} -ppn 1 rm /tmp/*

 for input in *.inp ; do
      export GMSPATH=/home/tsatta/gamess

     mpiexec -n ${NNODES} -ppn 1 $GMSPATH/bin/my_ipcrm
 
name=${input%.inp}
inp=${name}.inp
log=${name}.log
ulimit -s unlimited
export OMP_DEV_HEAPSIZE=500000
export OMP_DISPLAY_ENV=T
export OMP_NUM_THREADS=8
export OMP_STACKSIZE=1G

scratch=./scratch
#mkdir -p $scratch
      export SCR=/tmp
      #export SCR=$scratch
      #cp ${inp} $SCR/JOB.$JOBID.F05
      mpiexec -n ${NNODES} -ppn 1 cp ${inp} $SCR/JOB.$JOBID.F05
# sleep 10
      echo ${inp} JOB.$JOBID.F05

      export JOB=JOB.$JOBID

      export USERSCR=$SCR
      export AUXDATA=$GMSPATH/auxdata/
      source $GMSPATH/gms-files.bash
export OUTPUT=$USERSCR/${JOB}.F06

# for debug mode
#export LIBOMPTARGET_DEBUG=2
export IGC_ForceOCLSIMDWidth=16
      export ZES_ENABLE_SYSMAN=1
    export FI_CXI_DEFAULT_CQ_SIZE=131072

# for 1 gpu
#  export ZE_AFFINITY_MASK=0.0

# reduce MPI overhead
export MPIR_CVAR_ENABLE_GPU=0

# sleep 10
       mpiexec -n ${NRANKS} -ppn ${RANKS_PER_NODE}  \
    --cpu-bind verbose,depth -d ${OMP_NUM_THREADS} \
    gpu_tile_compact.sh  $GMSPATH/gamess.00.x >& $log

 #   mv ${JOB}.F06 $log

      # in the case of multiple nodes, this would hav to be done on each node. Since we're
      # focusing on single node jobs here, the following should be ok.
#      rm $USERSCR/JOB.*
#      rm $SCR/JOB.*
#      rm /tmp/JOB*
#      rm JOB.$JOBID.dat
#      rm -f JOB.$JOBID.trj
#      rm JOB.$JOBID.F*
#      rm -f JOB.$JOBID.rst
mpiexec -n ${NNODES} -ppn 1 rm /tmp/JOB*

#rm /tmp/*
      # cleaning up semaphores and distributed arrays if necessary
     mpiexec -n ${NNODES} -ppn 1 $GMSPATH/bin/my_ipcrm
 done
