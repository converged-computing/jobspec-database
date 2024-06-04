#!/bin/bash -l
#PBS -l walltime=0:10:00
#PBS -q debug-scaling 
#PBS -A radix-io
#PBS -l filesystems=home:grand:eagle

NNODES=`wc -l < $PBS_NODEFILE`

NWORKERS=$(( NNODES - 2))

NRANKS=2 # Number of MPI ranks to spawn per node
NDEPTH=16 # Number of hardware threads per rank (i.e. spacing between MPI ranks)

echo "WORKERS_NODES= ${NWORKERS}"
source  ~/spack/share/spack/setup-env.sh
spack env activate recup

LD_PRELOAD="/home/agueroudji/spack/opt/spack/linux-ubuntu22.04-skylake/gcc-11.4.0/darshan-runtime-3.4.4-kpqruvlpsxnx6bbam37q2loduoxnrmzt/lib/libdarshan.so"

cd $PBS_O_WORKDIR
SCHEFILE=scheduler.json

# Split nodes between the different steps
total=$(wc -l $PBS_NODEFILE | awk '{print $1}')

start_1=1
end_1=1

start_2=$(echo "${end_1}+1" | bc)
end_2=$(echo "${start_2}" | bc)

start_3=$(echo "${end_2}+1" | bc)
end_3=$(echo "${start_3}+${NWORKERS}-1" | bc)

start_4=$(echo "${end_3}+1" | bc)
end_4=$total

sed -n "${start_1},${end_1}p" $PBS_NODEFILE > SchedulerNode
sed -n "${start_2},${end_2}p" $PBS_NODEFILE > ClientNode
sed -n "${start_3},${end_3}p" $PBS_NODEFILE > WorkerNodes
sed -n "${start_4},${end_4}p" $PBS_NODEFILE > SimuNodes

echo launching Scheduler 
DARSHAN_ENABLE_NONMPI=1 DARSHAN_CONFIG_PATH="config.txt" LD_PRELOAD=$LIBDARSHAN mpiexec  -n 1 --ppn 1 -d ${NDEPTH} --hostfile SchedulerNode --exclusive --cpu-bind depth  dask scheduler --scheduler-file=$SCHEFILE 1>> scheduler.o  2>> scheduler.e  &

# Wait for the SCHEFILE to be created 
while ! [ -f $SCHEFILE ]; do
    sleep 3
    echo -n .
done

# Connect the client to the Dask scheduler
echo Connect Master Client  
DARSHAN_ENABLE_NONMPI=1 DARSHAN_CONFIG_PATH="config.txt" LD_PRELOAD=$LIBDARSHAN mpiexec  -n 1 --ppn 1  -d ${NDEPTH}  --hostfile ClientNode --exclusive --cpu-bind depth  `which python` image_processing.py --mode=distributed --scheduler-file=$SCHEFILE 1>> client.o 2>> client.e &

client_pid=$!

# Launch Dask workers in the rest of the allocated nodes 
echo Scheduler booted, Client connected, launching workers 

NPROC=$((NWORKERS * NRANKS))
DARSHAN_ENABLE_NONMPI=1 DARSHAN_CONFIG_PATH="config.txt" LD_PRELOAD=$LIBDARSHAN mpiexec  -n ${NPROC} --ppn ${RANKS}  -d ${NDEPTH}  --hostfile WorkerNodes --exclusive --cpu-bind depth  dask worker  --scheduler-file=$SCHEFILE 1>> worker.o 2>>worker.e  &

# Wait for the client process to be finished 
wait $client_pid
wait


                                                                                                                                                                                                 19,1          Top

