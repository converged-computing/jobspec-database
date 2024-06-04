#!/bin/bash -x
#SBATCH -p defq # partition (queue)
#SBATCH -N 1
#SBATCH --ntasks-per-node 1   # number of cores
#SBATCH --cpus-per-task=1
#SBATCH -t 0-4:00 # time (D-HH:MM)
#SBATCH -o nicename%j.out # STDOUT

ulimit -s 10240

module load boost-1.73.0
module load boost-mpi-1.73.0


export OMPI_MCA_pml="^ucx"
export OMPI_MCA_btl_openib_if_include="mlx5_4:1"

#export OMPI_MCA_pml="^ucx"
#export OMPI_MCA_btl_openib_if_include="mlx5_1:1"
export OMPI_MCA_btl_tcp_if_exclude=docker0,docker_gwbridge,eno1,eno2,lo,enp196s0f0np0,enp196s0f1np1,ib0,ib1,veth030713f,veth07ce296,veth50ead6f,veth73c0310,veth9e2a12b,veth9e2cc2e,vethecc4600,ibp65s0f1,enp129s0f0np0,enp129s0f1np1,ibp65s0f0
export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_mpi_cuda_support=0

mpirun -np $SLURM_NTASKS build/debug/csr ~/UiB-INF339/matrices/ljournal-2008.mtx

ln -f nicename${SLURM_JOB_ID}.out ljournal-2008${SLURM_NTASKS}.out
rm nicename${SLURM_JOB_ID}.out
