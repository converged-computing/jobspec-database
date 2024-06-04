#!/bin/bash -x
#SBATCH -p rome16q # partition (queue)
#SBATCH -N 1
#SBATCH --ntasks 8   # number of cores
#SBATCH -t 0-4:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

ulimit -s 10240

module purge
module load slurm/21.08.8
module load gcc/10.2.0
module load openmpi/gcc/64/4.1.5
module load metis


export OMPI_MCA_pml="^ucx"
export OMPI_MCA_btl_openib_if_include="mlx5_4:1"

#export OMPI_MCA_pml="^ucx"
#export OMPI_MCA_btl_openib_if_include="mlx5_1:1"
export OMPI_MCA_btl_tcp_if_exclude=docker0,docker_gwbridge,eno1,eno2,lo,enp196s0f0np0,enp196s0f1np1,ib0,ib1,veth030713f,veth07ce296,veth50ead6f,veth73c0310,veth9e2a12b,veth9e2cc2e,vethecc4600,ibp65s0f1,enp129s0f0np0,enp129s0f1np1,ibp65s0f0
export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_mpi_cuda_support=0



mpirun -np $SLURM_NTASKS build/Debug/1b ~/UiB-INF339/matrices/delaunay_n24.mtx
