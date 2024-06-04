#!/bin/bash -l
#SBATCH -J CESM2_SILHS_UWM_V2
#SBATCH -n 1   #total number of MPI ranks 
#SBATCH -N 1    #number of nodes requested
#SBATCH --tasks-per-node=1         #number of tasks per node
#SBATCH -t 00:20:00
#SBATCH -p dav
#SBATCH -A NTDD0004
#SBATCH --gres=gpu:v100:1       #kind of node and how many GPUs in it
#SBATCH -o gpu.out
#SBATCH --mem=0        #requesting all the available memory. slrum allocates a very little memory by default

#Total number of MPI ranks provided by n=N*tasks_per_node

#To see the PGI 19.4 custom modules
unset SLURM_MEM_PER_NODE
module use /glade/work/cponder/SHARE/Modules/Latest
module use /glade/work/cponder/SHARE/Modules/Legacy

module use --append /glade/work/cponder/SHARE/Modules/Bundles

for dir in /glade/work/cponder/SHARE/Modules/PrgEnv/*/*
do
    module use --append $dir
done

module purge
module load PrgEnv/PGI+OpenMPI/2019-04-30 
module load pgi
module load openmpi
ulimit -s unlimited
module list

export PGI_ACC_NOTIFY=2

echo $LD_LIBRARY_PATH
cd /glade/scratch/dennis/kernelOptimization/all/CESM2_SILHS_UWM_V2/v02

#To get the time executed on GPU
export PGI_ACC_TIME=1

#to use the IB channel
#export OMPI_MCA_btl_openib_if_include=mlx5_0

#command to execute the code if MPI was not built with srun support. (Custom built PGI 19.4 + OpenMPI was not built with slurm support)
#mpirun -n 8 ./atmosphere_model
srun ./kernel.exe

#command to execute the code if MPI was built with slurm support
#srun --mem=0 --mpi=pmix ./atmosphere_model

