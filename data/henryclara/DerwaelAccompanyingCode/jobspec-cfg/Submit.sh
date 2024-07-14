#!/bin/bash
#SBATCH -o Logs/SLURM_job.%j.%N.out
#SBATCH -e Logs/SLURM_job.%j.%N.err
#SBATCH -D .
#SBATCH -J Forward
#SBATCH --get-user-env
#SBATCH --account=bm1164
#SBATCH --mail-user=clara.henry@mpimet.mpg.de
#SBATCH --ntasks=80
#SBATCH --time=08:00:00
#SBATCH --partition=compute
#SBATCH --export=NONE
#=================================================================================================================
spack load intel-oneapi-mpi@2021.5.0%intel@2021.5.0
#spack load intel-mpi@2019.10.317
spack load intel-oneapi-compilers@2022.0.1
spack load intel-oneapi-mkl@2022.0.1%gcc@11.2.0
spack load metis@5.1.0%intel@2021.5.0

export OMPI_MCA_pml="ucx"
export OMPI_MCA_btl=self
export OMPI_MCA_osc="pt2pt"
export UCX_IB_ADDR_TYPE=ib_global
# for most runs one may or may not want to disable HCOLL
export OMPI_MCA_coll="^ml,hcoll"
export OMPI_MCA_coll_hcoll_enable="0"
export HCOLL_ENABLE_MCAST_ALL="0"
export HCOLL_MAIN_IB=mlx5_0:1
export UCX_NET_DEVICES=mlx5_0:1
export UCX_TLS=mm,knem,cma,dc_mlx5,dc_x,self
export UCX_UNIFIED_MODE=y
export HDF5_USE_FILE_LOCKING=FALSE
export OMPI_MCA_io="romio321"
export UCX_HANDLE_ERRORS=bt

ulimit -s 102400
ulimit -c 0

set -e
echo Here comes the Nodelist:
echo $SLURM_JOB_NODELIST

echo Here comes the partition the job runs in:
echo $SLURM_JOB_PARTITION
echo "Current directory:" $PWD
echo "Going into slurm submit dir."
cd $SLURM_SUBMIT_DIR
echo "Current directory:" $PWD
source ModulesPlusPaths2LoadIntelMPI.sh
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so

cp $ELMER_HOME/share/elmersolver/lib/FreeSurfaceSolver.so src/MyFreeSurfaceSolver.so
#echo $ELMER_HOME
#echo $ELMER_SOLVER_HOME

Counter=$1
echo Counter is: $Counter
if [ "${Counter}" -lt "10" ]; then
        CounterFormatted=$(printf %06d $Counter)
        Counter=$(($Counter+1))
        CounterFormattedNew=$(printf %06d $Counter)
	echo "Current directory:" $PWD
	echo "Next step: copy For.sif.bak"
        cp Forward.sif.bak Forward.sif
        sed -i "s/START/${CounterFormatted}/g" Forward.sif
        sed -i "s/END/${CounterFormattedNew}/g" Forward.sif
        echo $Counter
        make compile
        make ini
        make grid
	srun -l --mpi=pmi2 --export=ALL --cpu_bind=cores --distribution=block:cyclic -n 80 ElmerSolver_mpi Forward.sif
	echo "Done running Elmer"
	echo "Submitting next job"
	sbatch Submit.sh $Counter
	echo "Job submitted"
fi

