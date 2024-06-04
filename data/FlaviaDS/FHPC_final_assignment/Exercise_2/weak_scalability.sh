#!/bin/bash

#SBATCH --job-name="script"
#SBATCH --output=log.out
#SBATCH --partition=EPYC
#SBATCH	--nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --nodelist=epyc[005]

module load architecture/AMD
module load openBLAS/0.3.21-omp
module load mkl

rm gemm_oblas.x
rm gemm_mkl.x

srun -n 1 make cpu

export OMP_NUM_THREADS=64
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

for i in {2000..20000..1000}
do
   for j in {1..10}
   do
       srun ./gemm_oblas.x $i $i $i >> weak_scalability/double/cores_spread_oblas.csv
       srun ./gemm_mkl.x $i $i $i >> weak_scalability/double/cores_spread_mkl.csv
       echo
   done
   echo
done
