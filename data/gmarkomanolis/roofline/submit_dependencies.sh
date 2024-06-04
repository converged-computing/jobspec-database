#!/bin/bash
#SBATCH --job-name=lu_c_16
#SBATCH --nodes=1
#SBATCH --time=02:30:00
#SBATCH --output=lu_c_16.out
#SBATCH --error=lu_c_16.err

module swap PrgEnv-cray/5.2.82 PrgEnv-intel
module load advisor/2018.1.1.535164
module swap intel/15.0.2.164 intel/17.4.4.196

srun -n 16 --hint=nomultithread --threads-per-core=1 --multi-prog ./config_dependencies.txt

exit 0
