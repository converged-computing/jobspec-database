#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load openmpi/4.0.0
export LD_LIBRARY_PATH=/opt/lammps-7Aug19_nnp_plumed/lib/nnp/lib
#sh run_custom_minimize.sh
mpirun -np 10 /opt/lammps-7Aug19_nnp_plumed/src/lmp_mpi < unbiased.lmp
echo "completed 1"
#mv COLVARnvt_7 batch3/COLVARnvt_7
#mv out7.lammpstrj batch3/


