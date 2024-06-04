#!/bin/bash
#SBATCH --job-name=4mpi-alt          # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=1      # tasks per nodes
#SBATCH --cpus-per-task=16       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)

module purge

module load compilers/gcc/13 libraries/openmpi/5.0.3/gcc-13 tools/cmake

# add tcl and fftw that is was built with to path
export LD_LIBRARY_PATH="$HOME/build/NAMD/NAMD_3.0b6_Source/tcl/lib:$HOME/build/NAMD/NAMD_3.0b6_Source/fftw/lib:$LD_LIBRARY_PATH"

# add charmrun to path (so we can use net/mpi or whatever it was built with)
export PATH="$HOME/build/NAMD/NAMD_3.0b6_Source/charm-v7.0.0/bin:$PATH"

# write to correct output file
#sed -i -e 's:outputName.*:outputName          /home/ogooberman/ukscc-team-4/NAMD/outputs/4mpi-alternative-input:g' $HOME/build/stmv/stmv.namd

# This is used to launch SMP builds
mpirun ~/build/NAMD/NAMD_3.0b6_Source/Linux-ARM64-g++/namd3 +ppn 15 +pemap 1-15 +commap 0 ~/build/apoa1/apoa1.namd

# Launch a non SMP build
# mpirun -np 16 ~/build/NAMD/NAMD_3.0b6_Source/Linux-ARM64-g++/namd3 ~/build/stmv/stmv.namd