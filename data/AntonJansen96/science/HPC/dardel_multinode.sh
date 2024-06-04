#!/bin/bash
#SBATCH --time=0-0:10:00
#SBATCH --nodes=2
#SBATCH --partition=main
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=bench2
#SBATCH --account=snic2021-1-38
#SBATCH --mail-user=anton.jansen@scilifelab.se
#SBATCH --mail-type=ALL

# Load modules
ml PDC/21.09 
ml all-spack-modules/0.16.3
ml CMake/3.21.2

# Compile GROMACS
# mkdir build
# cd build
# cmake ~/Private/gromacs-2021 -DGMX_MPI=ON -DGMX_USE_RDTSCP=ON -DCMAKE_INSTALL_PREFIX=${PWD}/.. -DGMX_BUILD_OWN_FFTW=ON
# make -j 12
# make install -j 12
# cd ..
# rm -r build
source ${PWD}/bin/GMXRC

# GLIC_2n_64_4
# srun gmx_mpi grompp -f MD.mdp -c CA.pdb -p topol.top -n index.ndx -o MD.tpr
srun gmx_mpi mdrun -deffnm MD -npme 0 -g GLIC_2n_64_4_DLBNO.log -resetstep 20000 -ntomp 4 -dlb yes -pin on -pinstride 2
