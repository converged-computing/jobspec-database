#PBS -S /bin/bash
#PBS -q home
#PBS -N lammps
#PBS -l nodes=1:ppn=16:rack13
#PBS -l walltime=96:00:00
#PBS -o output2500_20x.txt
#PBS -e error2500_20x.txt
#PBS -V
#PBS -M achandrasekaran@ucsd.edu
#PBS -m abe
#PBS -A rangamani-hopper
cd /oasis/tscc/scratch/achandrasekaran/LAMMPS_codes/cross_link/fil_6/
source ~/.bashrc.mine
mpiexec -np 16 ~/mylammps/build/lmp -in lj_drop_20x.in
