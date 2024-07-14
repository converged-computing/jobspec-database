#!/bin/bash 

path=$1 
cd "${path}"
for atom in $(ls -d  ${HOME}/pegas/input_geometries/*/ | xargs -n 1 basename); do
cd $atom;
pwd
for displacement in $(ls); do
cd $displacement
pwd
mkdir gif
mkdir log
#mkdir folder_slurm
mv *.gif ./gif 
mv *.log ./log
#mv slurm* ./folder_slurm
cd ..
done
cd ..
done
