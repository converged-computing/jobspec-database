#!/bin/bash
### innanzitutto: PORCODDIO PBS
#PBS -A DD-23-44
#######PBS -q qgpu_eurohpc
#PBS -q qcpu
#######PBS -l select=1:ngpus=8:ncpus=128  
#PBS -l select=1:ncpus=128  
#PBS -l walltime=00:20:00
#PBS -N PLUTO_extrae
#PBS -m n
#PBS -M m.mencagli@cineca.it

## load modules
source ${HOME}/modules_files/pluto_mod ex

## print gpu info
#nvidia-smi

## run the pluto tests
#cd programming/gpluto_cpp/
cd programming/gpluto_EXTRAE/
#python3 test_pluto.py --auto-update
#python3 test_pluto.py --start-from 9:0

## run custom
#cd Test_Problems/ResRMHD/Blast   
cd Test_Problems/MHD/Orszag_Tang/

time make -j -B > makeout_prova 2>&1     

mkdir -p ex_mio/
cd ex_mio/
rm -rf * 
cp -v ../pluto .
cp -v ../pluto.ini .
cp -v $EXTRAE_HOME/share/example/MPI/extrae_explained.xml .
cp -v $EXTRAE_HOME/share/example/MPI/ld-preload/trace.sh . 

#nss profile --trace=cuda,mpi,nvtx,openacc mpirun -np 8 ./pluto    
#mpirun -np 8 ./pluto 
mpirun -np 2 ./trace.sh ./pluto
                                                                                                                                        
