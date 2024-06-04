#!/bin/bash
#PBS -k o
#PBS -M atalia.navarro.boullosa@gmail.com
#PBS -m abe
#Nombre del trabajo
#PBS -N simpleMC_ESO0140040
#Archivos de salida
#PBS -j oe
#
#Cola de ejecucion (-q cola)
#PBS -q mpi
#Recursos solicitados(nodos,cpus,mem,etc)
#PBS -l nodes=1:ppn=32

#Cargar y/o definir entorno
./miniconda3/bin/python
#To use MPI with the Intel compiler
#module load intel/12.1
#module load openmpi/intel

#Cambiar al directorio actual
#cd $PBS_0_WORKDIR
#export $PATH=/home/atalianb/SimpleMC_for_nested/
#Informacion del JOB
cd /home/atalianb/SimpleMC_for_nested/
echo =======================
echo Ejecutandose en:`hostname`
echo Fecha: `date`
echo Directorio:`pwd`
echo Recursos asignados_
echo `cat $PSB_NODEFILE`
NPROCS=`wc -l < $PBS_NODEFILE`
echo Total: $NPROCS cpus
echo ======================
cat $PBS_NODEFILE > $HOME/nodos
echo =====================
echo       SALIDA
echo =====================

#Inicia trabajo
python test.py
./a.out
#Termina trabajo
echo ==================
