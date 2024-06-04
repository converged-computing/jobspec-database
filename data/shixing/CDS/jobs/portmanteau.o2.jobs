#!/bin/bash
#PBS -l nodes=20:ppn=1
#PBS -l pmem=4g
#PBS -l walltime=2:00:00
#PBS -q isi

# generate the portmanteau analogy pairs
cd /home/nlg-05/xingshi/workspace/misc/CDS/py

mpiexec python -m portmanteau.analogy_mpi ../config/hpc.google.o2.cfg &> /home/nlg-05/xingshi/workspace/misc/CDS/log/portmanteau.o2.log.txt
