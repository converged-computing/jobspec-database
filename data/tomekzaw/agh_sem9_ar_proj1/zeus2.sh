#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 20
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid
#SBATCH --account=plgmpr21zeus

module add plgrid/tools/python-intel/3.6.5 2>/dev/null

zad=2

mpiexec -np 20 python3 main.py 100 $zad > /dev/null

echo "i,zad,N,np,real_N,time"

for i in {1..10} ; do
    for N in 2000 300 50 ; do
        for np in {1..20} ; do
            echo -n "$i,$zad,$N,$np,"
            mpiexec -np $np python3 main.py $N $zad
        done
    done
done
