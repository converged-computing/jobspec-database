#!/bin/bash
#PBS -N example
#PBS -j oe
#PBS -l select=1:aoe=quad_100
#PBS -l walltime=00:30:00
#PBS -A <YOUR-BUDGET-CODE>

cd $PBS_O_WORKDIR

# job size (edit as needed)
export NODES=1
(( ncores = NODES * 64 ))

# load GPAW stack
source /path/to/gpaw-stack/load.sh

# affinities and threading
export KMP_AFFINITY=balanced,granularity=fine
export KMP_HW_SUBSET=1T
export I_MPI_PIN_ORDER=compact
export OMP_NUM_THREADS=1

# TBB + huge pages
module load craype-hugepages2M
export LD_PRELOAD=$TBBROOT/lib/intel64/gcc4.7/libtbbmalloc_proxy.so.2:$TBBROOT/lib/intel64/gcc4.7/libtbbmalloc.so.2
export TBB_MALLOC_USE_HUGE_PAGES=1

# run 4 times to get some statistics
out=summary
echo "ID time" >> $out

for ((i=0; i<4; i++))
do
    id=$(( i + 1 ))
    echo $id
    echo "aprun -n $ncores gpaw-python input.py"
    aprun -n $ncores gpaw-python input.py
    time=$(grep SCF-cycle: output.txt | awk '{ print $2 }')
    mv -i output.txt output-${id}.txt
    echo "$id $time" >> $out
done

