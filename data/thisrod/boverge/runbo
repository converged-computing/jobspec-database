#!/bin/sh
# specify the queue name
#PBS -q largemem
# resource requests
#PBS -l nodes=1:ppn=32
#PBS -l walltime=6:00:00

# run process
module load matlab/R2015b
cd $HOME/bisection
/usr/bin/time matlab -r "parpool(32); xpsetup; BoseOne; quit"