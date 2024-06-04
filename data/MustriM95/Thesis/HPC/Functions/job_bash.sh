#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=1:mem=20gb
echo "load module"
module load anaconda3/personal
echo "copy files"
cp $HOME/foldername/* .
echo "tru julia"
julia $HOME/foldername/filename.jl
echo "move results home "
mv result_* $HOME/foldername
echo $SECONDS
# end file
