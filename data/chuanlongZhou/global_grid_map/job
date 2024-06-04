#!/bin/bash
#PBS -N Global_grid
#PBS -m a
#PBS -j oe
#PBS -q xlongp
#PBS -o Global_grid.o
#PBS -S /bin/bash
#PBS -v BATCH_NUM_PROC_TOT=8
#PBS -l nodes=1:ppn=8


module load python/3.9

cd /home/satellites8/czhou/global_grid_map

python export_html.py


