#!/bin/bash -i
#
#PBS -N datamanip
#PBS -l select=1:ncpus=32:mpiprocs=32:mem=500gb:ngpus=2,
#PBS -l walltime=6:00:00
#PBS -o datamanip_output.txt
#PBS -j oe

module load matlab/2022a
matlab -sd "CPSC-8420-Machine-Learning-Project" -r "edited_datamanip(fin=\"data/repositories.csv\", fout=\"data/full_data\") ; exit" -logfile edited_datamanip_log.txt
