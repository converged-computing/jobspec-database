
#!/bin/bash
#PBS -A qris-uq
#PBS -l walltime=3:15:00
#PBS -l select=1:ncpus=6:ompthreads=3:mem=100GB

cd $PBS_O_WORKDIR

module load R/3.5.0-intel gurobi/9.0.2

R CMD BATCH "/gpfs1/scratch/30days/uqibrito/Project05a_RafaelaSA/scripts/R_scripts/SA_06a-07_Prioritizr_RUN.R"
