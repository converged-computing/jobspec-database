#!/bin/bash
#SBATCH --job-name=multi_species_sim
##SBATCH --account=blanca-dame8201
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --output=wsindy_sweep%A-%a.log 
#SBATCH --mail-user=dame8201@colorado.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=blanca-bortz                              
#SBATCH --qos=blanca-bortz                                   
##SBATCH --qos=preemptable                                  

#SBATCH --array=1-10

ml purge
module load matlab/R2019b

save_dr=/projects/dame8201/datasets/woundhealing/multi-species-02-03-22-000-110/
script=/projects/dame8201/datasets/woundhealing/multi-species-02-03-22-000-110/sim2ndorderSDE_multispecies.m

matlab -nodesktop -nodisplay -r "nu_cell = {0,0,0}; include_spec=[1 1 0]; v_opt = 1; run('${script}'); script_txt=fileread('${script}'); save(['${save_dr}','sim',date,'_000_110_','toc',strrep(num2str(toc),'.',''),'.mat']);"
