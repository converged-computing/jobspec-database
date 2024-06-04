#!/bin/bash -e                                                                                                    

#SBATCH --account=vuw03073			#Type the project code you want to use for this analyis                       
#SBATCH --job-name=robustness		#This willl be the name appear on the queue                                                                                    
#SBATCH --mem=150MB				#Amount of memory you need                                                                                                  
#SBATCH --cpus-per-task=2               	#Amount of CPUs (logical)                                                                          
#SBATCH --time=00:15:00				#Duration dd-hh:mm:ss                                                                                           
#SBATCH --output=slurmOut/robustness.%j.txt	#Name of the output file                                                                              
#SBATCH --mail-type=ALL				#This will send you an email when the STARTS and ENDS                                                                                           
#SBATCH --mail-user=calquigs@gmail.com		#Enter your email address.                                                                                               
#SBATCH --profile=task
#SBATCH --array=0-143
#SBATCH --export NONE

export SLURM_EXPORT_ENV=ALL

#purging any modules/software loaded in the background prior to submitting script.(recommended)                   
module purge

files=(reinga_var/*)
file=${files[$SLURM_ARRAY_TASK_ID]}

module load Miniconda3
source activate opendrift_simon

python /nesi/project/vuw03073/opendriftxmoana/postprocessing/nparticle_robustness.py ${file}






