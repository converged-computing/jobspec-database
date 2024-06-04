#!/bin/bash -e

#SBATCH --account=vuw03073                      #Type the project code you want to use for this analyis        $
#SBATCH --job-name=bigmomma_assembly_test                #This willl be the name appear on the queue                    $
#SBATCH --mem=30G                                #Amount of memory you need                                     $
#SBATCH --cpus-per-task=2                       #Amount of CPUs (logical)                                      $
#SBATCH --time=02:00:00                         #Duration dd-hh:mm:ss                                          $
#SBATCH --output=slurmOut/bigmomma_assembly_test.%j.txt  #Name of the output file                                       $
#SBATCH --mail-type=ALL                         #This will send you an email when the STARTS and ENDS          $
#SBATCH --mail-user=calquigs@gmail.com          #Enter your email address.                                     $
#SBATCH --profile=task
#SBATCH --array=13                     # Array jobs
#SBATCH --export NONE

export SLURM_EXPORT_ENV=ALL

#purging any modules/software loaded in the background prior to submitting script.(recommended)                $
module purge

module load Miniconda3
source activate opendrift_simon
echo wtf

regions=('taranaki' 'waikato' '90milebeach' 'northland' 'hauraki' 'bay_o_plenty' 'east_cape' 'hawkes_bay' 'wairarapa' 'wellington' 'marlborough' 'kahurangi' 'west_coast' 'fiordland' 'southland' 'stewart_isl' 'otago' 'canterbury' 'kaikoura' 'chatham' 'auckland_isl')


python /nesi/project/vuw03073/testScripts/matrix_assembly.py ${regions[${SLURM_ARRAY_TASK_ID}]}

