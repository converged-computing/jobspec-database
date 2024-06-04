#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=venka210@umn.edu
#SBATCH --output=/home/shawa/venka210/simMutualism/reports/plotjob-%j.out

cd /home/shawa/venka210/simMutualism || return
module purge

module load matlab
matlab -nodisplay -r "generatePlots('basicSweep', 'figsBasicSweep', 'plotOutcomes', true)"
