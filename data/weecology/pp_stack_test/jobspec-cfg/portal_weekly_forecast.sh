#!/bin/bash
#SBATCH --job-name=portal_weekly_forecast
#SBATCH --mail-user=jsimonis@ufl.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --ntasks=1
#SBATCH --mem=2gb
#SBATCH --time=3:00:00
#SBATCH --partition=hpg2-compute
#SBATCH --output=test_log.out
date;hostname;pwd

source /etc/profile.d/modules.sh

module load git R singularity

rm -f portal_predictions_latest.sif
rm -rf pp_stack_test
rm -rf forecasts
singularity pull docker://weecology/portal_predictions
git clone https://github.com/weecology/pp_stack_test.git
cd pp_stack_test
singularity run ../portal_predictions_latest.sif Rscript stack_test.R
singularity run ../portal_predictions_latest.sif Rscript tests/testthat.R > ../testthat.log 2>&1
singularity run ../portal_predictions_latest.sif bash archive_hipergator.sh