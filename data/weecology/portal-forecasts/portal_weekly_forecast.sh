#!/bin/bash
#SBATCH --job-name=portal_weekly_forecast
#SBATCH --mail-user=portal-forecasts-aaaaamelbeyabgcqol6s4p4cja@weecology.slack.com
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --time=12:00:00
#SBATCH --partition=hpg2-compute
#SBATCH --output=/orange/ewhite/PortalForecasts/portal_weekly_forecast_log.out
#SBATCH --error=/orange/ewhite/PortalForecasts/portal_weekly_forecast_log.err

echo "INFO: [$(date "+%Y-%m-%d %H:%M:%S")] Starting Weekly Forecast on $(hostname) in $(pwd)"
cd /orange/ewhite/PortalForecasts/

echo "INFO [$(date "+%Y-%m-%d %H:%M:%S")] Loading required modules"
source /etc/profile.d/modules.sh
module load git R singularity

echo "INFO [$(date "+%Y-%m-%d %H:%M:%S")] Updating singularlity container"
singularity pull --force docker://weecology/portalcasting

echo "INFO [$(date "+%Y-%m-%d %H:%M:%S")] Updating portal-forecasts repository"
rm -rf portal-forecasts
git clone https://github.com/weecology/portal-forecasts.git
cd portal-forecasts

echo "INFO [$(date "+%Y-%m-%d %H:%M:%S")] Running Portal Forecasts"
singularity run ../portalcasting_latest.sif Rscript PortalForecasts.R  2>&1 || exit 1

echo "INFO [$(date "+%Y-%m-%d %H:%M:%S")] Checking if forecasts were successful"
# Redirect stderr(2) to stdout(1) if command fails, and exit script with 1
singularity run ../portalcasting_latest.sif Rscript tests/testthat/test-successful_forecasts.R > ../testthat.log 2>&1 || exit 1

echo "INFO [$(date "+%Y-%m-%d %H:%M:%S")] Archiving to GitHub and Zenodo"
singularity run ../portalcasting_latest.sif bash archive_hipergator.sh

echo "INFO [$(date "+%Y-%m-%d %H:%M:%S")] Checking if archiving to GitHub was successful"
singularity run ../portalcasting_latest.sif Rscript tests/testthat/test-forecasts_committed.R > ../testthat.log 2>&1 || exit 1
