#!/bin/bash -e

#SBATCH --account=vuw03073                      #Type the project code you want to use for this analyis		 $
#SBATCH --job-name=all_settlement_test          #This willl be the name appear on the queue			 $
#SBATCH --mem=3G                               #Amount of memory you need                                       $
#SBATCH --time=04:00:00                         #Duration dd-hh:mm:ss                                            $
#SBATCH --cpus-per-task=2                       #Amount of CPUs (logical)                                        $
#SBATCH --output=/nesi/project/vuw03073/testScripts/slurmOut/all_settlement_test_%a.%j.txt    #Name of the output file                                 $
#SBATCH --mail-type=ALL                         #This will send you an email when the STARTS and ENDS		 $
#SBATCH --mail-user=calquigs@gmail.com          #Enter your email address.                                       $
#SBATCH --profile=task
#SBATCH --array=0-11                     # Array jobs
#SBATCH --export NONE

export SLURM_EXPORT_ENV=ALL
export HDF5_USE_FILE_LOCKING=FALSE

#purging any modules/software loaded in the background prior to submitting script
module purge

#Load the required module for analysis/simulation
module load Miniconda3
source activate opendrift_simon

#Set variables
inPath='/nesi/nobackup/mocean02574/NZB_3/'
outPath='/nesi/nobackup/vuw03073/bigboy/all_settlement/'

#14pops
#names=('OPO' 'MAU' 'CAP' 'WEST' 'FLE' 'TAS' 'CAM' 'LWR' 'KAI' 'GOB' 'TIM' 'FIO' 'HSB' 'BGB')
#lons=(173.2 176.0 176.3 172.4 172.7 173.1 174.3 171.9 173.7 173.3 171.3 166.8 168.2 168.2)
#lats=(-35.5 -37.4 -40.9 -40.5 -40.6 -41.1 -41.7 -41.3 -42.4 -42.9 -44.4 -45.1 -46.8 -46.9)

#19pops
#names=('HOU' 'OPO' 'PAK' 'TEK' 'MAU' 'CAP' 'KAT' 'POG' 'GOL' 'TAS' 'LWR' 'NMC' 'GOB' 'JAB' 'TIM' 'FIO' 'RIV' 'HSB' 'BGB')
#lons=(173.1 173.2 174.8 175.3 176.0 176.3 173.7 174.2 172.8 173.1 171.9 171.1 173.3 168.6 171.3 166.8 167.5 168.2 168.2)
#lats=(-34.6 -35.5 -36.1 -36.7 -37.4 -40.9 -40.7 -40.9 -40.7 -41.1 -41.3 -42.3 -42.9 -43.8 -44.4 -45.1 -46.2 -46.8 -46.9)

#22pops
#names=('HOU' 'OPO' 'PAK' 'TEK' 'MAU' 'CAP' 'KAT' 'POG' 'GOL' 'TAS' 'LWR' 'NMC' 'GOB' 'JAB' 'TIM' 'FIO' 'RIV' 'HSB' 'BGB' 'HIC' 'RUA' 'DAB')
#lons=(173.1 173.2 174.8 175.3 176.0 176.3 173.7 174.2 172.9 173.1 171.9 171.1 173.3 168.6 171.3 166.8 167.6 168.2 168.2 178.3 174.6 174.8)
#lats=(-34.6 -35.5 -36.1 -36.7 -37.4 -40.9 -40.7 -40.9 -40.7 -41.1 -41.3 -42.3 -42.9 -43.8 -44.4 -45.1 -46.3 -46.8 -46.9 -37.4 -38.0 -41.4)

#testpops
names=('FIO' 'BGB' 'HSB' 'TIM')
lons=(166.8 168.2 168.2 171.3)
lats=(-45.1 -46.9 -46.8 -44.4)

#names=('BGB')
#lons=(168.2)
#lats=(-46.9)

#Create array of yyyymm
#months=(01 02 03 04 05 06 07 08 09 10 11 12)
months=(08)
#years=($(seq 1994 2016))
years=($(seq 1994 1996))
#years=(1994)
declare -a ym

for i in "${months[@]}"
do
    for j in "${years[@]}"
    do
         ym+=("$j""$i")
    done
done

num_runs_per_site=$((${#months[@]}*${#years[@]}))
name=${names[$(($SLURM_ARRAY_TASK_ID/$num_runs_per_site))]}
lon=${lons[$(($SLURM_ARRAY_TASK_ID/$num_runs_per_site))]}
lat=${lats[$(($SLURM_ARRAY_TASK_ID/$num_runs_per_site))]}

echo $num_runs_per_site
echo $name
echo ${ym[$(($SLURM_ARRAY_TASK_ID%$num_runs_per_site))]}

#run jobs
python /nesi/project/vuw03073/opendriftxmoana/scripts/moana_master.py -i $inPath -o $outPath -n $name -ym ${ym[$(($SLURM_ARRAY_TASK_ID%$num_runs_per_site))]} -lon $lon -lat $lat


