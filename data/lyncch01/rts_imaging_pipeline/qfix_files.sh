#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=workq
#SBATCH --account=mwaeor
#SBATCH --export=MWA_DIR
#SBATCH --array=0-1
#SBATCH -J edit_fixfile


#Date the calibration occured; leave as is if the same day as running this script, otherwise update
CURRENTDATE=`date +%Y_%m_%d`

#Read in obsID text file and select observation to process
obsarr=()
while IFS= read -r line || [[ "$line" ]]; do
        obsarr+=("$line")
done < LoBES13_093_corr.txt 
OBSID=${obsarr[${SLURM_ARRAY_TASK_ID}]}

#Make directory to hold calibrated data
mkdir ${OBSID}

cd "${OBSID}/"

#copy calibrated data and metafits to this new directory
DATPTH="/astro/mwaeor/MWA/data/${OBSID}/${CURRENTDATE}/uvdump*uvfits"

cp ${DATPTH} .

METPTH="/astro/mwaeor/MWA/data/${OBSID}/${CURRENTDATE}/${OBSID}.metafits"

cp ${METPTH} .

#Corrects the central frequency in the uvfits
module load python-singularity
time python ../correct_rtsuvfits.py 

#Creates measurement set for each uvfits file
module load casa/5.6.1-8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pawsey/intel/19.0.5/mkl/lib/intel64/
time casa --nologger -c ../rts_uv2ms.py 

#Corrects the antenna table so WSCLEAN won't complain
module load cotter
time python ../fixMS.py $OBSID 
















