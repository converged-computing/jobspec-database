#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=workq
#SBATCH --account=mwaeor
#SBATCH --export=MWA_DIR
#SBATCH --array=0-2
#SBATCH -M garrawarla
#SBATCH -J rts_setup

module use /pawsey/mwa/software/python3/modulefiles
module load python-singularity
module load mongoose
module load hyperdrive

#CURRENTDATE used to name run-directory
CURRENTDATE=`date +%Y_%m_%d`

#Read in obsID text file and select observation to process
obsarr=()
while IFS= read -r line || [[ "$line" ]]; do
        obsarr+=("$line")
done < LoBES13_093_corr.txt 
OBSID=${obsarr[${SLURM_ARRAY_TASK_ID}]}

##Creates MWA_DIR/data/${OBSID}/${CURRENTDATE} directory; copies run_rts.sh to this directory
python edit_rts.py ${CURRENTDATE} ${OBSID}

#Setup paths to where we will run the RTS and where the data (gpuboxes) is located
obspath="/astro/mwaeor/MWA/data/$OBSID/${CURRENTDATE}"
obspath2="/astro/mwaeor/MWA/data/$OBSID"

#Unzips the flag files
cd $obspath2
unzip -n "${OBSID}_flags.zip"

cd $obspath

# Copy metafits to run directory. Will find one and only one metafits file in the parent directory.
METAFITS=$(find .. -maxdepth 1 -name "*.metafits" -print -quit)
[ $META ] && echo "No metafits file in parent directory!" && exit 1
cp ${METAFITS} .
METAFITS="${PWD}/$(basename "${METAFITS}")"
echo "Using ${METAFITS}"

set -eux


# Ignore the "warning" that an observation shouldn't be used when all its
# delays are 32.
overwrite-metafits-delays "${METAFITS}"

#Use hyperdrive srclist by-beam to generate the source list to be used in the patch step
srclist by-beam -i rts -m ${OBSID}.metafits -n 300 -o rts --collapse-into-single-source --source-dist-cutoff 50 /astro/mwaeor/clynch/catalog/srclist_pumav3_EoR0aegean_fixedEoR1pietro+ForA_phase1+2_GSCRB_GLMGP.txt srclist_pumav3_EoR0aegean_fixedEoR1pietro+ForA_phase1+2_GSCRB_GLMGP_${OBSID}_patch300.txt


#Use hyperdrive srclist by-beam to generate the source list to be used to peel sources; comment out if not peeling
srclist by-beam -i rts -m ${OBSID}.metafits -n 10 -o rts --source-dist-cutoff 50 /astro/mwaeor/clynch/catalog/srclist_pumav3_EoR0aegean_fixedEoR1pietro+ForA_phase1+2_GSCRB_GLMGP.txt srclist_pumav3_EoR0aegean_fixedEoR1pietro+ForA_phase1+2_GSCRB_GLMGP_${OBSID}_peel10.txt

#Rename the flag files so that they will be read by the RTS
cd $obspath2
time reflag-mwaf-files

cd $obspath

# Generate the RTS patch .in file
rts-in-file-generator patch \
                      --base-dir .. \
                      --metafits "${METAFITS}" \
                      --srclist srclist_pumav3_EoR0aegean_fixedEoR1pietro+ForA_phase1+2_GSCRB_GLMGP_${OBSID}_patch300.txt \
					  --use-fee-beam \
                      --subband-ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
                      -o rts_patch.in

# Generate the RTS peel .in file; comment out if not peeling sources
rts-in-file-generator peel \
                      --base-dir .. \
                      --metafits "${METAFITS}" \
                      --srclist srclist_pumav3_EoR0aegean_fixedEoR1pietro+ForA_phase1+2_GSCRB_GLMGP_${OBSID}_peel10.txt \
		      		  --num-primary-cals 0 \ #Number of sources with full DD calibration
					  --use-fee-beam \
		              --num-cals 1 \ #Number of ionospheric calibrators
		              --num-peel 1 \ #Number of sources to remove from the data
		              --subband-ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
                      -o rts_peel.in

