#! /bin/bash

# Script for running the obiwan code within a Shifter container at NERSC
#only need to be set while running cosmos
export LEGACY_SURVEY_DIR=/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_data/cosmos/${cosmos_section}
#export LEGACY_SURVEY_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr9
export DUST_DIR=/global/cfs/projectdirs/cosmo/data/dust/v0_1
export UNWISE_COADDS_DIR=/global/cfs/projectdirs/cosmo/work/wise/outputs/merge/neo6/fulldepth:/global/cfs/projectdirs/cosmo/data/unwise/allwise/unwise-coadds/fulldepth
export GAIA_CAT_DIR=/global/cfs/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom-2
export GAIA_CAT_VER=2
export UNWISE_MODEL_SKY_DIR=/global/cfs/cdirs/cosmo/work/wise/unwise_catalog/dr3/mod
export TYCHO2_KD_DIR=/global/cfs/projectdirs/cosmo/staging/tycho2
export SKY_TEMPLATE_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/calib/sky_pattern
export PS1CAT_DIR=/global/cfs/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3
export LARGEGALAXIES_CAT=/global/cfs/cdirs/cosmo/staging/largegalaxies/v3.0/SGA-ellipse-v3.0.kd.fits
BLOB_MASK_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr8/south


export PYTHONPATH=./galsim_modules:/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_code/obiwan_dr9m/dev-9.6.6/py-9.6.6:/usr/local/lib/python:/usr/local/lib/python3.6/dist-packages:/src/unwise_psf/py:.
#export PYTHONPATH=/usr/local/lib/python:/usr/local/lib/python3.6/dist-packages:/src/unwise_psf/py:$PYTHONPATH
# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled



brick=$1
export BRICKNAME=${brick}

export RANDOMS_FROM_FITS=$CSCRATCH/Obiwan/dr9_LRG/obiwan_out/$name_for_run/divided_randoms/brick_${brick}.fits

maxmem=134217728
let usemem=${maxmem}*${threads}/32

bri=$(echo $brick | head -c 3)

log="${outdir}/logs/${bri}/log.$brick"

mkdir -p $(dirname $log)
echo logging to...${log}

echo -e "\n\n\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log
echo "PWD: $(pwd)" >> $log
echo >> $log
ulimit -a >> $log
echo >> $log
#tmplog="/tmp/$brick.log"

echo -e "\nStarting on $(hostname)\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log

echo Running on $(hostname)

python runbrick_sim.py \
--dataset ${dataset} \
--brick $brick \
--nobj ${nobj} --startid ${rowstart} \
--outdir $outdir \
--threads $threads \
--random_fn $RANDOMS_FROM_FITS \
--pickle "${outdir}/pickles/${bri}/${brick}/runbrick-%(brick)s-%%(stage)s.pickle" \
--checkpoint ${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle \
--stage writecat \
--run decam \
--add_sim_noise \
>> $log 2>&1 
#--write-stage coadds \

#--blob-mask-dir ${BLOB_MASK_DIR} \
status=$?
#cat $tmplog >> $log
#python legacypipe/rmckpt.py --brick $brick --outdir $outdir

