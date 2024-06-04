#!/bin/bash
#PBS -N operm_PI_655972
#PBS -e operm_PI_655972.err
#PBS -o operm_PI_655972.out
#PBS -l select=1:ncpus=6:mem=60gb,walltime=56:00:00
#PBS -q tillers

cd $PBS_O_WORKDIR

BASE_DIR="/zfs/tillers/panicle/lucas/projects/NAM_02-24-2020/separate_calls/data/variants/imputed/missing_filter/QTL_mapping/Beagle_imputed"
POPULATION=PI_655972

# module add singularity

cd ${POPULATION}
singularity run -B /zfs ~/singularity_containers/rqtl2.sif \
    Rscript ${BASE_DIR}/permutation_test.R ${POPULATION}

