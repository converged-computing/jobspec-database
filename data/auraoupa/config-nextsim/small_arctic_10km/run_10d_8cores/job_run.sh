#!/bin/bash

#OAR -n nextsim_small_arctic_10km
#OAR -l /nodes=1/core=8,walltime=01:30:00
#OAR --stdout nextsim_small_arctic_10km.out
#OAR --stderr nextsim_small_arctic_10km.err
#OAR --project pr-data-ocean


source env_dahu.src

CMD="mpirun --allow-run-as-root \
        --mca btl_vader_single_copy_mechanism none \
        --mca btl ^openib \
        --mca pml ob1 \
        -np 8 \
        nextsim.exec --config-files=/config_files/bbm_control.cfg"

/usr/local/bin/singularity exec $NEXTSIM_IMAGE_NAME $CMD
