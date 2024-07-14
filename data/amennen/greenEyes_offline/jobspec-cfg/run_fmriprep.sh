#! /bin/bash

# Run using something like:
# ./run_fmriprep.sh |& tee ../derivatives/logs/run_fmriprep.txt 

#bids_dir=/jukebox/hasson/snastase/narratives/prettymouth/
bids_dir=/jukebox/norman/amennen/prettymouth_fmriprep2/

singularity run --cleanenv \
    --bind $bids_dir:/home \
    /jukebox/hasson/singularity/fmriprep/fmriprep-v1.2.3.sqsh \
    --participant-label sub-$1 \
    --fs-license-file /home/code/fs-license.txt \
    --bold2t1w-dof 6 --nthreads 8 --omp-nthreads 8 \
    --output-space fsaverage6 template \
    --template MNI152NLin2009cAsym \
    --ignore slicetiming \
    --write-graph --work-dir /home/derivatives/work \
    /home /home/derivatives participant
