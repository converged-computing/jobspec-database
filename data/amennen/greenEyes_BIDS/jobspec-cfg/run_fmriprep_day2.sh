#!/bin/bash

data_dir=/jukebox/norman/amennen/RT_prettymouth/data #this is my study directory
bids_dir=$data_dir/bids/Norman/Mennen/5516_greenEyes #this is where BIDS formatted data will end up and should match the program card on the scanner


singularity run --cleanenv \
    --bind $bids_dir:/home \
    /jukebox/hasson/singularity/fmriprep/fmriprep-v1.2.3.sqsh \
    --participant-label sub-$1 \
    --session-id $2 \
    --fs-license-file /home/derivatives/license.txt \
    --no-submm-recon \
    --bold2t1w-dof 6 --nthreads 8 --omp-nthreads 8 \
    --output-space T1w template fsaverage6 \
    --template MNI152NLin2009cAsym \
    --ignore slicetiming \
    --write-graph --work-dir /home/derivatives/work \
    /home /home/derivatives participant

 # many usage options
 # SEE HERE: https://fmriprep.readthedocs.io/en/stable/usage.html

 # To only run for a specific task, add -t flag. For example: 
 #  -t study \
 
 # If you have more than 2 T1w images, you may want to run with longitudinal flag: 
 # --longitudinal \
