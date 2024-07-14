#! /bin/bash

module load fsl/6.0.2
module load pydeface/2.0.0

# hard code defaced dir to send all files - do this where I'll be sharing
sharing_dir=/jukebox/norman/amennen/RT_GREENEYES_DATASHARE
bids_dir=/jukebox/norman/amennen/RT_prettymouth/data/bids/Norman/Mennen/5516_greenEyes
sid=$1

subj_dir=sub-$sid

T1_original=$bids_dir/$subj_dir/ses-01/anat/${subj_dir}_ses-01_T1w.nii.gz
pydeface $T1_original

T1_defaced=$bids_dir/$subj_dir/ses-01/anat/${subj_dir}_ses-01_T1w_defaced.nii.gz
defaced_dir=$sharing_dir/$subj_dir/ses-01/anat/
# delete anat file already there
rm $defaced_dir/*
mv $T1_defaced $defaced_dir/