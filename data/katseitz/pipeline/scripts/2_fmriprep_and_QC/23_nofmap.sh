#!/usr/bin/bash

#SBATCH -A p31833
#SBATCH -p normal
#SBATCH -t 48:00:00
#SBATCH --mem=35G
#SBATCH -J fno_fmap

module purge
module load singularity/latest
echo "modules loaded" 
echo "beginning preprocessing"



#singularity run --cleanenv \
#-B /gpfs/projects/VanSnellenbergGroup/Queensland_Unprocessed_594Unrelated:/data \
#-B /gpfs/projects/VanSnellenbergGroup/Queensland_fMRIPrep_Outputs_Full/derivatives/fmriprep:/out \
#-B /gpfs/projects/VanSnellenbergGroup/Queensland_fMRIPrep_Outputs_Full/derivatives/freesurfer:/fsdir \
#-B /gpfs/projects/VanSnellenbergGroup/Queensland_fMRIPrep_Outputs_Full/derivatives/scratch:/scratch \
#-B /gpfs/projects/VanSnellenbergGroup/fMRIprep/license.txt:/opt/freesurfer/license.txt \
#-B /gpfs/projects/VanSnellenbergGroup/Queensland_fMRIPrep:/home/fmriprep \
#docker://nipreps/fmriprep:22.0.1 /data /out participant -w /scratch \
#--write-graph --participant-label 1042 \
#--output-spaces T1w MNI152NLin2009cAsym fsLR fsaverage --cifti-output 91k \
#--omp-nthreads 20 --nprocs 24 --mem_mb 120000 --use-syn-sdc \
#--medial-surface-nan --stop-on-first-crash --dummy-scans 5 -vv

#WITHOUT FMAP
singularity run --cleanenv -B /projects/b1108:/projects/b1108 \
-B /projects/b1108/studies/transitions2/data/processed/neuroimaging/fmriprep_ses-1:/out \
-B /projects/b1108/studies/transitions2/data/raw/neuroimaging/bids:/data \
-B /projects/b1108/studies/transitions2/data/processed/neuroimaging/fmriprep_ses-1/work:/work \
/projects/b1108/software/singularity_images/fmriprep-23.0.1.simg \
/data /out participant \
--participant-label ${1} --bids-filter-file bids_filter_file_ses-1.json \
--fs-license-file /projects/b1108/software/freesurfer_license/license.txt \
-w /work --ignore fieldmaps --skip_bids_validation
