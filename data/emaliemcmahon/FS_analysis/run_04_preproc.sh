#!/bin/bash -l
#SBATCH
#SBATCH --job-name=preproc
#SBATCH --time=10:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --mail-type=end

sid=$1

module load matlab
module load freesurfer
source ~/work/mcmahoneg/mri_data_anlys/studies/cont_actions/analysis/SetUpFreeSurfer.sh

preproc-sess \
	-s ${sid} \
	-df sessdir \
	-per-run \
	-fsd bold \
	-fwhm 0 \
	-force 

mv *.out ./slurm_out/
