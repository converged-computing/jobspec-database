#!/bin/bash 

##Job name
#SBATCH -J LMM
#SBATCH --mem-per-cpu=20G
#SBATCH --time=00-12:01:00
#SBATCH --cpus-per-task=2
#SBATCH --account=p23_lcbc
#SBATCH --output logs/slurm-%j.txt


echo "LOADING MATLAB MODULE"
module load matlab/R2018a
echo `which matlab`
echo "SOURCING FSL"
FSLDIR=/cluster/projects/p23/tools/mri/fsl/current
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
echo "SOURCING FREESURFER"
export FREESURFER_HOME=/cluster/projects/p23/tools/mri/freesurfer/freesurfer.6.0.0
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=${1}/subjects


#dirs
wdir=${1} 
adir=${2}
sdir=${3}
odir=${4}

#files
Y=${5}
Qdec=${6}
subj=${7}
surf=${8}
mask=${9}
cohort=${10}
f=${11}


#logfile
#mv logs/slurm-${SLURM_JOBID}.txt logs/slurm.LMM.${10}.${11}.${4}.${SLURM_JOBID}.log


###########
# INPUTS
###########
echo -e "
1 = $(echo $wdir)
2 = $(echo $adir)
3 = $(echo $sdir)
4 = $(echo $odir) 
5 = $(echo $Y)
6 = $(echo $Qdec)
7 = $(echo $subj)
8 = $(echo $surf)
9 = $(echo $mask)
10 = $(echo $cohort)
11 = $(echo $f)"
export HOME=$odir
matlab -nodisplay -r "PopAsym_LMM('$1', '$2', '$3', '$4', '$5', '$6', '$7', '$8', '$9', '$cohort', '$f')"

