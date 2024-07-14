#!/bin/bash

#PBS -l nodes=1:ppn=8
#PBS -l walltime=12:00:00
#PBS -N tracking
#PBS -V

module load mrtrix/0.2.12
module load freesurfer
module load fsl
#module load matlab
module load python

set -x
set -e

dwi=`jq -r '.dwi' config.json`
if [ $dwi != "null" ]; then
    export input_nii_gz=$dwi
    export BVECS=`jq -r '.bvecs' config.json`
    export BVALS=`jq -r '.bvals' config.json`
fi
dtiinit=`jq -r '.dtiinit' config.json`
if [ $dtiinit != "null" ]; then
    export input_nii_gz=$dtiinit/`jq -r '.files.alignedDwRaw' $dtiinit/dt6.json`
    export BVECS=$dtiinit/`jq -r '.files.alignedDwBvecs' $dtiinit/dt6.json`
    export BVALS=$dtiinit/`jq -r '.files.alignedDwBvals' $dtiinit/dt6.json`
fi

NUMFIBERS=`jq -r '.fibers' config.json`
MAXNUMFIBERSATTEMPTED=`jq -r '.fibers_max' config.json`
LMAX=`jq -r '.lmax' config.json`

#TODO - validate other fields?
if [[ $LMAX == "null" || -z $LMAX ]]; then
    echo "lmax is empty... calculating max lmax to use from $BVALS"
	LMAX=`./calculatelmax.py $BVALS`
fi
echo "Using LMAX: $LMAX"

#load bvals/bvecs
bvals=$(cat $BVALS | tr ',' ' ')
bvecs_x=$(cat $BVECS | tr ',' ' ' | head -1)
bvecs_y=$(cat $BVECS | tr ',' ' ' | head -2 | tail -1)
bvecs_z=$(cat $BVECS | tr ',' ' ' | tail -1)

#convert strings to array of numbers
bvecs_x=($bvecs_x)
bvecs_y=($bvecs_y)
bvecs_z=($bvecs_z)

#output grad.b
i=0
true > grad.b
for bval in $bvals; do
    echo ${bvecs_x[$i]} ${bvecs_y[$i]} ${bvecs_z[$i]} $bval >> grad.b
    i=$((i+1))
done

echo "converting dwi input to mif (should take a few minutes)"
[ ! -f dwi.mif ] && time mrconvert $input_nii_gz dwi.mif 

echo "extract b0 images, and compute their average to use as DWI reference"
#module unload mrtrix/0.2.12
#module load mrtrix/3.0
#[ ! -f b0.nii.gz ] && time dwiextract dwi.mif - -bzero | mrmath - mean - -axis 3 | mrconvert -quiet - b0.nii.gz 
#module unload mrtrix/3.0
#module load mrtrix/0.2.12
[ ! -f b0.nii.gz ] && time SINGULARITYENV_PYTHONNOUSERSITE=true singularity exec -e docker://brainlife/dipy:0.14.1 ./b0_extract.py $input_nii_gz b0.nii.gz

echo "create wm.nii.gz"
freesurfer=`jq -r '.freesurfer' config.json`
[ ! -f aparc+aseg.nii.gz ] && time mri_convert ${freesurfer}/mri/aparc+aseg.mgz aparc+aseg.nii.gz
[ ! -f wm_t1.nii.gz ] && time SINGULARITYENV_PYTHONNOUSERSITE=true singularity exec -e docker://brainlife/dipy:0.14.1 ./wm_from_aparc_aseg.py aparc+aseg.nii.gz wm_t1.nii.gz

echo "converting wm.nii.gz from t1- to dwi-space"
[ ! -f wm_dwi.nii.gz ] && time flirt -in wm_t1.nii.gz -ref b0.nii.gz -out wm_dwi.nii.gz -interp nearestneighbour
    
echo "converting wm.nii.gz to wm.mif"
[ ! -f wm.mif ] && time mrconvert -quiet wm_dwi.nii.gz wm.mif

###################################################################################################

echo "make brainmask from dwi data (about 18 minutes)"
[ ! -f brainmask.mif ] && time average -quiet dwi.mif -axis 3 - | threshold -quiet - - | median3D -quiet - - | median3D -quiet - brainmask.mif

###################################################################################################

echo "dwi2tensor"
[ ! -f dt.mif ] && time dwi2tensor -quiet dwi.mif -grad grad.b dt.mif 

echo "tensor2FA"
[ ! -f fa.mif ] && time tensor2FA -quiet dt.mif - | mrmult -quiet - brainmask.mif fa.mif

echo "erode"
[ ! -f sf.mif ] && time erode -quiet wm.mif -npass 3 - | mrmult -quiet fa.mif - - | threshold -quiet - -abs 0.7 sf.mif

echo "estimate response function"
[ ! -f response.txt ] && time estimate_response -quiet dwi.mif sf.mif -grad grad.b response.txt

echo "generating DT_STREAM"
#mrtrix doc says streamtrack/DT_STREAM doesn't need grad.. but without it, it fails
[ ! -f output.DT_STREAM.tck ] && time streamtrack -quiet DT_STREAM dwi.mif output.DT_STREAM.tck -seed wm.mif -mask wm.mif -grad grad.b -number $NUMFIBERS -maxnum $MAXNUMFIBERSATTEMPTED

# SD_PROB and SD_STREAM uses CSD lmax.N.mif (aka FOD?) (should take about 10 minutes to several hours - depending on lmax value) 
echo "computing lmax"
[ ! -f lmax.mif ] && time csdeconv -quiet dwi.mif -grad grad.b response.txt -lmax $LMAX -mask brainmask.mif lmax.mif

echo "generating SD_STREAM"
[ ! -f output.SD_STREAM.tck ] && time streamtrack -quiet SD_STREAM lmax.mif output.SD_STREAM.tck -seed wm.mif -mask wm.mif -grad grad.b -number $NUMFIBERS -maxnum $MAXNUMFIBERSATTEMPTED

echo "generating SD_PROB"
[ ! -f output.SD_PROB.tck ] && time streamtrack -quiet SD_PROB lmax.mif output.SD_PROB.tck -seed wm.mif -mask wm.mif -grad grad.b -number $NUMFIBERS -maxnum $MAXNUMFIBERSATTEMPTED

echo "creating neuro/dwi/recon datatype"
rm -f csd.nii.gz fa.nii.gz dt.nii.gz whitematter.nii.gz brainmask.nii.gz
mrconvert lmax.mif csd.nii.gz
mrconvert fa.mif fa.nii.gz
mrconvert dt.mif dt.nii.gz
mrconvert wm.mif whitematter.nii.gz
mrconvert -datatype Int8 brainmask.mif brainmask.nii.gz

echo "gathering infor for product.json"
track_info output.SD_PROB.tck | ./info2json.py > SD_PROB.json
track_info output.SD_STREAM.tck | ./info2json.py > SD_STREAM.json
track_info output.DT_STREAM.tck | ./info2json.py > DT_STREAM.json
./pull_form.py $input_nii_gz > forms.json

echo "writing product.json"
cat << EOF > product.json
{
    "meta": {
        "qform": $(jq .qform forms.json),
        "sform": $(jq .sform forms.json),
        "TractographyClass": "local",
        "StepSizeUnits": "mm",
        "SeedingMethod": "globalSeedNumber",
        "SeedingNumberMethod": "totalStreamlinesNumber",
        "TerminationCriterion": "leaveMask",
        "TerminationCriterionTest": "binary"
    },

    "dt_stream": {
        "track_info": $(jq . DT_STREAM.json),
        "meta": {
            "TractographyMethod": "deterministic",
            "StepSize": $(jq -r .step_size DT_STREAM.json),
            "AngleCurvature": $(jq -r .min_curv DT_STREAM.json)
        }
    },
    "sd_stream": {
        "track_info": $(jq . SD_STREAM.json),
        "meta": {
            "TractographyMethod": "deterministic",
            "StepSize": $(jq -r .step_size SD_STREAM.json),
            "AngleCurvature": $(jq -r .min_curv SD_STREAM.json)
        }
    },
    "sd_prob": {
        "track_info": $(jq . SD_PROB.json),
        "meta": {
            "TractographyMethod": "probabilistic",
            "StepSize": $(jq -r .step_size SD_PROB.json),
            "AngleCurvature": $(jq -r .min_curv SD_PROB.json)
        }
    },
    "recon": {
        "todo": true
    }
}
EOF

echo "all done"

