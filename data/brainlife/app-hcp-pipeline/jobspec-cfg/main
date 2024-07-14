#!/bin/bash
#PBS -l nodes=1:ppn=8,vmem=16gb,walltime=15:00:00
#PBS -N HCP-pipeline
#PBS -V

set -e
set -x

export FREESURFER_LICENSE="hayashis@iu.edu 29511 *CPmh9xvKQKHE FSg0ijTusqaQc"
echo $FREESURFER_LICENSE > license.txt

mkdir -p output

#convert input into bids
bl2bids

stage=`jq -r '.stage' config.json`
inT1w=`jq -r '.t1' config.json`
inT2w=`jq -r '.t2' config.json`
inFSDIR=`jq -r '.fsin' config.json`

skipbidsvalidation=""
[ "$(jq -r .skipbidsvalidation config.json)" == "true" ] && skipbidsvalidation="--skip-bids-validation"

processing_mode=`jq -r '.processing_mode' config.json`

#####################################################################################
#####################################################################################
# some logical checks
if [[ $inT1w = "null" ]] || [[ $inT2w = "null" ]] ; then
	echo "app needs minimally a T1w and a T2w. exiting"
	exit 1
fi

# avoid templateflow problems on HPC's running singularity
mkdir -p templateflow
export SINGULARITYENV_TEMPLATEFLOW_HOME=$PWD/templateflow

# set FreeSurfer
[ -z "$FREESURFER_LICENSE" ] && echo "Please set FREESURFER_LICENSE in .bashrc" && exit 1;

sub=$(jq -r "._inputs[0].meta.subject" config.json | tr -d "_")
if [[ -z $sub || $sub == null ]]; then
    sub=0
fi

# if freesurfer provided, copy it to the same level as output dir
# TODO why can't we just symlink this in?
if [[ $inFSDIR != "null" ]] ; then

    #clean up previous freesurfer dir if it exists
    rm -rf output/sub-$sub/T1w
    mkdir -p output/sub-$sub/T1w
    
    cp -rH $inFSDIR output/sub-$sub/T1w/sub-$sub
    chmod -R +rw output/sub-$sub/T1w/sub-$sub
    stage="PreFreeSurfer PostFreeSurfer"
fi

echo "Running stages: $stage"

singularity exec -e \
    -B `pwd`/bids:/bids \
    -B `pwd`/output:/output \
    docker://bids/hcppipelines:v4.3.0-3 \
    ./run.py /bids /output participant \
    --n_cpus 8 \
    --stages $stage \
    --license_key "$FREESURFER_LICENSE" \
    --participant_label $sub \
    --processing_mode $processing_mode \
    $skipbidsvalidation
