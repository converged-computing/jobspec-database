#!/bin/bash
#PBS -l nodes=1:ppn=8,walltime=24:00:00,mem=24gb
#PBS -N app-dwi-to-t1-nonlinear

function json_arr_to_bash_list {
    local v=$@
    if [[ $v =~ \[.*\] ]]; then
        v=$(echo $v | tr -d '\[\],')  
    fi
    echo $v
}
for prm in $(jq -r 'keys[]' config.json | grep -Ev ^_); do
    val=$(json_arr_to_bash_list $(jq -r ".$prm" config.json))
    eval "$prm=( $val )"
done

#BIDS
function check_required_fields {
    #these fields are required for dwipreproc
    local json=$1
    local f1=$(jq '.TotalReadoutTime' $json)
    local f2=$(jq '.PhaseEncodingDirection' $json)
    [[ -n $f1 && -n $f2 && $f1 != null && $f2 != null ]]
}

function copy_to_bids {
    # inputs:   (1) bids subdirectory, (2) bids image suffix,
    #           (3) BL variables name, (4) index in BL array
    #           (5) optinal run # suffix

    local sdir=$1
    local itype=$2
    local vname=$3
    local vind=$4
    local suffix=$5
    local msg meta metaval

    if [[ -z $vind ]]; then
        vind=0
    fi
    if [[ -n $suffix ]]; then
        suffix=${suffix}_
    fi

    local bids_dir=input/${bids_sub}/$sdir
    local bids_name=$suffix$itype
    local vvalue=$(eval echo \${$vname[$vind]})
    local outnm=${bids_dir}/${bids_sub}_${bids_name}

    mkdir -p $bids_dir
    ln -sf $(pwd)/$vvalue $outnm.nii.gz
    #make BIDS json sidecar from BL metadata
    jq --arg idprm $vname --argjson idx $vind \
        '[ ._inputs[] | select(.id == $idprm) | .meta ] | .[$idx]' \
        config.json > $outnm.json
    if [[ $sdir == dwi ]]; then
        if ! check_required_fields $outnm.json; then
            msg="TotalReadoutTime and PhaseEncodingDirection BIDS fields reqd "
            msg+="to run mrtrix3 dwipreproc. Missing for ${!vname}"
            echo $msg >&2
            exit 1
        fi
        for meta in bvec bval; do
            #BL names with "s"
            metaval=$(eval echo \${${vname/dwi/${meta}s}[$vind]}) 
            ln -sf $(pwd)/$metaval $outnm.$meta
        done
    fi
    echo $outnm
}

sub=$(jq -r "._inputs[0].meta.subject" config.json | tr -d "_")
if [[ -z $sub || $sub == null ]]; then
    sub=0
fi
bids_sub=sub-$sub

#clean up previous job (just in case)
rm -rf input workdir output

bidst1=$(copy_to_bids anat T1w t1)
bidsdwi=$(copy_to_bids dwi dwi dwi)

if [[ -z $bidst1 || -z $bidsdwi ]]; then
    echo "Missing or invalid input images"
    exit 1
fi

singularity run -e \
    docker://katealpert/dwi2t1-nonlinear:v0 \
    /bin/bash ./dwi2t1.sh $bidst1 $bidsdwi output workdir
res=$?

if [[ $res == 0 ]]; then
    rm -rf input workdir
fi

#exit code from the last command (singularity) will be used.
exit $res
