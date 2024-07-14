#!/bin/bash
## run as bsub -Is -n 1 -M 4 -W 16:00 ./runGistic.sh
#BSUB -n 1 -M 12 -W 359
#<usage>
[[ $# -gt 0 ]] || {
    echo "Description:"
    echo "This script to runs GISTIC for 20kb with parameters optimized for single-cell log-ratio cutoffs"
    echo ""
    echo "Usage:"    
    echo "bsub -n 1 -M 12 -W 359 ./src/05_runGISTIC_SC.sh <hsa37|hsa38> <confidence> path/to/file.cbs.seg <file.extension> <5k|20k|50k bin_level> <out_dir>"
    echo "bsub -n 1 -M 12 -W 359 ./src/05_runGISTIC_SC.sh hsa38 80 path/to/file.cbs.seg .cbs.seg 20k gistic_out/"
    echo ""
    exit 1;
}
#</usage>

set -e -x -o pipefail -u

GENOME=$1
CONF=$2
INFILE=$3
EXTENSION=$4
BIN_LEVEL=$5
OUT_DIR=$6

## GISTIC reference genome to use
GISTIC_PATH=/juno/work/singer/opt/gistic2/
if [ $GENOME == "hsa37" ]
then
    HGMAT=${GISTIC_PATH}/support/refgenefiles/hg19.UCSC.add_miR.140312.refgene.mat
    ## bin level to use
    if [ $BIN_LEVEL == "5k" ]
    then
	VBINS=res/gistic/grch37.bin.boundaries.5k.bowtie.k50.markerFile.txt
    elif [ $BIN_LEVEL == "20k" ]
    then
	VBINS=res/gistic/grch37.bin.boundaries.20k.bowtie.k50.markerFile.txt
    elif [ $BIN_LEVEL == "50k" ]
    then
	VBINS=res/gistic/grch37.bin.boundaries.50k.bowtie.k50.markerFile.txt
    fi
    
elif [ $GENOME == "hsa38" ]
then
    HGMAT=${GISTIC_PATH}/support/refgenefiles/hg38.UCSC.add_miR.160920.refgene.mat
    ## bin level to use
    if [ $BIN_LEVEL == "5k" ]
    then
	VBINS=res/gistic/grch38.bin.boundaries.5k.bwa.markerFile.txt
    elif [ $BIN_LEVEL == "20k" ]
    then
	VBINS=res/gistic/grch38.bin.boundaries.20k.bwa.markerFile.txt
    elif [ $BIN_LEVEL == "50k" ]
    then
	VBINS=res/gistic/grch38.bin.boundaries.50k.bwa.markerFile.txt
    fi
fi

## out stem
OUT_STEM=$( basename $INFILE $EXTENSION )   
BD=${OUT_DIR}/out_SC__${OUT_STEM}_${GENOME}_${BIN_LEVEL}_c${CONF}  ## 

[[ -d $BD ]] || mkdir -p $BD

## Set thresholds for copy number to AMP = log2(2.5/2), and DEL = log2(1.2/2)
~/bin/gistic2 -refgene ${HGMAT} -b ${BD} -seg ${INFILE} -mk $VBINS \
	      -ta 0.321 -td 0.736 -conf 0.${CONF} -broad 1 -twoside 1 \
	      -res 0.03 -genegistic 1 -rx 0 -js 8 -cap 3.3219\
	      -savegene 1 -gcm median -v 10 -armpeel 1


## __END__
## Usage: gp_gistic2_from_seg -b base_dir -seg segmentation_file
## -refgene ref_gene_file  [-mk markers_file] [-alf array_list_file(def:empty)]
## [-cnv cnv_file] [-ta amplifications_threshold(def=.1)] [-td deletions_threshold(def=.1)]
## [-js join_segment_size(def=8)] [-ext extension] [-qvt qv_thresh(def=0.25)]
## [-rx remove_x(def=1)] [-v verbosity_level(def=0)] [-cap cap_val(def=1.5]]
## [-broad run_broad_analysis(def=0)] [-brlen broad_length_cutoff(def=0.98)]
## [-maxseg max_sample_segs(def=2500)] [-res res(def=0.05)] [-conf conf_level(def=0.75)]
## [-genegistic do_gene_gistic(def=0)] [-smalldisk save_disk_space(def=0)]
## [-smallmem use_segarray(def=1)] [-savegene write_gene_files(def=0)]
## [-arb do_arbitration(def=1)] [-twosides use_two_sided(def=0)] [-peaktype peak_types(def=robust)]
## [-saveseg save_seg_data(def=1)] [-savedata write_data_files(def=1)]
## [-armpeel armpeel(def=1)] [-gcm gene_collapse_method(def=mean)]
## [-scent sample_center(def=median)] [-maxspace max_marker_spacing]
## [-logdat islog(def=auto-detect)]
