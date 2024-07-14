#!/bin/bash
#BSUB -n 3 -R "rusage[mem=3]" -R "hosts[span=1]" -W 358
#<usage>
[[ $# -gt 0 ]] || {
    echo "Description:"
    echo "This script to runs the VARBIN algorithm on 5k, 20k, and 50k bins in the"
    echo " a human reference genome.  It calculates ploidy within the specified range"
    echo " using LOWESS least square means (ginkgo single-cell method)."
    echo ""
    echo "Usage:"    
    echo "This script expects a reference name as the first argument, a bam file as the"
    echo " second argument; a file extension as the third, and the ploidy range as "
    echo " \`min' \`max' on the fourth and fifth arguments, respectively."
    echo ""
    echo "Example:"
    echo "bsub -n 3 -M 3 -W 89 ./src/02_varbin_pe.sh hsa38 path/to/file.dd.bam .bam 1.5 4.8"
    echo ""
    exit 1;
}
#</usage>

## suggestions from:
## https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -e -x -o pipefail -u

## location of cna utis is located in grch37
#cna_utils=$HOME/genomes/homo_sapiens/Ensembl/GRCh37.p13/Sequence/cna_utils/
cna_utils=/work/singer/opt/cna_utils/

GENOME=$1

if [ "$GENOME" = "hsa37" ]; then
    CNA_UTIL_GENOME=hg19
    REF_NAME=grch37
elif [ "$GENOME" = "hsa38" ]; then
    CNA_UTIL_GENOME=hg38
    REF_NAME=grch38
fi

BAM=$2

EXTENSION=$3
[[ ! -z "$EXTENSION" ]] || EXTENSION=.bam

MID=$( basename $BAM $EXTENSION )
echo $MID

## ploidy multipliers, min and max
MIN=$4
MAX=$5

GROUP=$(echo $( basename $(dirname $BAM) | sed -E 's/[PRMB]$//'))

module load R/R-4.0.5

for i in 5 20 50 100 120 200 ; do
    ## output dir
    OUT=varbin${i}k/${GROUP}
    [ -d $OUT ] || mkdir -p $OUT
    ## runing bins
    echo ${i}k bin count
    ## bin count    
    $cna_utils/scripts/getBinCounts.py \
	-i $BAM \
	-b $cna_utils/data/${CNA_UTIL_GENOME}_${i}k_gz_enc_bins.bed \
	-d $cna_utils/data/${CNA_UTIL_GENOME}_150bp_dz_enc.bed \
	-o ${OUT}/${MID}_${REF_NAME}.${i}k.bwa.bin.counts.bed \
	-v > ${OUT}/${MID}_${REF_NAME}.${i}k.bwa.bin.counts.stats.bed

    ## copy number estiamtion
    echo ${i}k varbin cn estimation
    ## copy number estimation with varbin
    $cna_utils/scripts/cnvProfile.R \
	-b ${OUT}/${MID}_${REF_NAME}.${i}k.bwa.bin.counts.bed \
	-g $cna_utils/data/${CNA_UTIL_GENOME}_${i}k_gz_enc_gc.bed \
	-n ${OUT}/${MID}_${REF_NAME}.${i}k.bwa \
	--minploidy=${MIN} --maxploidy=${MAX} \
	-v 2> ${OUT}/${MID}_${REF_NAME}.${i}k.quantal.log

    echo "cellID ploidy error" | tr ' ' '\t' >  ${OUT}/${MID}_${REF_NAME}.${i}k.quantal.ploidy.txt
    echo $MID \
	 $(grep "Ploidy" ${OUT}/${MID}_${REF_NAME}.${i}k.quantal.log  | tail -n 1 | sed -e 's/.*Ploidy: //') \
	 $(grep "Error" ${OUT}/${MID}_${REF_NAME}.${i}k.quantal.log  | tail -n 1 | sed -e 's/.*Error: //') | \
	tr " " "\t" >> ${OUT}/${MID}_${REF_NAME}.${i}k.quantal.ploidy.txt
done


## __EOF__

#% for i in 20 5 ; do 
#%     OUT=varbin${i}k/${GROUP}
#%     [ -d $OUT ] || mkdir -p $OUT
#%     ## runing bins
#%     echo ${i}k bin count
#% 
#%     $cna_utils/scripts/cnvProfile.R \
#% 	-b ${OUT}/${MID}_${REF_NAME}.${i}k.bwa.bin.counts.bed \
#% 	-g $cna_utils/data/${CNA_UTIL_GENOME}_${i}k_gz_enc_gc.bed \
#% 	-e $cna_utils/data/${CNA_UTIL_GENOME}_${i}k_gz_enc_badbins_gc.bed \
#% 	-n ${OUT}/${MID}_${REF_NAME}.${i}k.bwa.nobad \
#% 	--minploidy=${MIN} --maxploidy=${MAX} \
#% 	-v 2> ${OUT}/${MID}_${REF_NAME}.${i}k.nobad.quantal.log
#% done
#% 


