#PBS -l nodes=1:ppn=1
#PBS -l mem=100gb
#PBS -l walltime=80:00:00
#PBS -e ref_genome/biorep_merged/results/loops/logs/
#PBS -o ref_genome/biorep_merged/results/loops/logs/
#PBS -N run_fithichip_loopcalling_L10_t2t
#PBS -V

# example run:
# 1) qsub -t <index1>,<index2>,... workflow/scripts/run_fithichip_loopcalling_L10.sh
# 2) qsub -t <index-range> workflow/scripts/run_fithichip_loopcalling_L10.sh
# 3) qsub -t <index1>,<index2>,<index-range1> workflow/scripts/run_fithichip_loopcalling_L10.sh
# 4) qsub -t <index-range1>,<index-range2>,... workflow/scripts/run_fithichip_loopcalling_L10.sh
# 5) qsub -t <any combination of index + ranges> workflow/scripts/run_fithichip_loopcalling_L10.sh

# print start time message
start_time=$(date "+%Y.%m.%d.%H.%M")
echo "Start time: $start_time"

# print start message
echo "Started: run_fithichip_loopcalling_L10_t2t"

# run bash in strict mode
set -euo pipefail
IFS=$'\n\t'

# make sure to work starting from the base directory for this project 
cd $PBS_O_WORKDIR

# source tool paths
source workflow/scripts/loops/fithichip_source_paths.sh

# extract the sample information using the PBS ARRAYID
samplesheet="results/samplesheets/post-hicpro/human.biorep_merged.samplesheet.without_header.tsv"
sample_info=( $(cat $samplesheet | sed -n "${PBS_ARRAYID}p") )
sample_name="${sample_info[0]}"
org="${sample_info[1]}"

# printing sample information
echo
echo "Processing"
echo "----------"
echo "sample_name: $sample_name"
echo "org: $org"
echo

# identify hicpro validpairs file if avaliable
file_samplesheet="results/samplesheets/post-hicpro/human_t2t_biorep_merged.peaks_files.samplesheet.without_header.tsv"
unset IFS
sample_info=( $(grep "${sample_name}" ${file_samplesheet}) )

echo "# finding hicpro validpairs file"
if [ -f "${sample_info[1]}" ]; then
    echo "validpairs file found and will be used to call loops"
    pairs_file=${sample_info[1]}
else 
    echo "validpairs file not found"
    exit 2
fi

# identify peaks file depending on the peak mode selected
# peak mode
# 1 -> HiChIP-Peaks peaks
# 2 -> FitHiChIP peaks
# 3 -> ChIP-Seq peaks
peak_mode=3
echo 
echo "1: HiChIP-Peaks peaks"
echo "2: FitHiChIP peaks"
echo "3: ChIP-Seq peaks"
echo "Selected Peak Mode: $peak_mode"
echo
echo "# finding peaks file"

# HiChIP-Peaks peaks
if [ $peak_mode -eq 1 ]; then
    if [ -f "${sample_info[2]}" ]; then
        echo "hichip-peaks peaks found and will be used to call loops"
        peaks_file=${sample_info[2]}

        # make the output directory
        outdir_L10="${PBS_O_WORKDIR}/ref_genome/biorep_merged/results/loops/fithichip/${sample_name}_hichip-peaks.peaks/L10/"
        mkdir -p $outdir_L10
    else
        echo "no valid hichip-peaks peaks file found"
        exit 2
    fi
fi

# FitHiChIP peaks
if [ $peak_mode -eq 2 ]; then
    if [ -f "${sample_info[3]}" ]; then
        echo "fithichip peaks found and will be used to call loops"
        peaks_file=${sample_info[3]}

        # make the output directory
        outdir_L10="${PBS_O_WORKDIR}/ref_genome/biorep_merged/results/loops/fithichip/${sample_name}_fithichip.peaks/L10/"
        mkdir -p $outdir_L10
    else
        echo "no valid fithichip peaks file found"
        exit 2
    fi
fi

# Chip-Seq Peaks
if [ $peak_mode -eq 3 ]; then
    if [ -f "${sample_info[4]}" ]; then
        echo "chip-seq peaks found and will be used to call loops"
        peaks_file=${sample_info[4]}

        # make the output directory
        outdir_L10="${PBS_O_WORKDIR}/ref_genome/biorep_merged/results/loops/fithichip/${sample_name}_chipseq.peaks/L10/"
        mkdir -p $outdir_L10
    else
        echo "no valid chip-seq peaks file found"
        exit 2
    fi
fi

## determine correct chrsize file
if [[ "$org" == "Homo_Sapiens" ]];
then
    ChrSizeFile="/mnt/bioadhoc-temp/Groups/vd-ay/kfetter/hichip-db-loop-calling/ref_genome/chm13_refgenome/chrsize/chm13.chrom.sizes"
    echo "chrsizes: $ChrSizeFile"
elif [[ "$org" == "Mus_Musculus" ]];
then
    ChrSizeFile="/mnt/BioAdHoc/Groups/vd-ay/Database_HiChIP_eQTL_GWAS/Data/RefGenome/chrsize/mm10.chrom.sizes"
    echo "chrsizes: $ChrSizeFile"
else
    echo "org not found"
    exit
fi

####################################################################################################################

# generate config file for L, 5kb
configfile_L10="${outdir_L10}configfile_L10"
touch $configfile_L10
cat <<EOT >> $configfile_L10
# File containing the valid pairs from HiCPro pipeline 
ValidPairs=${pairs_file}

# File containing the bin intervals (according to a specified bin size)
# which is an output of HiC-pro pipeline
# If not provided, this is computed from the parameter 1
Interval=

# File storing the contact matrix (output of HiC-pro pipeline)
# should be accompanied with the parameter 2
# if not specified, computed from the parameter 1
Matrix=

# Pre-computed locus pair file
# of the format: 
# chr1 	start1 	end1 	chr2 	start2 	end2 	contactcounts
Bed=

# Boolean variable indicating if the reference genome is circular
# by default, it is 0
# if the genome is circular, the calculation of genomic distance is slightly different
CircularGenome=0

# File containing reference ChIP-seq / HiChIP peaks (in .bed format)
# mandatory parameter
PeakFile=${peaks_file}

# Output base directory under which all results will be stored
OutDir=${outdir_L10}

#Interaction type - 1: peak to peak 2: peak to non peak 3: peak to all (default) 4: all to all 5: everything from 1 to 4.
IntType=3

# Size of the bins [default = 5000], in bases, for detecting the interactions.
BINSIZE=10000

# Lower distance threshold of interaction between two segments
# (default = 20000 or 20 Kb)
LowDistThr=20000

# Upper distance threshold of interaction between two segments
# (default = 2000000 or 2 Mb)
UppDistThr=2000000

# Applicable only for peak to all output interactions - values: 0 / 1
# if 1, uses only peak to peak loops for background modeling - corresponds to FitHiChIP(S)
# if 0, uses both peak to peak and peak to nonpeak loops for background modeling - corresponds to FitHiChIP(L)
UseP2PBackgrnd=0

# parameter signifying the type of bias vector - values: 1 / 2
# 1: coverage bias regression	2: ICE bias regression
BiasType=1

# following parameter, if 1, means that merge filtering (corresponding to either FitHiChIP(L+M) or FitHiChIP(S+M))
# depending on the background model, would be employed. Otherwise (if 0), no merge filtering is employed. Default: 1
MergeInt=1

# FDR (q-value) threshold for loop significance
QVALUE=0.01

# File containing chromomosome size values corresponding to the reference genome.
ChrSizeFile=${ChrSizeFile}

# prefix string of all the output files (Default = 'FitHiChIP').
PREFIX=FitHiChIP-L10

# Binary variable 1/0: if 1, overwrites any existing output file. otherwise (0), does not overwrite any output file.
OverWrite=1
EOT

#####################################################################################################################

# run fithichip
echo
echo "running fithichip"
echo
$fithichip_call_loops -C $configfile_L10

# print end message
echo
echo "Ended: fithichip loop calling for L10"
echo

# print end time message
end_time=$(date "+%Y.%m.%d.%H.%M")
echo "End time: $end_time"