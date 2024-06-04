#!/usr/bin/env bash
# shellcheck disable=SC1091
# shellcheck disable=SC2153
# This script is used to run the job on the Gekko cluster.

#PBS -N hint-atac
#PBS -P sbs_liyh

#PBS -q q128
#PBS -l select=1:ncpus=16:mpiprocs=16:mem=16gb
#PBS -l walltime=4:00:00

#PBS -m bea
#PBS -M suffi.azizan@ntu.edu.sg
#PBS -j oe

# load module files
module load singularity

# load conda environment
eval "$(conda shell.bash hook)"
conda activate rgt-suite

out_prefix=$OUT_PREF
bam=$FILE
peaks=$MERGED_PEAKS
path_output=$OUT_PATH
experiment=$EXP_TYPE

echo "Running rgt-hint..."

# run rgt-hint based on the experiment type
if [[ "${experiment}" == "atacseq" ]]; then
    if rgt-hint footprinting --atac-seq --paired-end --organism=hg38 --output-prefix="${out_prefix}" --output-location="${path_output}" "${bam}" "${peaks}"; then
        echo "rgt-hint on ATAC-seq mode completed successfully."
    else
        echo "rgt-hint failed."
        exit 1
    fi
elif [[ "${experiment}" == "dnaseseq" ]]; then
    if rgt-hint footprinting --dnase-seq --bias-correction --organism=hg38 --output-prefix="${out_prefix}" --output-location="${path_output}" "${bam}" "${peaks}"; then
        echo "rgt-hint on DNAse-seq mode completed successfully."
    else
        echo "rgt-hint failed."
        exit 1
    fi
fi

echo "Running bedtools [getfasta] to get the sequences of the footprints for STREME de novo motif discovery..."

# run bedtools getfasta
if cut -f 1,2,3 "${path_output}"/"${out_prefix}".bed | bedtools getfasta -fi /home/suffi.azizan/scratchspace/inputs/genomes/GRCh38_no_alt_GCA_000001405.15.fasta -bed - -fo /home/suffi.azizan/scratchspace/outputs/hint-atac_streme-io/streme/inputs/"${out_prefix}"_footprints.fa; then
    echo "bedtools getfasta completed successfully."
    echo "running streme..."
    # run streme
    if singularity exec --bind /scratch/suffi.azizan:/mnt /home/suffi.azizan/installs/singularity_sifs/memesuite.sif streme --verbosity 1 --oc /mnt/outputs/hint-atac_streme-io/streme/outputs/"${out_prefix}" --dna --totallength 4000000 --minw 8 --maxw 15 --thresh 0.05 --align center --p /mnt/outputs/hint-atac_streme-io/streme/inputs/"${out_prefix}"_footprints.fa 2> /dev/null; then
        echo "streme completed successfully."
        echo "Workflow script completed with success."
    else
        echo "streme failed. Aborting..."
        exit 1
    fi
else
    echo "bedtools getfasta failed. Aborting..."
    exit 1
fi