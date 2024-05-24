#!/bin/bash
## Job Name
#SBATCH --job-name=20230426-pgen-fastqc-fastp-multiqc-RNAseq
## Allocation Definition
#SBATCH --account=srlab
#SBATCH --partition=srlab
## Resources
## Nodes
#SBATCH --nodes=1
## Walltime (days-hours:minutes:seconds format)
#SBATCH --time=5-00:00:00
## Memory per node
#SBATCH --mem=120G
##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=samwhite@uw.edu
## Specify the working directory for this job
#SBATCH --chdir=/gscratch/scrubbed/samwhite/outputs/20230426-pgen-fastqc-fastp-multiqc-RNAseq

### FastQC and fastp trimming of P.generosa RNA-seq data from 20220323.

https://robertslab.github.io/sams-notebook/2022/03/23/Differential-Gene-Expression-P.generosa-DGE-Between-Tissues-Using-Nextlow-NF-Core-RNAseq-Pipeline-on-Mox.html

### fastp expects input FastQ files to be in format: *[12]_val_[12].fq.gz
### E.g. heart_1_val_1.fq.gz


###################################################################################
# These variables need to be set by user

## Assign Variables

# Set FastQ filename patterns
fastq_pattern='*.fq.gz'
R1_fastq_pattern='*val_1.fq.gz'
R2_fastq_pattern='*val_2.fq.gz'

# Set number of CPUs to use
threads=40

# Input/output files
trimmed_checksums=trimmed_fastq_checksums.md5
fastq_checksums=input_fastq_checksums.md5

# FastQC output directory
output_dir=$(pwd)

# Data directories
reads_dir=/gscratch/scrubbed/samwhite/data/P_generosa/RNAseq/


## Inititalize arrays
raw_fastqs_array=()
R1_names_array=()
R2_names_array=()

# Paths to programs
fastp=/gscratch/srlab/programs/fastp.0.23.1
fastqc=/gscratch/srlab/programs/fastqc_v0.11.9/fastqc
multiqc=/gscratch/srlab/programs/anaconda3/bin/multiqc

# Programs associative array
declare -A programs_array
programs_array=(
[fastqc]="${fastqc}"
[fastp]="${fastp}" \
[multiqc]="${multiqc}"
)


###################################################################################

# Exit script if any command fails
set -e

# Load Python Mox module for Python module availability
module load intel-python3_2017

# Capture date
timestamp=$(date +%Y%m%d)

# Sync raw FastQ files to working directory
echo ""
echo "Transferring files via rsync..."
rsync --archive --verbose \
"${reads_dir}"${fastq_pattern} .
echo ""
echo "File transfer complete."
echo ""

### Run FastQC ###

### NOTE: Do NOT quote raw_fastqc_list
# Create array of trimmed FastQs
raw_fastqs_array=(${fastq_pattern})

# Pass array contents to new variable as space-delimited list
raw_fastqc_list=$(echo "${raw_fastqs_array[*]}")

echo "Beginning FastQC on raw reads..."
echo ""

# Run FastQC
${programs_array[fastqc]} \
--threads ${threads} \
--outdir ${output_dir} \
${raw_fastqc_list}

echo "FastQC on raw reads complete!"
echo ""

### END FASTQC ###

# Create arrays of fastq R1 files and sample names
# Do NOT quote R1_fastq_pattern variable
for fastq in ${R1_fastq_pattern}
do
  fastq_array_R1+=("${fastq}")

  # Use parameter substitution to remove all text up to and including last "." from
  # right side of string.
  R1_names_array+=("${fastq%%.*}")
done

# Create array of fastq R2 files
# Do NOT quote R2_fastq_pattern variable
for fastq in ${R2_fastq_pattern}
do
  fastq_array_R2+=("${fastq}")

  # Use parameter substitution to remove all text up to and including last "." from
  # right side of string.
  R2_names_array+=("${fastq%%.*}")
done


# Create MD5 checksums for raw FastQs
for fastq in ${fastq_pattern}
do
  echo "Generating checksum for ${fastq}"
  md5sum "${fastq}" | tee --append ${fastq_checksums}
  echo ""
done


### RUN FASTP ###

# Run fastp on files
# Adds JSON report output for downstream usage by MultiQC
# Trims 20bp from 5' end of all reads
# Trims poly G, if present
# Uses parameter substitution (e.g. ${R1_sample_name%%_*})to rm the _R[12] for report names.
echo "Beginning fastp trimming."
echo ""

for index in "${!fastq_array_R1[@]}"
do
  R1_sample_name="${R1_names_array[index]}"
  R2_sample_name="${R2_names_array[index]}"
  ${fastp} \
  --in1 ${fastq_array_R1[index]} \
  --in2 ${fastq_array_R2[index]} \
  --detect_adapter_for_pe \
  --trim_poly_g \
  --trim_front1 20 \
  --trim_front2 20 \
  --thread ${threads} \
  --html "${R1_sample_name%%_*}".fastp-trim."${timestamp}".report.html \
  --json "${R1_sample_name%%_*}".fastp-trim."${timestamp}".report.json \
  --out1 "${R1_sample_name}".fastp-trim."${timestamp}".fq.gz \
  --out2 "${R2_sample_name}".fastp-trim."${timestamp}".fq.gz

  # Generate md5 checksums for newly trimmed files
  {
      md5sum "${R1_sample_name}".fastp-trim."${timestamp}".fq.gz
      md5sum "${R2_sample_name}".fastp-trim."${timestamp}".fq.gz
  } >> "${trimmed_checksums}"
  
  # Remove original FastQ files
  echo ""
  echo " Removing ${fastq_array_R1[index]} and ${fastq_array_R2[index]}."
  rm "${fastq_array_R1[index]}" "${fastq_array_R2[index]}"
done

echo ""
echo "fastp trimming complete."
echo ""

### END FASTP ###


### RUN FASTQC ###

### NOTE: Do NOT quote ${trimmed_fastqc_list}

# Create array of trimmed FastQs
trimmed_fastq_array=(*fastp-trim*.fq.gz)

# Pass array contents to new variable as space-delimited list
trimmed_fastqc_list=$(echo "${trimmed_fastq_array[*]}")

# Run FastQC
echo "Beginning FastQC on trimmed reads..."
echo ""
${programs_array[fastqc]} \
--threads ${threads} \
--outdir ${output_dir} \
${trimmed_fastqc_list}

echo ""
echo "FastQC on trimmed reads complete!"
echo ""

### END FASTQC ###

### RUN MULTIQC ###
echo "Beginning MultiQC..."
echo ""
${multiqc} .
echo ""
echo "MultiQC complete."
echo ""

### END MULTIQC ###

####################################################################

# Capture program options
if [[ "${#programs_array[@]}" -gt 0 ]]; then
  echo "Logging program options..."
  for program in "${!programs_array[@]}"
  do
    {
    echo "Program options for ${program}: "
    echo ""
    # Handle samtools help menus
    if [[ "${program}" == "samtools_index" ]] \
    || [[ "${program}" == "samtools_sort" ]] \
    || [[ "${program}" == "samtools_view" ]]
    then
      ${programs_array[$program]}

    # Handle DIAMOND BLAST menu
    elif [[ "${program}" == "diamond" ]]; then
      ${programs_array[$program]} help

    # Handle NCBI BLASTx menu
    elif [[ "${program}" == "blastx" ]]; then
      ${programs_array[$program]} -help
    fi
    ${programs_array[$program]} -h
    echo ""
    echo ""
    echo "----------------------------------------------"
    echo ""
    echo ""
  } &>> program_options.log || true

    # If MultiQC is in programs_array, copy the config file to this directory.
    if [[ "${program}" == "multiqc" ]]; then
      cp --preserve ~/.multiqc_config.yaml multiqc_config.yaml
    fi
  done
fi


# Document programs in PATH (primarily for program version ID)
{
date
echo ""
echo "System PATH for $SLURM_JOB_ID"
echo ""
printf "%0.s-" {1..10}
echo "${PATH}" | tr : \\n
} >> system_path.log
