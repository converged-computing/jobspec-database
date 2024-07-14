
#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30 # max cores per user on highmem is 32 
#SBATCH --mem-per-cpu=3G # 1 GB per 1M reads (for a total of 90 GB over 30 CPUs)
#SBATCH --time=0-20:00:00 # 3 days, 1 HR per 1M reads
#SBATCH --output=transcript.quant.stdout
#SBATCH --mail-user=danielaz@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="transcript.quant.job"
#SBATCH -p intel # Available paritions: intel, batch, highmem, gpu, short (each has walltime and memory limits)

# Print current date
date

# Change directory to where you submitted the job from, so that relative paths resolve properly
cd /rhome/danielaz/bigdata/transcriptomics/raw_fastq

module load trinity-rnaseq/2.14.0

FASTQ=/rhome/danielaz/bigdata/transcriptomics/raw_fastq
SAMPLES=/rhome/danielaz/bigdata/transcriptomics/trinity/neoclara.trinity.IDs.txt
DE_NOVO=/rhome/danielaz/bigdata/transcriptomics/trinity/neoclara.deNovo.fa


align_and_estimate_abundance.pl --seqType fq  \
    --samples_file ${SAMPLES}  --transcripts ${DE_NOVO} \
    --est_method salmon  --trinity_mode   --prep_reference  

# --seqType : data input type, either fasta or fastq
# --samples file : list of samples
# --transcripts : de novo transcriptome
# --output-dir : name of directory to output files in (default to name of sample)
# --est_method : abundance estimation method (alignment free - Salmon)
# --trinity_mode : will automatically generated the gene_trans_map
# --prep_reference : prep reference (builds target index)

# NOTE: the sample file's second column (replicate) will be the name of the output directory.
# OUTPUTS: A directory per individual provided in the sample file with these files: 
    # aux_info dir
    # cmd_info.json
    # lib_format_counts.json
    # libParams dir
    # logs dir 
    # quant.sf
    # quant.sf.genes 

    # The quant.sf file is the file of interest and has the following information:
        # NumReads’ corresponding to the number of RNA-Seq fragments predicted to be derived from that transcript
        # TPM: transcripts per million 

# print name of node
hostname

# run time < 3 hours 
