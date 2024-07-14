#!/bin/bash
#
#SBATCH --chdir=/nobackup/proj/clsclmr/Ludwig_2019
#SBATCH -p defq
#SBATCH -A clsclmr
#SBATCH -t 48:00:00
#SBATCH -c 8
#

  ## load modules
module load Python/3.8.6-GCCcore-10.2.0;
module load SAMtools/1.12-GCC-10.2.0; 
module load Bowtie2/2.3.4.2-foss-2018b;
module load parallel/20200522-GCCcore-10.2.0

export PATH=`pwd`/software/bin/:$PATH


  ## Parse Arguments ##

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
  -b|--bam-directory)
    bamdir="$2"
    shift # past argument
    shift # past value
    echo "Bam directory: ${bamdir}"
  ;;

  -r|--reference)
    ref="$2" 
    echo "Reference: ${ref}"
    shift # past argument
    shift # past value 
  ;;

  -g|--group-name)
    g="$2"
    echo "Group name: ${g}"
    shift # past argument
    shift # past value 
  ;;

  -h|--help)
    echo "-g|--group-name       base name for data/group_(group_name)_SRRs.txt, eg. SRP149534"
    echo "-r|--reference        nuc/reference.fa file root name (without .fa), eg. nuc/parent_consensus or nuc/hg38"
    echo "-b|--bam-directory    name of bam directory without / eg. bam_consensus_nodups"
    shift # past argument
    exit 1 
  ;;

  -*|--*)
    echo "Unknown option ${1}. Valid aruguments:"
    echo "-g|--group-name       base name for data/group_(group_name)_SRRs.txt, eg. SRP149534"
    echo "-r|--reference        nuc/reference.fa root name without .fa, eg. nuc/parent_consensus"
    echo "-b|--bam-directory    name of bam directory without / eg. bam_cnodups"
    exit 1
  ;; 

  *)
    POSITIONAL_ARGS+=("$1") # save positional arg
    shift # past argument
  ;;

  esac
done

mkdir $bamdir


# read bulk ATAC-seq from TF1 cells into array
echo "Using group name (-g|--group-name): \"${g}\" to read SRR names from data/group_${g}_SRRs.txt"
if test -f "data/group_${g}_SRRs.txt"; then
  readarray -t rts < data/group_${g}_SRRs.txt;
else
  echo "Error: data/group_${g}_SRRs.txt could not be found. Make sure --group|-g is correct, and the file exists.  can be created manually (Line separated list of SRR names) or using a keyword with prefetch_files.sh. Exiting..."
  exit 1
fi


  ## build indices and set reference genome ##

if test -f "nuc/${ref}.1.bt2"; then
  echo "Reference indexes (${ref}.1.bt2 etc.) found."
elif test -f "nuc/${ref}.fa"; then
  echo "Reference fasta file found but not indexed, indexing..."
  bowtie2-build --threads 8 ${ref}.fa ${ref}; 
  echo "bowtie2 reference indices built"
else
  echo "Error: Fasta of reference sequence and its indexes (base file name: \"${ref}\") could not be found. Make sure --ref|-r is correct and the reference fasta file (.fa) has been downloaded. Exiting..."
  exit 1
fi

# tmp change ref
#ref="nuc/hg38"
 


#locale;
#export LANG=en_GB.utf8
#export LC_ALL="en_GB.utf8" 
#locale;


  ## Align reads ##

for rt in "${rts[@]}"
do

 echo ${rt};
 #echo Number of reads: $(cat fastq/${rt}.fastq|wc -l)/4|bc

 if [ -f "${bamdir}/${rt}.bam.bai" ]; then
  echo "${rt} already aligned";
 else 
   
  echo "Aligning ${rt} to ${ref} reference genome...";
  echo ${rt} >> alignment_stats/alignment_stdout.txt
# bowtie2 parameters: 
# -p 8 cores, -1 forward and -2 reverse read, -x ref, --local alignment (soft-clipping allowed), very sensitive (-L 20: 20 bp substrings in multiseed, -i s,1,0.50: shorter intervals between seed substrings, -D 20 -R 3: see manual), -t: time to align in stout,  out? -X 2000???.

# samtools view parameters: 
# - - (input from stdin, AGAIN to output to stout), -h (header), eg. -F 0 (do not output alignments with FLAG integer), eg. -q 10 (skip alignments with MAPQ quality <10), --un-gz unmapped reads to zipped file, -u outputs uncompressed bam into pipe.

# samtools sort: normally sorts by leftmost coordinates (for indexing), -n sorts by QNAME (for fixmate - so mate pairs can be labelled) 

# samtools fixmate adds mate score (ms) tags (used by markdup to select best reads): -m ms tags.

# samtools markdup: -s print basic stats, -r will REMOVE duplicates, 

  bowtie2 -p 8 -1 fastq/${rt}_1.fastq.gz -2 fastq/${rt}_2.fastq.gz -x ${ref} --local --sensitive -t --un-gz fastq/${rt}_unmapped.fastq.gz 2>> alignment_stats/alignment_stdout.txt | samtools view --threads 8 - -h -u | samtools sort --threads 8 -n - -u | samtools fixmate --threads 8 -m -u - - | samtools sort --threads 8 - -u | samtools markdup --threads 8 -s - ${bamdir}/${rt}.bam 2>> alignment_stats/alignment_stdout.txt 
# To align without marking dups, change the first samtools sort in pipe to index by coordinates (remove -n argument) and immediatelyoutput bam for indexing below.
# TODO add removal of optical duplicates with -d _ : what distance for Nextseq?

  # index sorted bam files:
  echo "Index"; 
  samtools index -@ 8 ${bamdir}/${rt}.bam ;

 fi
done


#locale;
export LANG=C.UTF-8 ; 
export LC_ALL= ;
locale;


# Copy stats from slurm outfile (stout) to alignment stats
cp slurm-${SLURM_JOB_ID}.out alignment_stats/alignment_stdout.txt
echo "Reference: ${ref} " >> alignment_stats/alignment_and_duplicate_summary_${bamdir}.txt
echo 'SRR Overall_alignment_rate Total_reads_bowtie2 Total_reads_markdup Total_duplicates Estimated_unique_lib_size' >> alignment_stats/alignment_and_duplicate_summary_${bamdir}.txt

for rt in "${rts[@]}"
do
 overall_alignment=`grep -A 20 "$rt" alignment_stats/alignment_stdout.txt | grep "overall" | cut -d "%" -f 1`;
 bowtie2_total_reads=`grep -A 20 "$rt" alignment_stats/alignment_stdout.txt | grep "reads; of these:" | cut -d " " -f 1`;
 markdup_total_reads=`grep -A 41 "$rt" alignment_stats/alignment_stdout.txt | grep "READ:" | cut -d " " -f 2`;
 total_duplicates=`grep -A 41 "$rt" alignment_stats/alignment_stdout.txt | grep "DUPLICATE TOTAL:" | cut -d " " -f 3`;
 unique_lib_size=`grep -A 41 "$rt" alignment_stats/alignment_stdout.txt | grep "ESTIMATED_LIBRARY_SIZE" | cut -d " " -f 2`; 
 echo "$rt $overall_alignment $bowtie2_total_reads $markdup_total_reads $total_duplicates $unique_lib_size" >> alignment_stats/alignment_and_duplicate_summary_${bamdir}.txt 
done


module purge;
