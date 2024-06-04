#!/usr/bin/env bash

#PBS -W group_list=bhurwitz
#PBS -q standard
#PBS -l place=free:shared
#PBS -l select=1:ncpus=12:mem=68gb
#PBS -l walltime=12:00:00
#PBS -l cput=144:00:00
#PBS -M scottdaniel@email.arizona.edu
#PBS -m bea

#make sure this matches ncpus in the above header!
export THREADS="--threads 12"
#
# runs centrifuge, brought to you by the good people who brought you bowtie2
#

# --------------------------------------------------
# singularity is needed to run singularity images
module load singularity
#
#LOAD REQUIRED R MODULES
#module load unsupported
#module load markb/R/3.1.1

# --------------------------------------------------

unset module
set -u

CONFIG="$SCRIPT_DIR/config.sh"

if [ -e $CONFIG ]; then
  . "$CONFIG"
else
  echo Missing config \"$CONFIG\"
  exit 1
fi

cd $PRJ_DIR

echo Host \"$(hostname)\"

echo Started $(date)

TMP_FILES=$(mktemp)

get_lines $TODO $TMP_FILES $PBS_ARRAY_INDEX $STEP_SIZE

NUM_FILES=$(lc $TMP_FILES)

if [[ $NUM_FILES -lt 1 ]]; then
    echo Something went wrong or no files to process
    exit 1
else
    echo Found \"$NUM_FILES\" files to process
fi

#RUN CENTRIFUGE ON ALL SEQUENCE FILES FOUND IN FIXED_DIR
while read FASTA; do

    if [[ $FASTA =~ "Long" ]]; then
        
        export cent="singularity exec \
            -B $DNA_LONG_DIR:$SING_WD,$(dirname $CENT_DB):$SING_CENT \
            $SING_IMG/centrifuge.img centrifuge" 

        BASE=$(basename $FASTA .fastq)
        R1=$SING_WD/$(basename $FASTA)

        OUT_DIR=$SING_WD/$(basename $CFUGE_LONG_DIR)

        echo "Doing Sample $BASE"

        $cent -x $SING_CENT/$DB -U $R1\
            -S $OUT_DIR/"$BASE"_centrifuge_hits.tsv \
            --report-file $OUT_DIR/"$BASE"_centrifuge_report.tsv \
            -$FILE_TYPE \
            --exclude-taxids $EXCLUDE \
            --host-taxids $INCLUDE \
            $THREADS
    else

        export cent="singularity exec \
            -B $DNA_OR_DIR:$SING_WD,$(dirname $CENT_DB):$SING_CENT \
            $SING_IMG/centrifuge.img centrifuge" 

        BASE=$(basename $FASTA .fastq)
        R1=$SING_WD/$(basename $FASTA)

        OUT_DIR=$SING_WD/$(basename $CFUGE_OR_DIR)

        echo "Doing Sample $BASE"

        $cent -x $SING_CENT/$DB -U $R1\
            -S $OUT_DIR/"$BASE"_centrifuge_hits.tsv \
            --report-file $OUT_DIR/"$BASE"_centrifuge_report.tsv \
            -$FILE_TYPE \
            --exclude-taxids $EXCLUDE \
            --host-taxids $INCLUDE \
            $THREADS

    fi

done < $TMP_FILES

echo "Finished at $(date)"
#CREATE BUBBLE PLOT VISUALIZATION
#just do this interactively later
#or not, try one more time
#still doesnt work
#TODO: make singularity container for R and try that way?
#$WORKER_DIR/centrifuge_bubble.R -d $CFUGE_DIR -o $PLOT_OUT -f $PLOT_FILE -t $PLOT_TITLE

#
#Centrifuge version 1.0.3-beta by the Centrifuge developer team (centrifuge.metagenomics@gmail.com)
#Usage:
#  centrifuge [options]* -x <cf-idx> {-1 <m1> -2 <m2> | -U <r> | --sra-acc <SRA accession number>} [-S <filename>] [--report-file <report>]
#
#  <cf-idx>   Index filename prefix (minus trailing .X.cf).
#  <m1>       Files with #1 mates, paired with files in <m2>.
#             Could be gzip'ed (extension: .gz) or bzip2'ed (extension: .bz2).
#  <m2>       Files with #2 mates, paired with files in <m1>.
#             Could be gzip'ed (extension: .gz) or bzip2'ed (extension: .bz2).
#  <r>        Files with unpaired reads.
#             Could be gzip'ed (extension: .gz) or bzip2'ed (extension: .bz2).
#  <SRA accession number>        Comma-separated list of SRA accession numbers, e.g. --sra-acc SRR353653,SRR353654.
#  <filename>      File for classification output (default: stdout)
#  <report>   File for tabular report output (default: centrifuge_report.tsv)
#
#  <m1>, <m2>, <r> can be comma-separated lists (no whitespace) and can be
#  specified many times.  E.g. '-U file1.fq,file2.fq -U file3.fq'.
#
#Options (defaults in parentheses):
#
# Input:
#  -q                 query input files are FASTQ .fq/.fastq (default)
#  --qseq             query input files are in Illumina's qseq format
#  -f                 query input files are (multi-)FASTA .fa/.mfa
#  -r                 query input files are raw one-sequence-per-line
#  -c                 <m1>, <m2>, <r> are sequences themselves, not files
#  -s/--skip <int>    skip the first <int> reads/pairs in the input (none)
#  -u/--upto <int>    stop after first <int> reads/pairs (no limit)
#  -5/--trim5 <int>   trim <int> bases from 5'/left end of reads (0)
#  -3/--trim3 <int>   trim <int> bases from 3'/right end of reads (0)
#  --phred33          qualities are Phred+33 (default)
#  --phred64          qualities are Phred+64
#  --int-quals        qualities encoded as space-delimited integers
#  --ignore-quals     treat all quality values as 30 on Phred scale (off)
#  --nofw             do not align forward (original) version of read (off)
#  --norc             do not align reverse-complement version of read (off)
#  --sra-acc          SRA accession ID
#
#Classification:
#  --min-hitlen <int>    minimum length of partial hits (default 22, must be greater than 15)
#  --min-totallen <int>  minimum summed length of partial hits per read (default 0)
#  --host-taxids <taxids> comma-separated list of taxonomic IDs that will be preferred in classification
#  --exclude-taxids <taxids> comma-separated list of taxonomic IDs that will be excluded in classification
#
# Output:
#  --out-fmt <str>       define output format, either 'tab' or 'sam' (tab)
#  --tab-fmt-cols <str>  columns in tabular format, comma separated
#                          default: readID,seqID,taxID,score,2ndBestScore,hitLength,queryLength,numMatches
#  -t/--time             print wall-clock time taken by search phases
#  --un <path>           write unpaired reads that didn't align to <path>
#  --al <path>           write unpaired reads that aligned at least once to <path>
#  --un-conc <path>      write pairs that didn't align concordantly to <path>
#  --al-conc <path>      write pairs that aligned concordantly at least once to <path>
#  (Note: for --un, --al, --un-conc, or --al-conc, add '-gz' to the option name, e.g.
#  --un-gz <path>, to gzip compress output, or add '-bz2' to bzip2 compress output.)
#  --quiet               print nothing to stderr except serious errors
#  --met-file <path>     send metrics to file at <path> (off)
#  --met-stderr          send metrics to stderr (off)
#  --met <int>           report internal counters & metrics every <int> secs (1)
#
# Performance:
#  -o/--offrate <int> override offrate of index; must be >= index's offrate
#  -p/--threads <int> number of alignment threads to launch (1)
#  --mm               use memory-mapped I/O for index; many 'bowtie's can share
#
# Other:
#  --qc-filter        filter out reads that are bad according to QSEQ filter
#  --seed <int>       seed for random number generator (0)
#  --non-deterministic seed rand. gen. arbitrarily instead of using read attributes
#  --version          print version information and quit
#  -h/--help          print this usage message
