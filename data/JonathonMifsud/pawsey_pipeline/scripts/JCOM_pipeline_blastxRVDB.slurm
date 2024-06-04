#!/bin/bash
###############################################################################################################
#                                            BatchArtemisSRAMiner                                             #   
#                                                JCO Mifsud                                                   # 
#                                                   2023                                                      # 
#                                                                                                             #
###############################################################################################################

# Underlying.slurm script that is run using the wrapper script YOURPROJECT_blastxRVDB.sh
# This script will run blastx on eacj contigs.fa file against the RVDB database
# the RVDB is a viral database that contains DNA and RNA viruses
# it is useful for picking up non-rdrp segements / contigs and DNA viruses
# It will then extract the contigs that have a blast hit to the RVDB database which can be used in later steps such as Blastx against the NR and NT databases.    

#PBS -P RDS-FSC-jcomvirome-RW
#SBATCH --job-name="blastx_RVDB_array"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=120G

#SBATCH --mail-user="$user@uni.sydney.edu.au"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=work

module load singularity/3.11.4

# blastx
function BlastxRVDB {
    singularity exec "$singularity_image" diamond blastx -q "$inpath"/"$library_id".contigs.fa -d "$db" -t "$tempdir" -o "$outpath"/"$library_id"_RVDB_blastx_results.txt -e 1E-10 -c1 -k 5 -b "$MEM" -p "$CPU" -f 6 qseqid qlen sseqid stitle pident length evalue --ultra-sensitive
}

#tool to extract contigs from trinity assembly Blast to fasta
function blastToFasta {
    grep -i ".*" "$outpath"/"$library_id"_RVDB_blastx_results.txt | cut -f1 | sort | uniq > "$outpath"/"$library_id""_temp_contig_names.txt" #by defult this will grab the contig name from every blast result line as I commonly use a custom protein database containing only viruses
	grep -A1 -I -Ff "$outpath"/"$library_id""_temp_contig_names.txt" "$inpath"/"$library_id".contigs.fa > "$outpath"/"$library_id"_RVDB_blastcontigs.fasta
    sed -i 's/--//' "$outpath"/"$library_id"_RVDB_blastcontigs.fasta # remove -- from the contigs
    sed -i '/^[[:space:]]*$/d' "$outpath"/"$library_id"_RVDB_blastcontigs.fasta # remove the white space
    sed --posix -i "/^\>/ s/$/"_$library_id"/" "$outpath"/"$library_id"_RVDB_blastcontigs.fasta # annotate the contigs
    rm "$outpath"/"$library_id""_temp_contig_names.txt"
}

# read in list of file names or accessions for example could be several fastq.gz files (paired or single) or just the accession id's
readarray -t myarray < "$file_of_accessions"
export library_run=${myarray["$SLURM_ARRAY_TASK_ID"]}
library_run_without_path="$(basename -- $library_run)"
library_id=$(echo $library_run_without_path | sed 's/\.contigs.fa//g')

# variables change these to whatever you like!
wd=/scratch/director2187/$user/"$root_project"/"$project"/blast_results
inpath=/scratch/director2187/$user/"$root_project"/"$project"/contigs/final_contigs   # location of reads and filenames
outpath=/scratch/director2187/$user/"$root_project"/"$project"/blast_results        # location of megahit output
tempdir=/scratch/director2187/$user/"$root_project"/
CPU=12
MEM=1       

# cd working dir
cd "$wd" || exit

BlastxRVDB
blastToFasta

