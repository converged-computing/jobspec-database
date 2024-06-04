#!/usr/bin/env bash

#Author: George Gruenhagen
#Purpose: Calculate read counts per allele for allele-specific expression analysis of RNAseq data

#Usage statement
usage () {
        echo "Usage: ase_count.bash <reference> <vcf_file_path> <gatk_file_path> [-I <bam_file>... or -D <bam_dir>] [options]
		reference: the path to the reference genome
		vcf_file_path: the path to the vcf file
		gatk_file_path: the path to a gatk executable binary
		-I bam file(s) to call (must use -I or -D)
		-D directory of bam files (must use -I or -D)
		
		-O name of output directory of output tables (default: output_table)
		-N Name of Job
		-l memory
		-t hh:mm:ss time needed, job will be killed if exceeded (default: walltime=80:00:00)
		-q specifies process queue
		-j controls what gets written to the ouputfile
		-o name of the job's outputfile
		-m controls when email is sent to the submitter
		-M email of submitter"
}

check_for_help() {
	if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
		usage;
		exit 0;
	fi
}

#Command-line options
get_input() {
	check_for_help "$1"
	
	ref=$1
	vcf=$2
	gatk=$3
	shift
	shift
	shift
	
	out="output_table"
	name="ase_count"
	memory="mem=16gb"
	time="walltime=80:00:00"
	cluster="biocluster-6"
	writingOpts="oe"
	outputFile="ase_count.out"
	emailOpts="abe"
	email="ggruenhagen3@gatech.edu"
	while getopts "I:D:O:N:l:t:q:j:o:m:M:h" opt; do
		case $opt in
		I ) bams+=("$OPTARG");;
		D ) bamDir=$OPTARG;;
		O ) out=$OPTARG ;;
		N ) name=$OPTARG ;;
		l ) memory=$OPTARG ;;
		t ) time=$OPTARG ;;
		q ) cluster=$OPTARG ;;
		j ) writingOpts=$OPTARG ;;
		o ) outputFile=$OPTARG ;;
		m ) emailOpts=$OPTARG ;;
		M ) email=$OPTARG ;;
		h ) usage; exit 0;
		esac
	done
}

check_files() {
	mkdir -p "$out"

	# If the bamDir is empty, then the user must have supplied -I bam files
	if [ -z "$bamDir" ]; then
		for bam in "${bams[@]}"; do
			if [ -z "$bam" ] || [ ! -f "$bam" ]; then
				echo "The bam file $bam does not exist. Exiting the program."
				usage
				exit 1
			fi
		done
	else
		# The user must have supplied a bam directory
		bamsString=""
		i=0
		for file in $bamDir/*.bam; do
			if [ -f "$file" ]; then
				((i=i+1))
				filename=$(basename -- "$file")
				no_ext="${filename%.*}"
				bamsString+="$gatk ASEReadCounter -R $ref -V biallelic.vcf -O $out/output.table.$no_ext -I $file "$'\n'
			fi
		done
		if [ -z "$bamsString" ]; then
			echo "No bam files in directory: $bamDir. Exiting the program."
			usage 
			exit 1
		fi
	fi
	
	if [ -z "$gatk" ] || [ ! -f "$gatk" ]; then
		echo "Invalid gatk file path. Exiting the program."
		usage
		exit 1
	fi
	if [ -z "$ref" ] || [ ! -f "$ref" ]; then
		echo "The reference file does not exist. Exiting the program."
		usage
		exit 1
	fi
	if [ -z "$vcf" ] || [ ! -f "$vcf" ]; then
		echo "Invalid vcf file path. Exiting the program."
		usage
		exit 1
	fi
}

list_bams() {
	if [ -z "$bamDir" ]; then
		bamsString=""
		i=0
		for bam in "${bams[@]}"; do
			((i=i+1))
			filename=$(basename -- "$file")
			no_ext="${filename%.*}"
			bamsString+="$gatk ASEReadCounter -R $ref -V biallelic.vcf -O $out/output.table.$no_ext -I $bam "$'\n'
		done
	fi
}

generate_pbs() {
	echo "#PBS -N $name
#PBS -l nodes=2:ppn=4
#PBS -l $time
#PBS -q $cluster
#PBS -j $writingOpts
#PBS -o $outputFile
#PBS -m $emailOpts
#PBS -M $email

cd \$PBS_O_WORKDIR
module load java/1.8.0_25
module load samtools/0.1.19
module load intel/14.0.2
module load perl/5.14.2
module load vcftools/0.1.14.10

# Purpose: Prepare vcf for ASEReadCounter. This means keep only SNPs and biallelic sites
$gatk SelectVariants --restrict-alleles-to BIALLELIC -R $ref -V $vcf -O biallelic.vcf --select-type-to-include SNP

# Do the actual ASE Read Counting
$bamsString
" > ase_count.pbs
}

main() {
	get_input "$@"
	check_files
	list_bams
	generate_pbs
	
	qsub ase_count.pbs
}


main "$@"
