#!/usr/bin/env bash

######
# For certain samples, there are replicated data so these bam files need to be merged, and a merged bigwig file needs to be generated.
# This script checks if there are replicated bam files for each sample, and if there are, it merges the bam files and generates a merged bigwig file.
###### MAKE SURE THAT THIS IS RUN ON WHERE THE GATHERED INPUT FOLDER IS LOCATED ######

# This script is used to run the job on the cluster.

#PBS -N 20230704_bam_merge_rep_files
#PBS -P sbs_liyh

#PBS -q q256_free
#PBS -l select=1:ncpus=16:mpiprocs=16:mem=32gb
#PBS -l walltime=8:00:00

#PBS -m bea
#PBS -M suffi.azizan@ntu.edu.sg

#PBS -o /home/suffi.azizan/scratchspace/pipeline_scripts/footprinting-workflow-scripts/logs/pbs_oe.txt
#PBS -e /home/suffi.azizan/scratchspace/pipeline_scripts/footprinting-workflow-scripts/logs/pbs_oe.txt
#PBS -j oe

set -e
# Load modules
module purge
module load r/gcc6/4.2.0

# load conda environment
eval "$(conda shell.bash hook)"
conda activate snakemake_tobias

gathered_input_root=$INPUT
dataset_list=$DATALIST

# grab the basename of all the datasets listed in the dataset list text file
readarray -t datasets < "${dataset_list}"

# print the datasets
echo '[' "${datasets[@]}" ']'

# iterate through each dataset
for dataset in "${datasets[@]}"; do
	# get the path of the dataset in the gathered_input_root
    dataset_path=$(find "${gathered_input_root}" -maxdepth 1 -mindepth 1 -type d -name "${dataset}")
	# check if the dataset path is empty; if it is empty, skip
	if [ -z "${dataset_path}" ]; then
		continue
	fi
    # iterate through each sample
	for sample_dir in "${dataset_path}"/*; do
		# check if the file is a directory; if it is NOT a directory, skip
		if [ ! -d "${sample_dir}" ]; then
			continue
		else
			# get the sample name
			sample_name=$(basename "${sample_dir}")
			# capture the sample number
			if [[ $sample_name =~ (sample[0-9]{1,3}) ]]; then
  				substring="${BASH_REMATCH[1]}"
  				echo "Current directory: $substring of ${dataset}"
			fi
			
			# check whether there are replicated files for each sample; in other words, if there are more than 1 bam file for each sample, then merge the bam files
			if [ "$(find "${sample_dir}" -type f \( -name "*.trim.srt.nodup.no_chrM_MT.bam" -o -name "*.nodup.no_chrM_MT.bam" \) | wc -l)" -gt 1 ]; then
				# initialize an array to store the bam files
				bam_files=()
				# find the bam files and store the paths in the array
				readarray -t bam_files < <(find "${sample_dir}" -type f \( -name "*.trim.srt.nodup.no_chrM_MT.bam" -o -name "*.nodup.no_chrM_MT.bam" \))
				# check the bam file array for each sample
				echo "Bam files of ${sample_name}:" "${bam_files[@]}"
				echo "Multiple bam files have been found for ${sample_name}."
				echo "Merging the bam files..."
				# merge the bam files
				if samtools merge -@ 16 -o "${sample_dir}/${substring}.trim.srt.nodup.rep-merged.bam" "${bam_files[@]}"; then
					echo "Replicated bam files have been merged. Indexing..."
					# index the merged bam file
					if samtools index "${sample_dir}/${substring}.trim.srt.nodup.rep-merged.bam"; then
						echo "Replicated sorted bam files have been merged and indexed."
					fi
					# generate merged bigwig file from the merged bam file
					if bedtools genomecov -ibam "${sample_dir}/${substring}.trim.srt.nodup.rep-merged.bam" -bg | sort -k1,1 -k2,2n -T /home/suffi.azizan/scratchspace/tmp > "${sample_dir}/${substring}.trim.srt.nodup.rep-merged.bedgraph" && bedGraphToBigWig "${sample_dir}/${substring}.trim.srt.nodup.rep-merged.bedgraph" /home/suffi.azizan/scratchspace/inputs/genomes/GRCh38_EBV.chrom.sizes.tsv "${sample_dir}/${substring}.trim.srt.nodup.rep-merged.bigwig"; then
						echo "Merged bigwig file has been generated."
					else
						echo "Merged bigwig file has NOT been generated. One of the commands has failed."
					fi
				else
					echo "Replicated sorted bam files failed to be merged due to samtools error."
				fi
			else
				echo "There are no replicated sorted bam files for ${sample_dir}. Moving on to the next sample."	
			fi
		fi
	done
done
echo "All datasets to be analyzed have been checked for replicated sorted bam files."
