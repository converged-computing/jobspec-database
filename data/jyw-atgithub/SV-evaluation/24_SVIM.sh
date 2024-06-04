#!/bin/bash

#SBATCH --job-name=SVIM
#SBATCH -A jje_lab
#SBATCH -p standard
#SBATCH --array=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=100G                 # requesting 180 GB (1 GB = 1,024 MB) local scratch
#SBATCH --constraint=fastscratch   # Intel AVX512 with /tmp on NVMe disk

source ~/.bashrc

ref="/dfs7/jje/jenyuw/Eval-sv-temp/reference"
ref_genome="${ref}/r649.rename.fasta" ##REMEMBER it is renamed
trimmed="/dfs7/jje/jenyuw/Eval-sv-temp/results/trimmed"
aligned_bam="/dfs7/jje/jenyuw/Eval-sv-temp/results/alignment"
SVs="/dfs7/jje/jenyuw/Eval-sv-temp/results/SVs"

nT=$SLURM_CPUS_PER_TASK

file=`head -n $SLURM_ARRAY_TASK_ID ${trimmed}/namelist_2.txt |tail -n 1`
name=`echo ${file} | cut -d '/' -f 8 |cut -d '.' -f 1 `
read_type=`echo ${name} | cut -d '_' -f 1 `
file="${aligned_bam}/${name}.trimmed-ref.sort.bam"

module load python/3.10.2
#"svim reads"requires samtools, but "svim alignment" does not
##some parameter names are changed. Check the svim alignment --help messages.
#--max_consensus_length 100000
svim alignment --sample ${name}_svim \
--min_mapq 20 --min_sv_size 50 \
--max_sv_size 10000000 \
--position_distance_normalizer 900 --cluster_max_distance 0.3 \
${SVs}/${name}_SVIM ${file} ${ref_genome}

cp ${SVs}/${name}_SVIM/variants.vcf ${SVs}/${name}.SVIM.vcf

module unload python/3.10.2

echo "This is the end"