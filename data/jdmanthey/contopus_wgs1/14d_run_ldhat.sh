#!/bin/sh
#SBATCH --chdir=./
#SBATCH --job-name=ldhat
#SBATCH --nodes=1 --ntasks=2
#SBATCH --partition quanah
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-237

module load intel R

source activate bcftools

# define variables

chrom_array=$( head -n${SLURM_ARRAY_TASK_ID} ldhat_helper.txt | tail -n1 | cut -f2 )

start_array=$( head -n${SLURM_ARRAY_TASK_ID} ldhat_helper.txt | tail -n1 | cut -f3 )

end_array=$( head -n${SLURM_ARRAY_TASK_ID} ldhat_helper.txt | tail -n1 | cut -f4 )

workdir=/lustre/scratch/jmanthey/15_contopus/08_ldhat

# eastern ldhat window
output=eastern

# add the header
cat ${output}_header.txt > ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.recode.vcf

# pull out the interval
tabix ${workdir}/${output}_ldhat.vcf.gz ${chrom_array}:${start_array}-${end_array} >> ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.recode.vcf

# simplify the vcf
bcftools query -f '%POS\t%REF\t%ALT[\t%GT]\n' ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.recode.vcf > ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.simple.vcf

# convert to the format for ldhat
Rscript make_ldhat_files.r ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.simple.vcf

$ run ldhat interval
/lustre/work/jmanthey/LDhat-master/interval -seq ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.sites \
-loc ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.locs \
-lk ${workdir}/${output}_lk.txt -bpen 10 -its 3000000 -samp 3000 -prefix ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.

# run ldhat stat for summarization
/lustre/work/jmanthey/LDhat-master/stat -input ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.rates.txt \
-burn 301 -prefix ${workdir}/windows/${output}__${chrom_array}__${start_array}__${end_array}.

# rename the mapping file 
mv ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.mapping ${workdir}/windows/${output}__${chrom_array}__${start_array}__${end_array}.mapping

# remove all extra intermediate files
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.locs
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.recode.vcf
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.simple.vcf
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.sites
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.bounds.txt
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.new_lk.txt
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.type_table.txt
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.rates.txt



# western ldhat window
output=western

# add the header
cat ${output}_header.txt > ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.recode.vcf

# pull out the interval
tabix ${workdir}/${output}_ldhat.vcf.gz ${chrom_array}:${start_array}-${end_array} >> ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.recode.vcf

# simplify the vcf
bcftools query -f '%POS\t%REF\t%ALT[\t%GT]\n' ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.recode.vcf > ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.simple.vcf

# convert to the format for ldhat
Rscript make_ldhat_files.r ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.simple.vcf

$ run ldhat interval
/lustre/work/jmanthey/LDhat-master/interval -seq ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.sites \
-loc ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.locs \
-lk ${workdir}/${output}_lk.txt -bpen 10 -its 3000000 -samp 3000 -prefix ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.

# run ldhat stat for summarization
/lustre/work/jmanthey/LDhat-master/stat -input ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.rates.txt \
-burn 301 -prefix ${workdir}/windows/${output}__${chrom_array}__${start_array}__${end_array}.

# rename the mapping file 
mv ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.mapping ${workdir}/windows/${output}__${chrom_array}__${start_array}__${end_array}.mapping

# remove all extra intermediate files
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.locs
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.recode.vcf
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.simple.vcf
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.sites
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.bounds.txt
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.new_lk.txt
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.type_table.txt
rm ${workdir}/windows/${chrom_array}__${start_array}__${end_array}.rates.txt







