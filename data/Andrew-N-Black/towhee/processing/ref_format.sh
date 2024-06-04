#!/bin/bash
#SBATCH --job-name=ref_format_ncbi
#SBATCH -A fnrchook
#SBATCH -t 12-00:00:00 
#SBATCH -N 1 
#SBATCH -n 64
#SBATCH -e %x_%j.err
#SBATCH -o %x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=

module load bioinfo
module load bioawk
module load seqtk
module load samtools
module load BEDTools
module load BBMap
module load r
module load bedops
export PATH=$PATH:~/genmap-build/bin


###prep reference genome for mapping####
#Reduce fasta header length
#reformat.sh in=original.fa out=new.fa trd=t -Xmx20g overwrite=T
#sort by length
#sortbyname.sh in=new.fa out=ref.fa -Xmx20g length descending overwrite=T
#remove sequences smaller that 100kb prior to any repeatmasking
#bioawk -c fastx '{ if(length($seq) > 100000) { print ">"$name; print $seq }}' ref.fa > ref_100kb.fa
#rm new.fa

#index ref.fa and ref_100kb.fa for step3, step4, and step5 
#samtools faidx ref_100kb.fa

#prep repeatmasked file for later processing, create a rm.out if one is not available. 
	module --force purge
	module load biocontainers/default
	module load repeatmasker
	RepeatMasker -pa 64 -a -qq -species Melozone -dir . ref_100kb.fa 
	cat ref_100kb.fa.out  | tail -n +4 | awk '{print $5,$6,$7,$11}' | sed 's/ /\t/g' > repeats.bed 


module --force purge
module load bioinfo
module load bioawk
module load seqtk
module load samtools
module load cmake/3.9.4
module load BEDTools
module load BBMap
module load R
module load bedops
export PATH=$PATH:~/genmap-build/bin

####assess mappability of reference####
genmap index -F ref_100kb.fa -I index -S 50 # build an index 

# compute mappability, k = kmer of 100bp, E = # two mismatches
mkdir mappability
genmap map -K 100 -E 2 -T 64 -I index -O mappability -t -w -bg                

# sort bed 
sortBed -i repeats.bed > repeats_sorted.bed 

# make ref.genome
awk 'BEGIN {FS="\t"}; {print $1 FS $2}' ref_100kb.fa.fai > ref.genome 

# sort genome file
awk '{print $1, $2, $2}' ref.genome > ref2.genome
sed -i 's/ /\t/g' ref2.genome
sortBed -i ref2.genome > ref3.genome
awk '{print $1, $2 }' ref3.genome > ref_sorted.genome
sed -i 's/ /\t/g' ref_sorted.genome
rm ref.genome
rm ref2.genome
rm ref3.genome

# find nonrepeat regions
bedtools complement -i repeats_sorted.bed -g ref_sorted.genome > nonrepeat.bed

# clean mappability file, remove sites with <1 mappability                                                    
awk '$4 == 1' mappability/ref_100kb.genmap.bedgraph > mappability/map.bed                                           
awk 'BEGIN {FS="\t"}; {print $1 FS $2 FS $3}' mappability/map.bed > mappability/mappability.bed

# sort mappability 
sortBed -i mappability/mappability.bed > mappability/mappability2.bed
sed -i 's/ /\t/g' mappability/mappability2.bed

# only include sites that are nonrepeats and have mappability ==1
bedtools subtract -a mappability/mappability2.bed -b repeats_sorted.bed > mappability/map_nonreapeat.bed

# sort file -- by chr then site
bedtools sort -i mappability/map_nonreapeat.bed > mappability/filter_sorted.bed

# merge overlapping regions
bedtools merge -i mappability/filter_sorted.bed > mappability/merged.bed


# make bed file with the 100k and merged.bed (no repeats, mappability =1 sites) from below
awk '{ print $1, $2, $2 }' ref_100kb.fa.fai > ref_100kb.info

# replace column 2 with zeros
awk '$2="0"' ref_100kb.info > ref_100kb.bed

# make tab delimited
sed -i 's/ /\t/g' ref_100kb.bed

# only include scaffolds in merged.bed if they are in ref_100kb.bed
bedtools intersect -a ref_100kb.bed -b ./mappability/merged.bed > ok.bed	

# make chrs.txt
cut -f 1 ref_100kb.bed > chrs.txt
	



