#!/bin/bash
# Aims: To investigate the Whole-Genome (WG) insertion/deletion (indel) profile of rice genomic samples under TALEN treatment.
# Author: Chun-Yuan Huang, Ph.D.
# Last Update: 2/12/2016

##-------------------------------------------
## Workflow description:
#1. Blat of sample contigs (by SOAP-denovo assembly) against WG reference.
#   To speed up the blat step, sample contigs are splitted into several parts to run blat in parallel.
#2. Generate filtered blat hits table for contigs that contain gaps whose total length
#   larger than the defined size (e.g. GAP_MINSIZE=100, defined in the source file).
#3. Calculate the Q_gapStarts, Q_gapSizes, T_gapStarts and T_gapSizes based on each gap using the block info
#   from the Blat result, where Q is for Query (the samples) and T is for Target (the reference genome)
#4. Generate indel tables for each insertion and deletion.
#5. Make indel summary table that combines same indels from all samples in each row based on the indel features
#6. Use the indel summary table to retrieve indel sequences.

source ./0SOURCE

#-----
#1. Blat of sample contigs (by SOAP-denovo assembly) against WG reference.
# psl header is below. rep_match is Number of matching bases that are part of repeats. \
# Q is for Query, T is for Target, or Sample/Reference in Blastn.
# block_count is Number of blocks in the alignment. blockSizes is Comma-separated list of sizes of each block.
# match(#1), mis-match(#2), rep_match(#3), nCount(#4), Q_gap_count(#5), Q_gap_bases(#6), T_gap_count(#7), T_gap_bases(#8),
# strand(#9), Q_name(#10), Q_size(#11), Q_start(#12), Q_end(#13), T_name(#14), T_size(#15), T_start(#16), T_end(#17),
# block_count(#18), blockSizes(#19), qStarts(#20), tStarts(#21).
# psl have gap info in fields #5 (Q_gap_count), #6 (Q_gap_bases), #7 (T_gap_count), #8 (T_gap_bases)
# psl can be further simplified to the best representatives using pslReps

for i in ${SAMPLE}
do
QUERY=${i}-${SEQ}
QUERYNAME=${i}.${SEQNAME}
echo "
#!/bin/bash
#PBS -m abe
#PBS -l nodes=1:ppn=${PPN},vmem=${VMEM}gb,walltime=${WALLTIME}:00:00
#PBS -M ${EMAIL}
#PBS -N x1-WGblat
#PBS -j oe
module add python
module add blat

cd ${prereq_DIR}
# split sample into parts
\rm -r ${QUERY}.*
gt splitfasta -numfiles ${PARTS} ${QUERY}

# run blat of sample parts in parallel
cd ${run_DIR}
for ((k=1; k<=${PARTS}; k++))
do
blat -t=dna -q=dna -out=psl -minIdentity=${PERC_IDENTITY} -minScore=${MIN_SCORE} -mask=lower \
${prereq_DIR}/${REFSEQ} ${prereq_DIR}/${QUERY}.\$k blat.${DB_REFSEQ}.${QUERYNAME}.\$k.psl &
done

# wait until all blat processes are finished
wait

# Combine all parts back into one file
cd ${run_DIR}
\rm blat.${DB_REFSEQ}.${QUERYNAME}.psl
\rm blat.${DB_REFSEQ}.${QUERYNAME}.Reps.psl
\rm blat.${DB_REFSEQ}.${QUERYNAME}.Reps.psr

for ((k=1; k<=${PARTS}; k++))
do
tail -n+6 blat.${DB_REFSEQ}.${QUERYNAME}.\$k.psl |\
cat - >> blat.${DB_REFSEQ}.${QUERYNAME}.psl
done

# Put back the header for the Reps.psl file
head -5 blat.${DB_REFSEQ}.${QUERYNAME}.1.psl |\
cat - blat.${DB_REFSEQ}.${QUERYNAME}.psl > tmp && \
\mv tmp blat.${DB_REFSEQ}.${QUERYNAME}.psl

# Filter for only the representative records
pslReps -singleHit blat.${DB_REFSEQ}.${QUERYNAME}.psl blat.${DB_REFSEQ}.${QUERYNAME}.Reps.psl \
blat.${DB_REFSEQ}.${QUERYNAME}.Reps.psr

" > WGblat_${i}-on-${REFSEQNAME}.qsub
qsub WGblat_${i}-on-${REFSEQNAME}.qsub

done

