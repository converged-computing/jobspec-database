#!/bin/bash
#PBS -m abe
#PBS -l nodes=1:ppn=20,vmem=40gb,walltime=12:00:00
#PBS -N x2-WGindelT
#PBS -j oe

module add python
module add blat

# Aims: To investigate the Whole-Genome (WG) insertion/deletion (indel) profile of rice genomic samples under TALEN treatment.
# Author: Chun-Yuan Huang, Ph.D.
# Last Update: 2/12/2016

cd $PBS_O_WORKDIR
source ./0SOURCE
cd ${run_DIR}
for i in ${SAMPLE}
do
QUERY=${i}-${SEQ}
QUERYNAME=${i}.${SEQNAME}

echo "
#!/bin/bash
source ${WORK_DIR}/0SOURCE
cd ${run_DIR}
mkdir -p t_${i}
cd t_${i}
ln -s ../blat.${DB_REFSEQ}.${QUERYNAME}.Reps.psl .

#-----
#1. Generate filtered blat hits table for contigs that contain gaps in query (insertion) or target (deletion).
#
tail -n+6 blat.${DB_REFSEQ}.${QUERYNAME}.Reps.psl |\
 awk -v gap_minnum="${GAP_MINNUM}" '{ OFS=\"\\t\"; if ( \$5 >= gap_minnum || \$7 >= gap_minnum ) print \$0 }'\
 > blat.${DB_REFSEQ}.${QUERYNAME}.psl.gaps

#-----
#2. Calculate the Q_gapStarts, Q_gapSizes, T_gapStarts and T_gapSizes based on each gap using the block info from the Blat result.
#
# make a table with these info:
# Q_name(#10), T_name(#14), Q_gap_count(#5), Q_gap_bases(#6), T_gap_count(#7), T_gap_bases(#8),
# Q_size(#11), Q_start(#12), Q_end(#13), T_size(#15), T_start(#16), T_end(#17),
# block_count(#18), blockSizes(#19), qStarts(#20), tStarts(#21)
#
awk '{ print \$10,\$14,\$5,\$6,\$7,\$8,\$11,\$12,\$13,\$15,\$16,\$17,\$18,\$19,\$20,\$21 }' blat.${DB_REFSEQ}.${QUERYNAME}.psl.gaps\
 | sort -k 2 -t \" \" > tmp && \mv tmp blat.${DB_REFSEQ}.${QUERYNAME}.psl.gaps
echo \"Q_name T_name Q_gap_count Q_gap_bases T_gap_count T_gap_bases \
Q_size Q_start Q_end T_size T_start T_end block_count blockSizes qStarts tStarts\" |\
 cat - blat.${DB_REFSEQ}.${QUERYNAME}.psl.gaps > blat.${DB_REFSEQ}.${QUERYNAME}.psl.Gaps

# now generate psl.perGap records from each psl.gap record
\rm blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap
len=\`awk 'END { print NR }' blat.${DB_REFSEQ}.${QUERYNAME}.psl.gaps\`
for ((k=1; k<=\$len; k++))
do
head -\$k blat.${DB_REFSEQ}.${QUERYNAME}.psl.gaps | tail -1 > rec
cut -d \" \" -f1 rec > rec.Qname
cut -d \" \" -f2 rec > rec.Tname
cut -d \" \" -f3 rec > rec.QgapCount
cut -d \" \" -f4 rec > rec.QgapBases
cut -d \" \" -f5 rec > rec.TgapCount
cut -d \" \" -f6 rec > rec.TgapBases

cut -d \" \" -f7 rec > rec.Qsize
cut -d \" \" -f8 rec > rec.Qstart
cut -d \" \" -f9 rec > rec.Qend
cut -d \" \" -f10 rec > rec.Tsize
cut -d \" \" -f11 rec > rec.Tstart
cut -d \" \" -f12 rec > rec.Tend

cut -d \" \" -f13 rec > rec.blockCount
cut -d \" \" -f14 rec > rec.blockSizes
cut -d \" \" -f15 rec > rec.qStarts
cut -d \" \" -f16 rec > rec.tStarts

blockCount=\`cat rec.blockCount\`
\rm rec.perGap
for ((m=1; m<\$blockCount; m++))
do
blockSize=\`cut -d\",\" -f \$m rec.blockSizes\`

Qname=\`cat rec.Qname\`
Tname=\`cat rec.Tname\`
QgapCount=\`cat rec.QgapCount\`
QgapBases=\`cat rec.QgapBases\`
qStart=\`cut -d\",\" -f \$m rec.qStarts\`
qEnd=\`expr \$qStart + \$blockSize\`

TgapCount=\`cat rec.TgapCount\`
TgapBases=\`cat rec.TgapBases\`
tStart=\`cut -d\",\" -f \$m rec.tStarts\`
tEnd=\`expr \$tStart + \$blockSize\`

next=\`expr \$m + 1\`
qStartNext=\`cut -d\",\" -f \$next rec.qStarts\`
qSizeNext=\`cut -d\",\" -f \$next rec.blockSizes\`
qEndNext=\`expr \$qStartNext + \$qSizeNext\`
tStartNext=\`cut -d\",\" -f \$next rec.tStarts\`
tSizeNext=\`cut -d\",\" -f \$next rec.blockSizes\`
tEndNext=\`expr \$tStartNext + \$tSizeNext\`

qGapSize=\`expr \$qStartNext - \$qEnd\`
qGapStart=\$qEnd
qGapEnd=\$qStartNext

tGapSize=\`expr \$tStartNext - \$tEnd\`
tGapStart=\$tEnd
tGapEnd=\$tStartNext

echo \"\$Qname \$Tname \$QgapCount \$QgapBases \$TgapCount \$TgapBases \$blockCount \$qStart \$qEnd \$qGapSize \$qStartNext \$qEndNext \$tStart \$tEnd \$tGapSize \$tStartNext \$tEndNext\" >> rec.perGap
done

cat rec.perGap >> blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap
done

" > x2-WGindelT-perGap-${i}
done


## Submit 
cd ${run_DIR}
for i in ${SAMPLE}
do
sh x2-WGindelT-perGap-${i} &
done

# Wait for all jobs are finished
wait

##-----------
#3. Generate indel tables for insertion and deletion based on the filter of GAP_MINSIZE and GAP_MAXSIZE 
## Using locus10036 as a typical perGap record example
# Qname Tname QgapCount QgapBases TgapCount TgapBases blockCount qStart qEnd qGapSize qStartNext qEndNext tStart tEnd tGapSize tStartNext tEndNext
# 1125885 locus10036 4 13 4 164 8 0 135 4 139 232 380 515 0 515 608
# 1125885 locus10036 4 13 4 164 8 139 232 6 238 392 515 608 1 609 763
# 1125885 locus10036 4 13 4 164 8 238 392 0 392 401 609 763 154 917 926
# 1125885 locus10036 4 13 4 164 8 392 401 0 401 425 917 926 7 933 957
# 1125885 locus10036 4 13 4 164 8 401 425 1 426 430 933 957 0 957 961
# 1125885 locus10036 4 13 4 164 8 426 430 2 432 513 957 961 0 961 1042
# 1125885 locus10036 4 13 4 164 8 432 513 0 513 2499 961 1042 2 1044 3030

# There are total of 8 blocks (blockCount=8), resulted from 4 insertions (QgapCount=4) and 4 deletions (TgapCount=4).
# The gaps due to insertions can be inferred from qGapSize of 4, 6, 1, 2 (non-zeros), sum up to 13 bp (QgapBases=13).
# The gaps due to deletions can be inferred from tGapSize of 1, 154, 7, 2 (none-zeros), sum up to 164 bp (TgapBases=164).
#
cd ${run_DIR}
for i in ${SAMPLE}
do
QUERY=${i}-${SEQ}
QUERYNAME=${i}.${SEQNAME}
\cp t_${i}/blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap . 

awk -v gap_minsize="${GAP_MINSIZE}" -v gap_maxsize="${GAP_MAXSIZE}" '{ OFS="\t"; if ( $15 >= gap_minsize && $15 <= gap_maxsize ) print $0 }' blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap | sort -t " " -k2 | uniq > blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap.t
echo "Qname Tname QgapCount QgapBases TgapCount TgapBases blockCount qStart qEnd qGapSize qStartNext qEndNext tStart tEnd tGapSize tStartNext tEndNext" | cat - blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap.t > blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap.T

awk -v gap_minsize="${GAP_MINSIZE}" -v gap_maxsize="${GAP_MAXSIZE}" '{ OFS="\t"; if ( $10 >= gap_minsize && $10 <= gap_maxsize ) print $0 }' blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap | sort -t " " -k2 | uniq > blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap.q
echo "Qname Tname QgapCount QgapBases TgapCount TgapBases blockCount qStart qEnd qGapSize qStartNext qEndNext tStart tEnd tGapSize tStartNext tEndNext" | cat - blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap.q > blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap.Q

done

##-----------
#4. Make a indel summary table that combines same indels from all samples in each row based on the indel features, with the following columns:
# Tname TgapStart TgapSize Qname qGapStart qGapSize Samples, such as:
# locus12345 1114 0 1357246 2476 500 YB1;YB2;YB4;YB5;YB6;YB8;
# Note "q" (query) is for insertion, and "t" (target) is for deletion.

for indel in q t
do
cd ${run_DIR}
mkdir -p ${indel}
cd ${indel}
\rm -r *.sample
for i in ${SAMPLE}
do
QUERY=${i}-${SEQ}
QUERYNAME=${i}.${SEQNAME}
cp ${run_DIR}/blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap.${indel} .

len=`awk 'END { print NR }' blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap.${indel}`
for ((k=1; k<=$len; k++))
do
head -$k blat.${DB_REFSEQ}.${QUERYNAME}.psl.perGap.${indel} | tail -1 > rec

cut -d " " -f1 rec > rec.Qname
cut -d " " -f2 rec > rec.Tname
cut -d " " -f9 rec > rec.qEnd
cut -d " " -f10 rec > rec.qGapSize
cut -d " " -f14 rec > rec.tEnd
cut -d " " -f15 rec > rec.tGapSize

Qname=`cat rec.Qname`
Tname=`cat rec.Tname`
qEnd=`cat rec.qEnd`
qGapStart=$qEnd
qGapSize=`cat rec.qGapSize`
tEnd=`cat rec.tEnd`
tGapStart=$tEnd
tGapSize=`cat rec.tGapSize`

echo "$i" >> $Tname.$tGapStart.$tGapSize.$Qname.$qGapStart.$qGapSize.sample
done
done

#5. MAKE A summary indel table by getting all file names of $Tname.$tGapStart.$tGapSize.$Qname.$qGapStart.$qGapSize, 
# then making the newlines in each above file into ";", so
# YB1
# YB2
# YB3
# would become 
# YB1;YB2;YB3

ls *.sample > iloci.list
for x in $(cat iloci.list)
do
awk 1 ORS=';' $x > tmp && \mv tmp $x
done

# Aggregate the above file contents about indel-sample into a table
grep '' /dev/null *.sample > iloci.table
# convert . and : into tab
sed 's/\./\t/g' iloci.table | sed 's/\:/\t/g' | \
awk '{ FS="\t"; print $1,$2,$3,$4,$5,$6,$8 }' | sort -k1.6 -n > tmp && \mv tmp iloci.table


##---------------------------
#6. Use the summary indel table iloci.table (without geneID to avoid generating duplicates) to retrieve fasta sequences.
\cp ${bin_DIR}/getTarget.py ${run_DIR}/${indel}/.
\cp iloci.table ILOCI.LIST
\cp iloci.table Seq.${indel}_Lg${GAP_MINSIZE}.table
echo "Tname tGapStart tGapSize Qname qGapStart qGapSize Samples" | cat - Seq.${indel}_Lg${GAP_MINSIZE}.table > tmp && \
\mv tmp Seq.${indel}_Lg${GAP_MINSIZE}.table

for i in ${SAMPLE}
do
grep $i ILOCI.LIST > ILOCI.LIST_${i}
len=`awk 'END { print NR }' ILOCI.LIST_${i}`
\rm Seq.${indel}_Lg${GAP_MINSIZE}_${i}.fasta

for ((k=1; k<=$len; k++))
do
head -$k ILOCI.LIST_${i} | tail -1 > rec
cut -d " " -f1 rec > rec.Tname
cut -d " " -f2 rec > rec.tGapStart
cut -d " " -f3 rec > rec.tGapSize
cut -d " " -f4 rec > rec.Qname
cut -d " " -f5 rec > rec.qGapStart
cut -d " " -f6 rec > rec.qGapSize
cut -d " " -f7 rec > rec.Samples

tGapStart=`cat rec.tGapStart`
tGapSize=`cat rec.tGapSize`
tGapEnd=`expr $tGapStart + $tGapSize`
qGapStart=`cat rec.qGapStart`
qGapSize=`cat rec.qGapSize`
qGapEnd=`expr $qGapStart + $qGapSize`
Samples=`cat rec.Samples`

if [ ${indel} == "q" ]; then 
    query=`cat rec.Qname` 
    blastdbcmd -db ${prereq_DIR}/DB_${i}_contig -entry "lcl|${query}" -dbtype nucl -outfmt "%f" -out ${i}-${query}.fa
    sed -i -e "s/>lcl|/>/" ${i}-${query}.fa
    python getTarget.py ${i}-${query}.fa ${i}-${query}_GAP_${qGapStart}_${qGapEnd} ${qGapStart} ${qGapEnd}
    sed -i "s/^>${query}/>${i}-${query}_GAP_${qGapStart}_${qGapEnd}/" ${i}-${query}_GAP_${qGapStart}_${qGapEnd}.fasta
    cat ${i}-${query}_GAP_${qGapStart}_${qGapEnd}.fasta >> Seq.${indel}_Lg${GAP_MINSIZE}_${i}.fasta
elif [ ${indel} == "t" ]; then 
    query=`cat rec.Tname` 
    python getTarget.py ${prereq_DIR}/${query}.fa ${i}-${query}_GAP_${tGapStart}_${tGapEnd} ${tGapStart} ${tGapEnd}
    sed -i "s/^>${query}/>${i}-${query}_GAP_${tGapStart}_${tGapEnd}/" ${i}-${query}_GAP_${tGapStart}_${tGapEnd}.fasta
    cat ${i}-${query}_GAP_${tGapStart}_${tGapEnd}.fasta >> Seq.${indel}_Lg${GAP_MINSIZE}_${i}.fasta
else 
    echo "This is neither insertion nor deletion" >> err_indel
fi

done
done

done

##---------------------------
#7. Report
cd ${data_DIR}
for indel in q t
do
\cp ${run_DIR}/${indel}/Seq*_Lg* .
done

rename Seq.q SeqIns Seq.q*
rename Seq.t SeqDel Seq.t*

