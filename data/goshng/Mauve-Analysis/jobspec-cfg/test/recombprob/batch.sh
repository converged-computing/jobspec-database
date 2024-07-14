REPLICATE=1
for i in {1..5}; do
  RECOMBPROB=output/cornellf/3/run-analysis/recombprob-ref$i-rep$REPLICATE
  RECOMBPROBWIG=output/cornellf/3/run-analysis/recombprobwig-ref$i-rep$REPLICATE
  mkdir $RECOMBPROBWIG
  for j in {0..8}; do
    for k in {0..8}; do
      c=$((j * 9 + k + 3))
      echo -e "track type=wiggle_0\nfixedStep chrom=chr1 start=1 step=1 span=1" > $RECOMBPROBWIG/$j-$k 
      cut -f $c $RECOMBPROB >> $RECOMBPROBWIG/$j-$k 
    done
  done
done
exit

########################################################
# create a recombprob-ref5-rep1 file and others.
#
DBNAMECHOICES=(/Volumes/Elements/Documents/Projects/mauve/bacteria/cornell_sde1/CP002215.gbk \
  /Volumes/Elements/Documents/Projects/mauve/bacteria/Streptococcus_dysgalactiae_equisimilis_GGS_124_uid59103/NC_012891.gbk \
  /Volumes/Elements/Documents/Projects/mauve/bacteria/cornell_sdd/sdd.gbk \
  /Volumes/Elements/Documents/Projects/mauve/bacteria/Streptococcus_pyogenes_MGAS315_uid57911/NC_004070.gbk \
  /Volumes/Elements/Documents/Projects/mauve/bacteria/Streptococcus_pyogenes_MGAS10750_uid58575/NC_008024.gbk)
REORDERCHOICES=(1 2 3 4 5)
for j in {0..4}; do
  GBK=${DBNAMECHOICES[$j]}
  i=${REORDERCHOICES[$j]}
  perl pl/recombination-intensity1-probability.pl wiggle -xml /Users/goshng/Documents/Projects/mauve/output/cornellf/3/run-clonalorigin/output2/1/core_co.phase3.xml -xmfa2maf /Users/goshng/Documents/Projects/mauve/output/cornellf/3/data/core_alignment.maf -xmfa /Users/goshng/Documents/Projects/mauve/output/cornellf/3/data/core_alignment.xmfa -refgenome $i -gbk $GBK -ri1map /Users/goshng/Documents/Projects/mauve/output/cornellf/3/run-analysis/rimap.txt -clonaloriginsamplesize 1001 -out /Users/goshng/Documents/Projects/mauve/output/cornellf/3/run-analysis/recombprob-ref$i-rep1 -rimapdir /Users/goshng/Documents/Projects/mauve/output/cornellf/3/run-analysis/rimap-2
done
