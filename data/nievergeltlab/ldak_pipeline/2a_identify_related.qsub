#!/bin/bash
#PBS -V

#Calculate GRMs in LDAK, just for the relatedness analysis

while getopts l:e:p:q:w:r:n:g:u:o:x:z: option
do
  case "${option}"
    in
      e) ldak=${OPTARG};; #LDAK binary location (full path)
      l) ldakmodel=${OPTARG};; #Use LDAK model or GCTA model? if using GCTA, no weights are used. I don't recommend using LDAK model with admixed people. 
      p) power=${OPTARG};; #Set power for LDAK GRM calculation
      q) pruning=${OPTARG};; #LD prune data prior to making GRM
      w) windowkb=${OPTARG};; #Window size for LD pruning
      r) r2=${OPTARG};; #r2 for LD pruning
      n) sname=${OPTARG};; #Study name
      g) genodir=${OPTARG};; #Directory of genotypes used in GRM calculation
      u) useweights=${OPTARG};; #Use weights in computation of GRM
      o) outdir=${OPTARG};; #Output folder location
      x) lisa=${OPTARG};;
      z) snplist=${OPTARG};;
    esac
done

 if [ $lisa != "yes" ]
 then
  TMPDIR=$lisa
 fi
 
 if [ $TMPDIR == "" ]
 then 
  exit 110 #FAIL if tmpdir not set!"
 fi
   
#Set chromosome
chr=$PBS_ARRAYID

#IF GCTA model specified, weights will be ignored and power will be -1
if [ $ldakmodel == "gcta" ]
then
 power=-1
 param="--ignore-weights YES"
fi

if [ $useweights == "no" ]
then
 param="--ignore-weights YES"
fi

 if [ $snplist != xxxx ]
 then
  echo "Including SNPlist" $snplist
  extract="--extract "$snplist"_"$chr".snplist"
 fi

cp "$genodir"/"$sname"_"$chr".* "$TMPDIR"/.
 
#Prune each chromosome. Make kinship based on pruned chromosomes. 
#Notice that I am performing calc kinship direct - that is to say rather than calculating kinship on a split up dataset, which requires weightings
#I do it on the whole chr at once

cd $TMPDIR

if [ $pruning != "xxxx" ]
then
 $ldak --thin prune_"$chr" --bfile "$sname"_"$chr"  --window-kb $windowkb --window-prune $r2 $extract
 prune_flag="--extract prune_"$chr".in"
 echo "pruning to markers in $pruning"
fi

mkdir kins
 $ldak --calc-kins-direct kins/"$sname"_"$chr" --bfile "$sname"_"$chr" $prune_flag $param --power $power
cp -r kins/* $outdir/.

 if [ $lisa != "yes" ]
 then
  rm "$TMPDIR"/*
  rm -f $TMPDIR/kins
 fi
