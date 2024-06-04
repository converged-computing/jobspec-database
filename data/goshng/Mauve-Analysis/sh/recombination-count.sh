###############################################################################
# Copyright (C) 2011 Sang Chul Choi
#
# This file is part of Mauve Analysis.
# 
# Mauve Analysis is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Mauve Analysis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Mauve Analysis.  If not, see <http://www.gnu.org/licenses/>.
###############################################################################

function recombination-count {
  PS3="Choose the species for $FUNCNAME: "
  select SPECIES in ${SPECIESS[@]}; do
    if [ "$SPECIES" == "" ];  then
      echo -e "You need to enter something\n"
      continue
    else  
      echo -n "What repetition do you wish to run? (e.g., 1) "
      read REPETITION
      g=$REPETITION
      set-more-global-variable $SPECIES $REPETITION
      NREPLICATE=$(grep ^REPETITION${REPETITION}-CO2-NREPLICATE species/$SPECIES | cut -d":" -f2)

      NUMBER_BLOCK=$(echo `ls $DATADIR/core_alignment.xmfa.*|wc -l`)
      NUMBER_SPECIES=$(echo `grep gbk data/$SPECIES|wc -l`)
      echo -e "The number of blocks is $NUMBER_BLOCK."
      echo -e "The number of species is $NUMBER_SPECIES."
      echo "NUMBER_BLOCK and NUMBER_SAMPLE must be checked"

      echo -n "Do you wish to count recombination events across blocks (or obsiter) (y/n)? "
      read WISH
      if [ "$WISH" == "y" ]; then
        for h in $(eval echo {1..$NREPLICATE}); do
          perl pl/count-observed-recedge.pl obsiter \
            -d $RUNCLONALORIGIN/output2/${h} \
            -n $NUMBER_BLOCK \
            -out $RUNANALYSIS/obsiter-recedge-$h.txt &
        done
        wait
        echo "Check $NREPLICATE files $RUNANALYSIS/obsiter-recedge-#.txt"
      fi

      echo -n "Do you wish to count recombination events (y/n)? "
      read WISH
      if [ "$WISH" == "y" ]; then
        for h in $(eval echo {1..$NREPLICATE}); do
          perl pl/count-observed-recedge.pl obsonly \
            -d $RUNCLONALORIGIN/output2/${h} \
            -n $NUMBER_BLOCK \
            -endblockid \
            -obsonly \
            -out $RUNANALYSIS/obsonly-recedge-$h.txt &
        done
        wait
        echo "Check $NREPLICATE files $RUNANALYSIS/obsonly-recedge-#.txt"
      fi

      echo -n "Do you wish to compute the prior expected number of recombination events (y/n)? "
      read WISH
      if [ "$WISH" == "y" ]; then
        CO2REPLICATE=$(grep ^REPETITION${REPETITION}-CA1-CO2ID species/$SPECIES | cut -d":" -f2)
        h=$CO2REPLICATE
        echo -n "Do you wish to do that in the cluster (y/n)? "
        read WISH2
        if [ "$WISH2" == "y" ]; then
          echo -n "Do you wish to prepare it for the cluster (y/n)? "
          read WISH3
          if [ "$WISH3" == "y" ]; then
            recombination-count-gui
          fi
          echo -n "Do you wish to receive it for the cluster (y/n)? "
          read WISH3
          if [ "$WISH3" == "y" ]; then
            scp -qr \
              $CAC_MAUVEANALYSISDIR/output/$SPECIES/$REPETITION/run-clonalorigin/output2/priorcount \
              $RUNCLONALORIGIN/output2
          fi
        else
          PRIORCOUNTDIR=$RUNCLONALORIGIN/output2/priorcount
          mkdir $PRIORCOUNTDIR
          rm 3
          JOBIDFILE=$RUNCLONALORIGIN/rc.jobidfile
          rm -f $JOBIDFILE
          for i in $(eval echo {1..$NUMBER_BLOCK}); do
            # echo "$RUNCLONALORIGIN/output2/${h}/core_co.phase3.xml.$i"
            if [ -f "$RUNCLONALORIGIN/output2/${h}/core_co.phase3.xml.$i" ]; then
              echo "src/warggui/b/gui \
                $RUNCLONALORIGIN/output2/$h/core_co.phase3.xml.$i \
                > $PRIORCOUNTDIR/$i.txt" >> 3 
#           $GUI -b \
#             -o $RUNCLONALORIGIN/output2/${h}/core_co.phase3.xml.$i \
#             -H 3 \
#             > $PRIORCOUNTDIR/$i.txt
            else
              echo "Block: $i was not found" 1>&2
            fi
            echo -ne "  Block: $i\r";
          done 
        fi
      fi

      echo -n "Do you wish to compute heatmaps (y/n)? "
      read WISH
      if [ "$WISH" == "y" ]; then
        for h in $(eval echo {1..$NREPLICATE}); do
          PRIORCOUNTDIR=$RUNCLONALORIGIN/output2/priorcount
          perl pl/count-observed-recedge.pl heatmap \
            -d $RUNCLONALORIGIN/output2/${h} \
            -e $PRIORCOUNTDIR \
            -endblockid \
            -n $NUMBER_BLOCK \
            -out $RUNANALYSIS/heatmap-recedge-${h}.txt &
            # -s $NUMBER_SPECIES \
        done
        wait
      fi

      echo -n "Do you wish to plot a heatmap of numbers of recombination events (y/n)? "
      read WISH
      if [ "$WISH" == "y" ]; then
        # CO2REPLICATE=$(grep ^REPETITION${REPETITION}-CA1-CO2ID species/$SPECIES | cut -d":" -f2)
        # h=$CO2REPLICATE
        for h in $(eval echo {1..$NREPLICATE}); do
          CO2REPLICATE=$h
          recombination-count-plot-heatmap
          echo Check $RUNANALYSIS/heatmap-recedge-$CO2REPLICATE.R.ps
        done
      fi

      echo -n "Do you wish to count recombination events only for the SPY and SDE (y/n)? "
      read WISH
      if [ "$WISH" == "y" ]; then
        for h in $(eval echo {1..$NREPLICATE}); do
          perl pl/count-observed-recedge.pl obsonly \
            -d $RUNCLONALORIGIN/output2/${h} \
            -n $NUMBER_BLOCK \
            -endblockid \
            -lowertime 0.045556 \
            -out $RUNANALYSIS/obsonly-recedge-time-$h.txt
          echo "Check file $RUNANALYSIS/obsonly-recedge-time-$h.txt"
        done
      fi

      break
    fi
  done
}


function recombination-count-gui {
      echo "  Creating jobidfile..."
      WALLTIME=$(grep ^REPETITION${REPETITION}-CA1-WALLTIME species/$SPECIES | cut -d":" -f2)
      NUMBER_BLOCK=$(echo `ls $DATADIR/core_alignment.xmfa.*|wc -l`)
      JOBIDFILE=$RUNCLONALORIGIN/gui.jobidfile
      CCLONALORIGINDIR=output/$SPECIES/$REPETITION/run-clonalorigin
      rm -f $JOBIDFILE
      for b in $(eval echo {1..$NUMBER_BLOCK}); do
        echo "./warg \
          $CCLONALORIGINDIR/output2/$h/core_co.phase3.xml.$b \
          > $CCLONALORIGINDIR/output2/priorcount/$b.txt" >> $JOBIDFILE 
      done

      scp -q $JOBIDFILE \
          $CAC_MAUVEANALYSISDIR/output/$SPECIES/$REPETITION/run-clonalorigin
      scp -q cac/sim/batch_task_gui.sh \
          $CAC_MAUVEANALYSISDIR/output/$SPECIES/$REPETITION/run-clonalorigin/batchjob.sh
      scp -q cac/sim/run2.sh \
          $CAC_MAUVEANALYSISDIR/output/$SPECIES/$REPETITION/run-clonalorigin/run.sh

cat>$RUNCLONALORIGIN/batch.sh<<EOF
#!/bin/bash
#PBS -l walltime=${WALLTIME}:00:00,nodes=1
#PBS -A ${BATCHACCESS}
#PBS -j oe
#PBS -N $PROJECTNAME-CA1
#PBS -q ${QUEUENAME}
#PBS -m e
# #PBS -M ${BATCHEMAIL}
#PBS -t 1-PBSARRAYSIZE

WARG=\$HOME/usr/bin/warg
WARGGUI=\$HOME/usr/bin/gui

mkdir -p \$PBS_O_WORKDIR/output2/priorcount

function copy-data {
  cd \$TMPDIR
  # Copy excutables
  cp \$WARGGUI warg
  # Copy shell scripts
  cp \$PBS_O_WORKDIR/batchjob.sh .
  # Create directories
  mkdir -p $CCLONALORIGINDIR/output2/priorcount
  cp -r \$PBS_O_WORKDIR/output2/$CO2REPLICATE $CCLONALORIGINDIR/output2
  # Create a status directory
  mkdir \$PBS_O_WORKDIR/status/\$PBS_ARRAYID
}

function retrieve-data {
  cp -r $CCLONALORIGINDIR/output2/priorcount \$PBS_O_WORKDIR/output2
  # Remove the status directory.
  rm -rf \$PBS_O_WORKDIR/status/\$PBS_ARRAYID
}

function process-data {
  cd \$TMPDIR
  CORESPERNODE=8
  for (( i=1; i<=CORESPERNODE; i++))
  do
    bash batchjob.sh \\
      \$i \\
      \$PBS_O_WORKDIR/gui.jobidfile \\
      \$PBS_O_WORKDIR/gui.lockfile \\
      \$PBS_O_WORKDIR/status/\$PBS_ARRAYID \\
      PBSARRAYSIZE&
  done
}

copy-data
process-data; wait
retrieve-data
cd \$PBS_O_WORKDIR
rm -rf \$TMPDIR
EOF
      scp -q $RUNCLONALORIGIN/batch.sh \
          $CAC_MAUVEANALYSISDIR/output/$SPECIES/$REPETITION/run-clonalorigin

      echo -e "Go to $CAC_MAUVEANALYSISDIR/output/$SPECIES/$REPETITION/run-clonalorigin"
      echo -e "$ bash run.sh"
      echo -e "Choose the number of compute nodes!"
}

function recombination-count-plot-heatmap {
  NUMBERBRANCH=$(( NUMBER_SPECIES * 2 - 1))
  INDEXORDER=$(grep ^REPETITION${REPETITION}-CA1-INDEXORDER species/$SPECIES | cut -d":" -f2)
  ANAME=$(grep ^REPETITION${REPETITION}-CA1-ANAME species/$SPECIES | cut -d":" -f2)
cat>$RUNANALYSIS/heatmap-recedge-test.R<<EOF
numberBranch <- $NUMBERBRANCH
numberElement <- numberBranch * numberBranch 
A <- matrix(scan("$RUNANALYSIS/heatmap-recedge-$CO2REPLICATE.txt", n=numberElement), numberBranch, numberBranch, byrow = TRUE)
for (i in 1:numberBranch)
{
  for (j in 1:numberBranch)
  {
    if (A[i,j] == 0)
    {
      A[i,j] <- NA
    }
    else
    {
      A[i,j] <- log2(A[i,j])
    }
  }
}

B <- A
indexReorder <- c($INDEXORDER) + 1
for (i in 1:numberBranch)
{
  for (j in 1:numberBranch)
  {
    A[i,j] <- B[indexReorder[i],indexReorder[j]]
  }
}

library(colorspace)
library(gplots)

# Aname <- c("SDE1", "SDE", "SDE2", "SD", "SDD", "ROOT", "SPY1", "SPY", "SPY2")
Aname <- c($ANAME)

# b<-seq(-max(abs(A))-0.1,max(abs(A))+0.1,length.out=42)
b<-seq(-2.2,2.2,length.out=42)
postscript("$RUNANALYSIS/heatmap-recedge-$CO2REPLICATE.R.ps", width=10, height=10, horizontal = FALSE, onefile = FALSE, paper = "special")

heatmap.2(A,
  Rowv=FALSE,
  Colv=FALSE,
  dendrogram= c("none"),
  distfun = dist,
  hclustfun = hclust,
  xlab = "", ylab = "",
  key=TRUE,
  keysize=1,
  trace="none",
  density.info=c("none"),
  margins=c(10, 8),
  breaks=b,
  col=diverge_hcl(41),
  na.color="black",
  labRow=Aname,
  labCol=Aname
)
dev.off()

print (A, digits=2, width=100)

A <- matrix(scan("$RUNANALYSIS/obsonly-recedge-$CO2REPLICATE.txt", n=numberElement), numberBranch, numberBranch, byrow = TRUE)
B <- A
indexReorder <- c($INDEXORDER) + 1
for (i in 1:numberBranch)
{
  for (j in 1:numberBranch)
  {
    A[i,j] <- B[indexReorder[i],indexReorder[j]]
  }
}
print (A, digits=4, width=100)
EOF
  Rscript $RUNANALYSIS/heatmap-recedge-test.R
}
