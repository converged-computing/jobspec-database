#! /bin/bash

set -e

DBG=
PROJECT="NTI104"        #defined by the user

if [ $2 -gt 1024 ]; then
REQ_WALLTIME="00:30:00"
else
REQ_WALLTIME="00:20:00"
fi

SUBMIT_ARGS=""
WAIT="0"

while true; do
  case "$1" in
    -h|--help )
      echo "Usage: $0 [options] [.rea stem]"
      echo
      echo "Usable options:"
      echo "-h --help: Get help"
      echo "-d --debug: Run in debugger"
      echo "-n|--nodes N: Set number of nodes to N"
      echo "-t NN:NN:NN: requested amount of computer time"
      echo "-s|--submit-args \"-arg1 -arg2\" extra arguments to qsub"
      echo "-w|--wait wait until job is completed"
      exit 1
      shift
      ;;
    -d|--debug )
      echo "*** running in debugger"
      DBG="gdb"
      shift
      ;;
    -n|-nodes|--nodes )
      shift
      CORECOUNT="$1"
      shift
      ;;
    -t )
      shift
      REQ_WALLTIME="$1"
      shift
      ;;
    -s|--submit-args )
      shift
      SUBMIT_ARGS="$1"
      shift
      ;;
    -w|--wait )
      shift
      WAIT="1"
      ;;
    * )
      break
      ;;
  esac
done

rm -f xxt_map.rea

# automatically find .rea file, if unique

if test "$1" = ""; then
  COUNTREA=`ls *.rea | wc -l`
  if test $COUNTREA = 1; then
    REAFILE=`ls *.rea`
    echo "*** found only $REAFILE, picking that one"
  else
    echo "Must specify .rea file; there is more than one here:"
    ls *.rea | cat 
    exit 1
  fi
else
  REAFILE=$1
  CORECOUNT=$2
fi

SESSION=${REAFILE%.[rR][eE][aA]}
echo "*** running session $SESSION"

mvifthere()
{
  if test -f "$1"; then
    mv "$1" "$2"
  fi
}


cpifthere()
{
  if test -f "$1"; then
    cp "$1" "$2"
  fi
}

echo $SESSION >  SESSION.NAME
echo `pwd`'/' >>  SESSION.NAME

cpifthere $SESSION.gph nekgraph
cpifthere $SESSION.map xxt_map.rea
mvifthere $SESSION.log $SESSION.log1
mvifthere $SESSION.fld $SESSION.fld1
mvifthere $SESSION.his $SESSION.his1
mvifthere $SESSION.sch $SESSION.sch1
mvifthere $SESSION.out $SESSION.out1
mvifthere $SESSION.ore $SESSION.ore1
mvifthere $SESSION.nre $SESSION.nre1

# this is for cray system
if    test -d /lustre  ; then 

        rm -f logfile
        rm -f $SESSION.np=$CORECOUNT.output
        OUTFILE="`pwd`/$SESSION.np=$CORECOUNT-cray-`date "+%F_%H_%M_%S"`"
        touch $SESSION.rea
        touch $OUTFILE.output
        ln $OUTFILE.output $SESSION.np=$CORECOUNT.output
        ln $OUTFILE.output logfile

        if ! pbsfile=`mktemp $SESSION.pbs.XXXXXX` ; then
           echo "Failed to create temp file for qsub! Exiting"
           exit 1
        fi
        chmod 777 $pbsfile

        echo "#!/bin/bash -l" >> $pbsfile
        echo "#PBS -A $PROJECT" >> $pbsfile
        echo "#PBS -N $SESSION" >> $pbsfile
        echo "#PBS -o $PWD/$SESSION.np=$CORECOUNT-cray-`date "+%F_%H_%M_%S"`.output" >> $pbsfile
        echo "#PBS -e $PWD/$SESSION.np=$CORECOUNT-cray-`date "+%F_%H_%M_%S"`.error" >> $pbsfile
        echo "#PBS -l walltime=$REQ_WALLTIME,nodes=$3" >> $pbsfile
        echo "#PBS -j oe" >> $pbsfile
        echo " cd `pwd`">> $pbsfile
        #echo " export ATP_ENABLED=1" >> $pbsfile
        echo " export MPICH_RDMA_ENABLED_CUDA=1" >> $pbsfile
        echo " export CRAY_ACC_NO_ASYNC=0" >> $pbsfile #cray compiler specific
        echo " export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH" >> $pbsfile
        echo " aprun -n $CORECOUNT -N 1 ./nek5000 ">> $pbsfile
        #export CUDA_PROFILE=1

        qsub -V $pbsfile
        rm $pbsfile
        echo "job submitted on OLCF Cray, #GPUs=$CORECOUNT, #Cray_nodes=$3"
        qstat -a|grep $USER

# this is on a desktop   
else

  USERNAME="`basename $HOME`"
  OUTDIR="./vtk"
  OUTFILE="$SESSION"

  if ! mkdir -p $OUTDIR &> /dev/null ; then
    OUTDIR="./vtk"
    mkdir -p $OUTDIR
  fi

  if test "$2" = ""; then
    echo "This is to run with MPI: must specify np# $2"
    exit 1
  fi

  echo "Job to be submitted with np=$2 "
  mpiexec -np $2 $DBG ./nek5000 $OUTFILE > $OUTFILE.np=$2.output  

fi
