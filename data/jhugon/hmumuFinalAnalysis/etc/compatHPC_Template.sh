#! /bin/bash
#
#PBS -r n

##Job settings
#PBS -N statsJobCompat
#PBS -o outstatsJobCompat
#PBS -e errstatsJobCompat
#PBS -m a
#PBS -M jhugon@phys.ufl.edu

#Multiple Job Submission:
#Jobs will have a variable called $PBS_ARRAYID
#that will be one of the following numbers
#PBS -t 0-YAYYAYYAY

### Need > 4 GB (8 Works) of memory for 7P8 TeV combination
### 3 GB is fine for all others

##Job Resources
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=1
#PBS -l pmem=8000mb

# initialize environment for worker
STARTTIME=`date +%s`

export SCRAM_ARCH=slc5_amd64_gcc462
export OSG_APP=/osg/app
export VO_CMS_SW_DIR=${OSG_APP}/cmssoft/cms
export CMS_PATH=${VO_CMS_SW_DIR}
cd $CMS_PATH
source cmsset_default.sh;

echo "Job running on `hostname` at `date`"
echo "ArrayId: $PBS_ARRAYID"

# Vars
DIR=$PBS_O_WORKDIR

# enter working area
cd $DIR
echo "In Dir:"
pwd

eval `scram runtime -sh`

#####################################
#####Begin Real Job##################
#####################################

ntoyjobs=WOWOWOWO
ntoys=50
ifiletorun=$(( $PBS_ARRAYID / $ntoyjobs ))
toysrun=$(( $PBS_ARRAYID % $ntoyjobs ))
echo "ntoyjobs: $ntoyjobs"
echo "ifiletorun: $ifiletorun"
echo "toysrun: $toysrun"

ifile="0"
for f in `ls *.txt`; do
  if [ "$ifile" -eq "$ifiletorun" ]; then
    FILENAME=$f
  fi
  ifile=$(( $ifile + 1 ))
done

echo "Input Filename: $FILENAME"

if [ -z "$FILENAME" ]; then
  echo "Error: Couldn't find filename for PBS_ARRAYID!! Exiting."
  exit "1"
fi

TXTSUFFIX=".txt"
DIRNAME="Dir"$FILENAME"DirToysRun"$toysrun
ROOTFILENAME=${FILENAME%$TXTSUFFIX}.root

mkdir $DIRNAME
cp $FILENAME $DIRNAME/
cp $ROOTFILENAME $DIRNAME/
cd $DIRNAME

STARTTIME=`date +%s`
echo "running runGOF.sh"
date

combine -M ChannelCompatibilityCheck --saveFitResult --rMax 60 --rMin -60 $FILENAME -t $ntoys -s $(( 121324 + $PBS_ARRAYID)) >> logCCCToys

mv higgsCombineTest.ChannelCompatibilityCheck.*.root ../$FILENAME.CCC-Toys-$toysrun.root

rm -f roostats*
rm -f higgsCombineTest*.root

if [ "$toysrun" -eq "0" ]; then
  echo "running the Observed compatability"
  combine -M ChannelCompatibilityCheck --saveFitResult --rMax 60 --rMin -60 $FILENAME >> logCCC
  mv higgsCombineTest.ChannelCompatibilityCheck.*.root ../$FILENAME.CCC.root

  rm -f roostats*
  rm -f higgsCombineTest*.root
fi

date
echo "done"

ENDTIME=`date +%s`
echo "Took $(( $ENDTIME - $STARTTIME )) seconds"
