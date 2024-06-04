#! /bin/bash
#
#PBS -r n

##Job settings
#PBS -N statsJobGOF
#PBS -o outstatsJobGOF
#PBS -e errstatsJobGOF
#PBS -m a
#PBS -M jhugon@phys.ufl.edu

#Multiple Job Submission:
#Jobs will have a variable called $PBS_ARRAYID
#that will be one of the following numbers
#PBS -t 1-YAYYAYYAY

##Job Resources
#PBS -l walltime=01:35:00
#PBS -l nodes=1:ppn=1
#PBS -l pmem=2500mb

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

ifile="1"
for f in `ls *.txt`; do
  if [ "$ifile" -eq "$PBS_ARRAYID" ]; then
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
DIRNAME="Dir"$FILENAME"Dir"
ROOTFILENAME=${FILENAME%$TXTSUFFIX}.root

mkdir $DIRNAME
cp $FILENAME $DIRNAME/
cp $ROOTFILENAME $DIRNAME/
cd $DIRNAME

# Takes 30 min of UFTrig01 w/ 1 processor for 
#   BDTCutCatVBFBDTOnly_7P8TeV_125.0 1000 toys

STARTTIME=`date +%s`
echo "running runGOF.sh"
date

iSeed="0"
while true; do
  combine -M GoodnessOfFit --algorithm saturated $FILENAME -t 100 -s $(( 124389 + $iSeed )) >> logGOF
  iSeed=$(( $iSeed + 1 ))
  if [ "$iSeed" -gt 10 ]; then
    break
  fi
done

hadd -f -k $FILENAME.GOF-Toys.root higgsCombineTest.GoodnessOfFit.*.root

rm -f roostats*
rm -f higgsCombineTest*.root

combine -M GoodnessOfFit --algorithm saturated $FILENAME >> logGOF
mv higgsCombineTest.GoodnessOfFit.*.root $FILENAME.GOF.root

rm -f roostats*
rm -f higgsCombineTest*.root

cp *GOF*.root ..

date
echo "done"

ENDTIME=`date +%s`
echo "Took $(( $ENDTIME - $STARTTIME )) seconds"
