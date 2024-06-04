#! /bin/bash
#
#PBS -r n

##Job settings
#PBS -N statsTYPE
#PBS -o joboutStatsJobTYPE
#PBS -e joberrStatsJobTYPE

#Multiple Job Submission:
#Jobs will have a variable called $PBS_ARRAYID
#that will be one of the following numbers
#PBS -t 1-NJOBS

##Job Resources
#PBS -l walltime=TIME
#PBS -l nodes=1:ppn=1
#PBS -l pmem=5000mb

# initialize environment for worker
starttime=$(date +%s)

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
for f in `ls PREFIX*.txt`; do
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
#ROOTFILENAME=${FILENAME%$TXTSUFFIX}.root
WSFILENAME=${FILENAME%$TXTSUFFIX}_ws.root

text2workspace.py -m 125 -D data_obs $FILENAME -o $WSFILENAME

mkdir $DIRNAME
cp $WSFILENAME $DIRNAME/
cd $DIRNAME

MUMAX=150
RMAX=$MUMAX
RMIN=-$MUMAX

echo "executing combine -M Asymptotic --rMax $RMAX $WSFILENAME >& $FILENAME.out"
combine -M Asymptotic --rMax $RMAX $WSFILENAME >& $FILENAME.out

echo "executing combine -M ProfileLikelihood --rMax $RMAX -d $WSFILENAME --signif >& $FILENAME.sig"
combine -M ProfileLikelihood --rMax $RMAX -d $WSFILENAME --signif >& $FILENAME.sig

#echo "executing combine -M ProfileLikelihood --rMax $RMAX -d $WSFILENAME --signif --expectSignal=1 -t -1 --toysFreq >& $FILENAME.expsig"
#
#combine -M ProfileLikelihood --rMax $RMAX -d $WSFILENAME --signif --expectSignal=1 -t -1 >& $FILENAME.expsig
##combine -M ProfileLikelihood -d $WSFILENAME --signif --expectSignal=1 -t -1 --toysFreq >& $FILENAME.expsig

echo "executing combine -M MaxLikelihoodFit --rMin $RMIN --rMax $RMAX --plots --saveNormalizations $WSFILENAME >& $FILENAME.mu"
combine -M MaxLikelihoodFit --rMin $RMIN --rMax $RMAX --plots --saveNormalizations $WSFILENAME >& $FILENAME.mu

echo "executing combine -M MultiDimFit --rMin $RMIN --rMax $RMAX --algo=singles $WSFILENAME>& $FILENAME.mu2"
combine -M MultiDimFit --rMin $RMIN --rMax $RMAX --algo=singles $WSFILENAME >& $FILENAME.mu2

#combine -M ChannelCompatibilityCheck --saveFitResult --rMin $RMIN --rMax $RMAX $WSFILENAME >> logCCC2
#mv higgsCombineTest.ChannelCompatibilityCheck.*.root ../$FILENAME.CCC2.root

cp $FILENAME.* ..
cp mlfit.root ../$FILENAME.root

date
endtime=$(date +%s)
echo "Job took "$((endtime - starttime))" seconds"
echo "Job took "$(echo "scale = 2; ($endtime - $starttime)/60." | bc)" minutes"
echo "Job took "$(echo "scale = 2; ($endtime - $starttime)/3600." | bc)" hours"
echo "done"

