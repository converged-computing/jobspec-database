#! /bin/bash
#
#PBS -r n

##Job settings
#PBS -N MakeCards
#PBS -o joboutMakeCardsJob
#PBS -e joberrMakeCardsJob

#Multiple Job Submission:
#Jobs will have an environmental variable called $PBS_ARRAYID
#that will be one of the following numbers

# This controls what higgs masses are used

#PBS -t 120-150
##PBS -t 125

##Job Resources
#PBS -l walltime=0:30:00
#PBS -l nodes=1:ppn=1
#PBS -l pmem=3000mb

starttime=$(date +%s)

# initialize environment for worker
export SCRAM_ARCH=slc5_amd64_gcc472
export OSG_APP=/osg/app
export VO_CMS_SW_DIR=${OSG_APP}/cmssoft/cms
export CMS_PATH=${VO_CMS_SW_DIR}
. ${CMS_PATH}/cmsset_default.sh;

echo "Job running on `hostname` at `date`"
echo "ArrayId: $PBS_ARRAYID"

CMSSW_DIR=/scratch/osg/jhugon/hmumu/stats/CMSSW_6_1_1
cd $CMSSW_DIR
echo "Running cmsenv in dir: `pwd`"
eval `scram runtime -sh` #this means cmsenv in a way that works in a script

DIR=/scratch/osg/jhugon/hmumu/stats/CMSSW_6_1_1/finalAnalysis

# enter working area
cd $DIR
echo "In Dir:"
pwd

#####################################
#####Begin Real Job##################
#####################################

echo `date`

command="./makeCards.py -m $PBS_ARRAYID"
logname="JOBS/logMakeCards$PBS_ARRAYID"
echo "Running '$command'"
echo "Output redirected to '$logname'"

$command >& $logname

echo `date`
endtime=$(date +%s)
echo "Job took "$((endtime - starttime))" seconds"
echo "Job took "$(echo "scale = 2; ($endtime - $starttime)/60." | bc)" minutes"
echo "Job took "$(echo "scale = 2; ($endtime - $starttime)/3600." | bc)" hours"
echo "Done!"


