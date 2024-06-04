#!/bin/bash
#
#PBS -r n

##Job settings
#PBS -N FitBias
#PBS -o joboutFitBiasJob
#PBS -e joberrFitBiasJob

#Multiple Job Submission:
#Jobs will have an environmental variable called $PBS_ARRAYID
#that will be one of the following numbers

# This controls what job group numbers are used

##PBS -t 0-140
#PBS -t 0-1400

##Job Resources
#PBS -l walltime=1:00:00
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

categories01Tight="Jets01PassPtG10BB Jets01PassPtG10BO Jets01PassPtG10BE Jets01PassPtG10OO Jets01PassPtG10OE Jets01PassPtG10EE"
categories01Loose="Jets01FailPtG10BB Jets01FailPtG10BO Jets01FailPtG10BE Jets01FailPtG10OO Jets01FailPtG10OE Jets01FailPtG10EE"
categories2="Jet2CutsVBFPass Jet2CutsGFPass Jet2CutsFailVBFGF"
categoriesAll=$categories01Tight" "$categories01Loose" "$categories2  # 15 total
categoriesImportant="Jets01PassPtG10BB Jets01PassPtG10BO Jets01PassPtG10BE Jet2CutsVBFPass Jet2CutsGFPass Jet2CutsFailVBFGF"
categoriesImportant01="Jets01PassPtG10BB Jets01PassPtG10BO Jets01PassPtG10BE"

#categoriesToRun=$categoriesImportant
categoriesToRun=$categoriesAll

nCats=$(echo $categoriesToRun | wc -w)
#echo $nCats" "$categoriesToRun

categoriesToRun=($categoriesToRun)

#for iCat in $(seq 0 $(($nCats-1))); do
#  cat=${categoriesToRun[iCat]}
#  echo $iCat" "$cat
#done

grp=$(echo "10000 + $PBS_ARRAYID / $nCats" | bc)
iCat=$(echo "$PBS_ARRAYID % $nCats" | bc)
cat=${categoriesToRun[iCat]}

echo "Number of categories: "$nCats
echo "Category number to run is: "$iCat
echo "Category to run is: "$cat
echo "Group number is: "$grp

echo `date`

command="./fitBiasStudy.py $grp $cat"
logname="JOBS/logFitBias_"$cat"_"$grp"_"$PBS_ARRAYID
echo "Running '$command'"
echo "Output redirected to '$logname'"

$command >& $logname

echo `date`
endtime=$(date +%s)
echo "Job took "$((endtime - starttime))" seconds"
echo "Job took "$(echo "scale = 2; ($endtime - $starttime)/60." | bc)" minutes"
echo "Job took "$(echo "scale = 2; ($endtime - $starttime)/3600." | bc)" hours"
echo "Done!"



