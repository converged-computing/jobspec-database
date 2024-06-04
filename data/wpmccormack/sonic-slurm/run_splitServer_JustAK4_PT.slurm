#!/bin/bash
#SBATCH --job-name=run_splitServer_JustAK4_PT      # Job name
##SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=email@ufl.edu    # Where to send mail 
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task    
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
##SBATCH --mem=1gb                    # Job memory request
#SBATCH --time=05:00:00              # Time limit hrs:min:sec
##SBATCH --array=0-$3
#SBATCH --output=splitServer_JustAK4_PT_arrays/out_%A_%a.log     # Standard output and error log
##SBATCH --nodelist=cvmfs-test-compute-0-7,cvmfs-test-compute-0-8,cvmfs-test-compute-0-9,cvmfs-test-compute-0-10
#SBATCH --exclusive

hostname
echo $1
echo $2

mkdir /tmp/dir_${SLURM_JOB_ID}_${SLURM_PROCID}
cd /tmp/dir_${SLURM_JOB_ID}_${SLURM_PROCID}
pwd

current_epoch=$(date +%s)
echo $current_epoch
target_epoch=$(date -d "$3" +%s)
echo $target_epoch
sleep_seconds=$(( $target_epoch - $current_epoch ))
echo $sleep_seconds
sleep $sleep_seconds

export SCRAM_ARCH=slc7_amd64_gcc900
source /cvmfs/cms.cern.ch/cmsset_default.sh
CMS_PATH=~/cms_path
ll $CMS_PATH/SITECONF/local/JobConfig/site-local-config.xml
scram p CMSSW_11_3_0
cd CMSSW_11_3_0
eval `scramv1 runtime -sh`
cd ../
mkdir test_REAL
cd test_REAL
pwd
cp ~/CMSSW_12_0_0_pre5_test_splitServer_4000_3.tgz .
tar -xf CMSSW_12_0_0_pre5_test_splitServer_4000_3.tgz
cd CMSSW_12_0_0_pre5/src/sonic-workflows
cp ~/allSonicTriton_cff_JustAK4.py ../Configuration/ProcessModifiers/python/allSonicTriton_cff.py
cp ~/pfParticleNet_NOAK8_cff.py ../RecoBTag/ONNXRuntime/python/pfParticleNet_cff.py
scramv1 b -r ProjectRename # this handles linking the already compiled code - do NOT recompile
eval `scramv1 runtime -sh`
ln -s /home/wmccorma_mit_edu/local_files/*.root .
#cp ~/run_files_splitServer.py .
#cp ~/step2_PAT_files.py .
cp ~/file_of_files_2.txt ./file_of_files.txt
#cmsRun run_files_splitServer.py address="34.135.193.168","35.193.173.208","35.184.63.191" port=$2 tmi=True threads=4 maxEvents=8
cmsRun run_files_splitServer.py address=$1 port=$2 tmi=True threads=4 modifiers="particleNetPTSonicTriton" maxEvents=8

current_epoch2=$(date +%s)
echo $current_epoch2
target_epoch2=$(date -d "$3" +%s)
extrasecs=300
target_epoch2=$(( $target_epoch2 + $extrasecs ))
echo $target_epoch2
sleep_seconds2=$(( $target_epoch2 - $current_epoch2 ))
echo $sleep_seconds2
sleep $sleep_seconds2

#cmsRun run_files.py address="$1" port=$2 tmi=True modifiers="particleNetPTSonicTriton" threads=4 maxEvents=1000 &> outThroughput.txt
#cmsRun run_files_splitServer.py address="34.135.193.168","35.193.173.208","35.184.63.191" port=$2 tmi=True  threads=4 maxEvents=1000 &> outThroughput.txt
cmsRun run_files_splitServer.py address=$1 port=$2 tmi=True  threads=4 modifiers="particleNetPTSonicTriton" maxEvents=1000 &> outThroughput.txt
#python3.6 ~/throughputFinder.py outThroughput.txt ~/ALL_arrays/Synchro${SLURM_ARRAY_JOB_ID}.txt
python3.6 ~/throughputFinderTimeDiff.py outThroughput.txt ~/splitServer_JustAK4_PT_arrays/Synchro${SLURM_ARRAY_JOB_ID}.txt
cd /tmp
cd /tmp
rm -rf dir_${SLURM_JOB_ID}_${SLURM_PROCID}
