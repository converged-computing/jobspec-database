#!/bin/bash
#PBS -q atlasq
#PBS -N nessy1
#PBS -M cernetic@mps.mpg.de
#PBS -j oe
#PBS -l nodes=1:ppn=48
#PBS -l walltime=1:00:00

# name(s) of module(s) to load
MODULES=mvapich2_pgi

# initialize modules environment
. $MODULESHOME/init/bash

# load module(s)
module load $MODULES
module load intel

# change to submit dir
cd $PBS_O_WORKDIR

# number of processes
NP=$(cat $PBS_NODEFILE | wc -l)

# number of processes per node
PPN=$(($NP/$(cat $PBS_NODEFILE|uniq|wc -l)))

# set number of hardware contexts
export PSM_SHAREDCONTEXTS_MAX=$(((($PPN+3))/4))

# start your job
	currentFile="4a"
	subBinsFile="      reduced='.r"
	subBinsFile+=$currentFile"'"
	index=859
	 `sed -i "${index}s/.*/${subBinsFile}/" ../COSI_new/cosi/inibl0.for`
	export TERM=xterm
	/usr/bin/make 64 -C ../COSI_new/cosi/ 
	cp ../COSI_new/out-intel-x86_64/fioss8.exe ./
	export PATH=/scratch/cernetic/nessyH1/scripts/:$PATH
	export COSI_BIN=/scratch/cernetic/nessyH1/scripts/
	export MAX_PROC=48
	fioss_do2 1010 10 8980 > nohup
# unload module(s)
module unload $MODULES

spectraFilename="spectra"$currentFile"sub"
python /home/cernetic/Documents/sorting/lopa-sorting/mergeMdisp.py mdisp/ /home/cernetic/Desktop/$spectraFilename
rm -rf ctrl-* JOB-* freq*log mdisp* lopa*
python /home/cernetic/Documents/sorting/lopa-sorting/mdispCompare.py /home/cernetic/Documents/sorting/lopa-sorting/bins /home/cernetic/Desktop/$spectraFilename /home/cernetic/Desktop/spectra
