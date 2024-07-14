#!/bin/bash

##
## @file run_prun.sh
##
## Description of the script file
##
## @author Jakub Cuth <Jakub.Cuth@cern.ch>
## @date 2016-04-26


PRUN(){

    jobname=$1
    shift
    njobs=$1
    shift
    arguments=$*

    if [[ $target =~ compile ]]
    then

        echo NOT TESTED $target && exit 6
 prun \
 --osMatching \
 --bexec "chmod 777 compile_grid.sh; ./compile_grid.sh  $DYTURBOVERSION;" \
 --exec "chmod 777 run_grid.sh; ./run_grid.sh %RNDM:0 ${arguments} ;" \
 --extFile dyturbo-${DYTURBOVERSION}.tar.gz \
 --nJobs $njobs \
 --outDS ${CERNGROUP}.${CERNUSER}.${jobname}_${gridv}/ \
 --outputs results.root,results.txt \
 --tmpDir /tmp/${CERNUSER} \
 --nGBPerJob=MAX \
 --destSE=ANALY_CERN_SLC6 \
 --rootVer=$ROOTVERSION --cmtConfig=$CMTVERSION 

    else

prun \
--osMatching \
--exec "chmod 777 run_grid.sh; ./run_grid.sh %RNDM:0 ${arguments} ;" \
--nJobs $njobs \
--maxFileSize=25000000 \
--outDS ${CERNGROUP}.${CERNUSER}.${jobname}_${gridv}/ \
--outputs results.root,results.txt \
--noCompile \
--tmpDir /tmp/${CERNUSER} \
--nGBPerJob=MAX \
--destSE=ANALY_CERN_SLC6 \
--rootVer=$ROOTVERSION --cmtConfig=$CMTVERSION $OFFICIAL

    fi
}


