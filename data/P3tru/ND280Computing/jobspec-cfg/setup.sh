#!/bin/bash

## source this script to setup python path to look for the nd280Computing tools and the
## ND280COMPUTINGROOT env variable.
export ND280COMPUTINGROOT=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd) ## export this as an env var
echo '$ND280COMPUTINGROOT='$ND280COMPUTINGROOT

## Set the CVS path
export CVSROOT=:ext:${USER}@repo.nd280.org:/home/trt2kmgr/T2KRepository
echo '$CVSROOT='$CVSROOT

## Python 
PYTHONPATH=$PYTHONPATH:$ND280COMPUTINGROOT/tools 
export LD_LIBRARY_PATH=$ROOTSYS/lib:$PYTHONDIR/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH

## Some GRID environment variables (also set up by ND280GRID.SetGridEnviron()) ###################
export LFC_HOST=lfc.gridpp.rl.ac.uk
export LFC_HOME=/grid/t2k.org/nd280
export GLOBUS_FTP_CLIENT_GRIDFTP2=true
export LCG_CATALOLOG_TYPE=lfc
#export FTS_SERVICE=https://lcgfts3.gridpp.rl.ac.uk:8443/services/FileTransfer
export FTS_SERVICE=https://lcgfts3.gridpp.rl.ac.uk:8446
export ND280TRANSFERS=/opt/ppd/t2k/users/stewartt/t2k_logs/transfers

if [ -n "${VAR:-x}" ]; then
    export LCG_GFAL_INFOSYS=lcg-bdii.gridpp.ac.uk:2170
fi

if [ -z "$ND280TRANSFERS" ]; then
    echo "Setting ND280TRANSFERS to $HOME"
    export ND280TRANSFERS=$HOME
fi

if [ -z "$ROOTSYS" ]; then
    echo "Setting ROOTSYS to /usr/local/root"
    export ROOTSYS=/usr/local/root
fi

if [ -z "$MYPROXY_SERVER" ]; then
    echo "Setting MYPROXY_SERVER to myproxy.gridpp.rl.ac.uk"
    export MYPROXY_SERVER=myproxy.cern.ch
fi

if [ -w /scratch/$USER ]; then
    echo "Setting scratch location to /scratch/$USER"
    export ND280SCRATCH=/scratch/$USER
fi

if [ -e /cvmfs/t2k.egi.eu ]; then
    echo "Setting VO_T2K_ORG_SW_DIR location to /cvmfs/t2k.egi.eu"
    export VO_T2K_ORG_SW_DIR=/cvmfs/t2k.egi.eu
fi

if [ -z "$ND280JOBS" ]; then
    echo "Setting ND280JOBS to $HOME"
    export ND280JOBS=$HOME
fi

if [ -z "$DN" ]; then
    echo "Searching for distinguished name in ~/.globus/* ..."
    DN=$(grep -h subject= ~/.globus/* | sort | uniq)
    export DN=\'${DN/subject=/}\'
    echo "Found DN : $DN"
fi

##################################################################################################


## Ask the info service how many t2k jobs are running on the grid
alias t2kJobQuery="lcg-info --vo t2k.org --list-ce --query 'VORunningJobs>=1 | VOWaitingJobs>=1' --attrs 'VORunningJobs,VOWaitingJobs,VOFreeJobSlots'"

## Login to CVMFS upload server
alias cvmfsUpload="gsissh -p 1975 cvmfs-upload01.gridpp.rl.ac.uk"

## Get job statuses from 'tmpjid' (list of JIDs), write ouput to 'statuses'
alias statusGet='glite-wms-job-status --noint -i tmpjid > statuses'

## Dump the 'statuses'
alias statusDump='for status in $(grep Current statuses | awk "{print \$3}" | sort | uniq); do printf "%15s %8d\n" "$status" "$(grep $status statuses | wc -l)"; done'

## Check the statuses every 5m
alias statusLoop='clear; while true; do echo; echo "Getting statuses..."; statusGet; statusDump; echo "Sleeping..."; sleep 5m; done'


## Function to make lists of LFNs given an LFC directory
makeFileList() 
{
    # given an $lfcdir e.g. 
    #    raw/ND280/ND280/00008000_00008999
    #
    # create a list of LFNs for the files therein and write to e.g. 
    #    $ND280TRANSFERS/raw.ND280.ND280.00008000_00008999.list
    #
    # useful for submitting file transfers

    # is $lfcdir defined?
    if [ -z "$lfcdir" ]; then 
	echo '$lfcdir is not set'
	return; 
    fi 
    
    # extract $LFC_HOME from path if present
    if [[ $lfcdir == $LFC_HOME* ]]; then 
	lfcdir=${lfcdir/$LFC_HOME\//}
    fi
    
    # path to file list
    filelist=${ND280TRANSFERS}/${lfcdir//\//.}.list 
    
    # get the lfns
    for file in $(lfc-ls $lfcdir); do 
	echo lfn:$LFC_HOME/$lfcdir/$file
    done > $filelist
    
    # file name cannot be returned, so print to stdout for assigment via filelist=$(makeFileList)
    echo $filelist
}

## Function to query job status on arc CEs
arcJobQuery()
{
 
    # get the list of arc/nordugrid CEs from the bdii and loop
    for CE in $(lcg-infosites --vo t2k.org ce | grep nordugrid | awk '{print $6}' | cut -d':' -f1); do

	echo "Querying $CE..."

	# the ldap query and some formatting
	ldapsearch -x -h ${CE} -p 2135 -b "o=grid" "nordugrid-job-globalowner=${DN//\'}" | grep "nordugrid-job-status:" | awk '{print $2}' | sort | uniq -c

	echo
	done
	
}
