#!/bin/bash
redmine=hpcbio-redmine@igb.illinois.edu
if [ $# != 4 ]
then
        MSG="Parameter mismatch."
        echo -e "program=$0 stopped. Reason=$MSG" | mail -s 'Variant Calling Workflow failure message' "$redmine"        
        exit 1;
fi

umask 0027
set -x
echo `date`	
scriptfile=$0
runfile=$1
email=$2
exitstatus=$3
reportticket=$4
numdays=30
outputdir=$( cat $runfile | grep -w OUTPUTDIR | cut -d '=' -f2 )
deliveryfolder=$( cat $runfile | grep -w DELIVERYFOLDER | cut -d '=' -f2 )
genderinfo=$( cat $runfile | grep -w GENDERINFORMATION | cut -d '=' -f2 )
sampleinfo=$( cat $runfile | grep -w SAMPLEINFORMATION | cut -d '=' -f2 )
scriptdir=$( cat $runfile | grep -w SCRIPTDIR | cut -d '=' -f2 )
pbsprj=$( cat $runfile | grep -w PBSPROJECTID | cut -d '=' -f2 )
pbsqueue=$( cat $runfile | grep -w PBSQUEUEWGEN | cut -d '=' -f2 )
autoarchive=$( cat $runfile | grep -w AUTOARCHIVE | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
groupname=$( cat $runfile | grep -w GROUPNAME | cut -d '=' -f2 )
analysis=$( cat $runfile | grep -w ANALYSIS | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
pipeid=$( cat $outputdir/logs/pbs.CONFIGURE )


set +x; echo -e "\n the delivery folder should be populated already with results \n" >&2; set -x

if [ `expr ${#deliveryfolder}` -lt 2 ]
then
	delivery=$outputdir/delivery
else
	delivery=$outputdir/$deliveryfolder
fi

set +x; echo -e "\n populating the delivery/docs folder with documents runfiles etc \n" >&2; set -x

mkdir -p ${delivery}/docs
cp $outputdir/*.txt ${delivery}/docs
cp $outputdir/*.list ${delivery}/docs
cp $sampleinfo  ${delivery}/docs
cp $genderinfo  ${delivery}/docs

echo `date`

set +x; echo -e "\n making the delivery folders group read/writable to the group \n" >&2; set -x

newgrp $groupname
chmod -R 770 ${delivery}
TopOutputLogs=$outputdir/logs
        
if [ $autoarchive == "YES" ]
then

	set +x; echo -e "\n constructing and launching archive job \n" >&2; set -x


	qsub_archive=$TopOutputLogs/qsub.archive.Project_${pipeid}

	echo "#PBS -A $pbsprj" >> $qsub_archive
	echo "#PBS -N ${pipeid}_archiveOutput" >> $qsub_archive
	echo "#PBS -l walltime=01:00:00" >> $qsub_archive # 1 hour should be more than enough
	echo "#PBS -l nodes=1:ppn=1" >> $qsub_archive
	echo "#PBS -o $TopOutputLogs/log.archive.Project_${pipeid}.ou" >> $qsub_archive
	echo "#PBS -e $TopOutputLogs/log.archive.Project_${pipeid}.in" >> $qsub_archive
	echo "#PBS -q $pbsqueue" >> $qsub_archive
	echo "#PBS -m a" >> $qsub_archive
	echo "#PBS -M $email" >> $qsub_archive
	echo "$scriptdir/autoArchive.sh $runfile $email $TopOutputLogs/log.archive.Project_${pipeid}.er $TopOutputLogs/log.archive.Project_${pipeid}.ou"  >> $qsub_archive
	qsub_archive=`qsub $qsub_archive`
	echo $qsub_archive >> $TopOutputLogs/pbs.Archive
	echo `date`	
fi


set +x; echo -e "\n now putting together the second part of the Summary.Report file with the list of jobs executed inside this pipeline \n" >&2; set -x

        
listjobids=$( cat $outputdir/logs/pbs.* cat $outputdir/logs/*/pbs.* | sort | uniq | tr "\n" "\t" )


if [ $exitstatus == "exitok" ]
then
	MSG="Variant calling workflow with id: [$pipeid] by username: $USER finished with ALL  jobs with exit code 0 at: "$( echo `date` )
else
	MSG="Variant calling workflow with id: [$pipeid] by username: $USER finished with SOME jobs with exit code 0  at: "$( echo `date` )
fi

LOGS="Results and execution logs can be found at \n$outputdir\n\nJOBIDS\n\n$listjobids\n\nThis jobid:${PBS_JOBID}\n\n"
echo -e "$MSG\n\nDetails:\n\n$LOGS\n$detjobids\n\nPlease view $outputdir/logs/Summary.Report" | mail -s "[Task #${reportticket}]" "$redmine,$email"
echo -e "$MSG\n\nDetails:\n\n$LOGS\n$detjobids" >> $outputdir/logs/Summary.Report
cp  $outputdir/logs/Summary.Report ${delivery}/docs/Summary.Report      

chmod g+r $TopOutputLogs
        
set +x; echo -e "\n Done. Exiting now \n" >&2; set -x

