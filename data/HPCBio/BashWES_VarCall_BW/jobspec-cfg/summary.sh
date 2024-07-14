#!/bin/bash
redmine=hpcbio-redmine@igb.illinois.edu
##redmine=grendon@illinois.edu
if [ $# != 4 ]
then
        MSG="Parameter mismatch."
        echo -e "program=$0 stopped. Reason=$MSG" | mail -s 'Variant Calling Workflow failure message' "$redmine"
        
        exit 1;
fi


set -x
echo `date`	
scriptfile=$0
runfile=$1
elog=$2
olog=$3
qsubfile=$4
LOGS="jobid:${PBS_JOBID}\nqsubfile=$qsubfile\nerrorlog=$elog\noutputlog=$olog"

outputdir=$( cat $runfile | grep -w OUTPUTDIR | cut -d '=' -f2 )
sampleinfo=$( cat $runfile | grep -w SAMPLEINFORMATION | cut -d '=' -f2 )        
reportticket=$( cat $runfile | grep -w REPORTTICKET | cut -d '=' -f2 )
deliverydir=$( cat $runfile | grep -w DELIVERYFOLDER | cut -d '=' -f2 )
summaryok="YES"

if [ ! -s ${outputdir}/logs/mail.${analysis}.FAILURE ]
then
    MSG="Variant calling workflow run by username: ${USER}\nfinished successfully at: "$( echo `date` )
    echo -e ${MSG}  >> $outputdir/$deliverydir/docs/Summary.Report
else
    MSG="Variant calling workflow run by username: ${USER}\nfinished with FAILUERS at: "$( echo `date` )
    echo -e ${MSG}  >> $outputdir/$deliverydir/docs/Summary.Report
fi
qcfails=`grep FAIL ${outputdir}/delivery/docs/QC_test_results.txt | wc -l`
qcwarns=`grep WARN ${outputdir}/delivery/docs/QC_test_results.txt | wc -l`
if [ $qcfails -gt 0  ]
then
    echo -e "\nQC HAD FAILURES" >> $outputdir/$deliverydir/docs/Summary.Report
    summaryok="QC"
fi
if [ $qcwarns -gt 0  ]
then
    echo -e "QC HAD WARNINGS" >> $outputdir/$deliverydir/docs/Summary.Report
    summaryok="QC"
fi



set +x
echo -e "\n\n##################################################################"
echo -e "####   Now putting together the second part of the Summary.   ####"
echo -e "####   Report file with the list of jobs executed             ####"
echo -e "##################################################################\n\n"
set -x 
listjobids=$( cat $outputdir/logs/pbs.* | sort | uniq | tr "\n" "\t" )

LOGS="Results and execution logs can be found at $outputdir\n\nDeliverables are located at $outputdir/$deliverydir\n\nJOBIDS\n\n$listjobids\n\nThis jobid:${PBS_JOBID}\n\n"
echo -e "\n\n$LOGS" >> $outputdir/$deliverydir/docs/Summary.Report


chmod -R g+r $outputdir
set +x
echo -e "\n\n############################################################################################################"
echo -e "############                              DONE. EXITING NOW                                      ###########"
echo -e "############################################################################################################\n\n"
set -x
