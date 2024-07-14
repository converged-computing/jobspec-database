#!/bin/bash
#
# start_realrecal_block.sh
# Second module in the analysis pipeline.This module launches qsubs to convert input bams aligned elsewhere and to launch realignment schedulers according to analysis type
#redmine=hpcbio-redmine@igb.illinois.edu
redmine=grendon@illinois.edu

if [ $# != 5 ]
then
	MSG="parameter mismatch. "
	echo -e "program=$0 stopped. Reason=$MSG" | mail -s 'Variant Calling Workflow failure message' "$redmine"
	exit 1;
fi

echo -e "\n\n############# START REALRECAL BLOCK: This module launches qsubs to convert input bams aligned elsewhere and to launch realignment schedulers according to analysis type  ###############\n\n" >&2
umask 0027
set -x
echo `date`
scriptfile=$0
runfile=$1
elog=$2
olog=$3
email=$4
qsubfile=$5
LOGS="jobid:${PBS_JOBID}\nqsubfile=$qsubfile\nerrorlog=$elog\noutputlog=$olog"

if [ !  -s $runfile ]
then
	MSG="$runfile configuration file not found"
	echo -e "Program $0 stopped. Reason=$MSG" | mail -s "Variant Calling Workflow failure message" "$redmine"
	exit 1;
fi



set +x; echo -e "\n\n" >&2;
# wrapping commends in echoes, so that the output logs would be easier to read: they will have more structure
echo "####################################################################################################" >&2
echo "##################################### PARSING RUN INFO FILE ########################################" >&2
echo "##################################### AND SANITY CHECK      ########################################" >&2
echo "####################################################################################################" >&2
echo -e "\n\n" >&2; set -x;


sampledir=$( cat $runfile | grep -w INPUTDIR | cut -d '=' -f2 )
outputdir=$( cat $runfile | grep -w OUTPUTDIR | cut -d '=' -f2 )
thr=$( cat $runfile | grep -w PBSTHREADS | cut -d '=' -f2 )
refdir=$( cat $runfile | grep -w REFGENOMEDIR | cut -d '=' -f2 )
scriptdir=$( cat $runfile | grep -w SCRIPTDIR | cut -d '=' -f2 )
refgenome=$( cat $runfile | grep -w REFGENOME | cut -d '=' -f2 )
input_type=$( cat $runfile | grep -w INPUTTYPE | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
inputformat=$( cat $runfile | grep -w INPUTFORMAT | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
analysis=$( cat $runfile | grep -w ANALYSIS | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
skipvcall=$( cat $runfile | grep -w SKIPVCALL | cut -d '=' -f2 )
paired=$( cat $runfile | grep -w PAIRED | cut -d '=' -f2 )
rlen=$( cat $runfile | grep -w READLENGTH | cut -d '=' -f2 )
multisample=$( cat $runfile | grep -w MULTISAMPLE | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
region=$( cat $runfile | grep -w CHRINDEX | cut -d '=' -f2 )
resortbam=$( cat $runfile | grep -w RESORTBAM | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
revertsam=$( cat $runfile | grep -w REVERTSAM | cut -d '=' -f2  )
indices=$( echo $region | sed 's/:/ /g' )
picardir=$( cat $runfile | grep -w PICARDIR | cut -d '=' -f2 )
samdir=$( cat $runfile | grep -w SAMDIR | cut -d '=' -f2 )
run_method=$( cat $runfile | grep -w RUNMETHOD | cut -d '=' -f2 )

set +x; echo -e "\n\n\n############ checking output folder\n" >&2; set -x

if [ ! -d $outputdir ]
then
	mkdir -p $outputdir
fi 

set +x; echo -e "\n\n\n############ checking input type: WGS or WES\n" >&2; set -x

if [ $input_type == "GENOME" -o $input_type == "WHOLE_GENOME" -o $input_type == "WHOLEGENOME" -o $input_type == "WGS" ]
then
	pbscpu=$( cat $runfile | grep -w PBSCPUALIGNWGEN | cut -d '=' -f2 )
	pbsqueue=$( cat $runfile | grep -w PBSQUEUEWGEN | cut -d '=' -f2 )
elif [ $input_type == "EXOME" -o $input_type == "WHOLE_EXOME" -o $input_type == "WHOLEEXOME" -o $input_type == "WES" ]
then
	pbscpu=$( cat $runfile | grep -w PBSCPUALIGNEXOME | cut -d '=' -f2 )
	pbsqueue=$( cat $runfile | grep -w PBSQUEUEEXOME | cut -d '=' -f2 )
else
	MSG="Invalid value for INPUTTYPE=$input_type in configuration file."
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi

set +x; echo -e "\n\n\n############ checking resortbam option\n" >&2; set -x

if [ $resortbam != "1" -a $resortbam != "0" -a $resortbam != "YES" -a $resortbam != "NO" ]
then
	MSG="Invalid value for RESORTBAM=$resortbam"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi
if [ $resortbam == "1" ]
then
	$resortbam="YES"
fi
if [ $resortbam == "0" ]
then
	$resortbam="NO"
fi

set +x; echo -e "\n\n\n############ checking tool dirs\n" >&2; set -x

if [ ! -d $scriptdir ]
then
	MSG="$scriptdir script directory not found"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi
if [ ! -d $refdir ]
then
	MSG="$refdir directory of reference genome  not found"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi
if [ ! -d $picardir ]
then
	MSG="$picardir picard directory  not found"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi

set +x; echo -e "\n\n\n############ checking sample configuration file\n" >&2; set -x

numsamples=`wc -l $outputdir/SAMPLENAMES.list | cut -d ' ' -f 1`
if [ $numsamples -lt 1 ]
then
	MSG="No samples found in INPUTDIR=$sampledir."
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi


set +x; echo -e "\n\n" >&2;
echo "####################################################################################################" >&2
echo "#####################################                       ########################################" >&2
echo "#####################################  CREATE  DIRECTORIES  ########################################" >&2
echo "#####################################                       ########################################" >&2
echo "####################################################################################################" >&2
echo -e "\n\n" >&2; set -x;


TopOutputLogs=$outputdir/logs

if [ -d $TopOutputLogs ]
then
	echo "$TopOutputLogs already exists"
	pbsids=""
else
	mkdir -p $TopOutputLogs
fi

pipeid=$( cat $TopOutputLogs/pbs.CONFIGURE )



RealignOutputLogs=$TopOutputLogs/realign
if [ ! -d $RealignOutputLogs ]
then
	mkdir $RealignOutputLogs
fi

#`chmod -R 770 $RealignOutputLogs/`
# where messages about failures will go
truncate -s 0 $RealignOutputLogs/FAILEDmessages
if [ ! -d $RealignOutputLogs/FAILEDjobs ]
then
	mkdir $RealignOutputLogs/FAILEDjobs
else
	rm -r $RealignOutputLogs/FAILEDjobs/*
fi
#`chmod -R 770 $RealignOutputLogs/FAILEDjobs`


set +x; echo -e "\n\n" >&2;
echo "#############################################################################################################" >&2
echo "#####################################                                ########################################" >&2
echo "#####################################  PREPROCESSING BLOCK           ########################################" >&2
echo "#####################################  BAM files aligned elsewhere   ########################################" >&2
echo "#####################################                                ########################################" >&2
echo "#############################################################################################################" >&2
echo -e "\n\n" >&2; set -x;

listfiles="";
sep=":";
JOBSmayo=""  

if [ $inputformat == "BAM" ]
then
	set +x; echo -e "\n # alignment was done elsewhere; starting the workflow with realignment.\n" >&2; set -x

	##############################################################
	#this section needs editing
	##############################################################
      
	while read SampleName
	do
		set +x; echo -e "\n # processing next sample\n" >&2; set -x
		if [ `expr ${#SampleName}` -lt 7 ]
		then
			set +x; echo -e "\n # skipping empty line\n" >&2; set -x
		else
			set +x; echo -e "\n # processing $SampleName. The line should have two fields <samplename> <bamfile>\n" >&2; set -x

			prefix=$( echo $sampledetail | cut -d ' ' -f1 )
			inbamfile=$( echo $sampledetail | cut -d ' ' -f2 )

			if [ ! -s $inbamfile ]
			then
				MSG="parsing $outputdir/SAMPLENAMES_multiplexed.list file failed. realignment failed to start"
				echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
				exit 1;
			fi

			set +x; echo -e "\n################ creating folder for this sample's prior results\n" >&2; set -x
			
			outputalign=$outputdir/$prefix/align
			outputlogs=$TopOutputLogs/$prefix/logs
			tmpbamfile=$inbamfile
			sortedplain=${prefix}.wrg.sorted.bam
			sorted=${prefix}.wdups.sorted.bam
			sortednodups=${prefix}.nodups.sorted.bam

			if [ ! -d $outputdir/$prefix/ ]
			then
				mkdir -p $outputalign
				mkdir -p $outputlogs				
			else
				`rm -r $outputlogs/*`
			fi


			if [ $resortbam == "YES" ]
			then

				set +x; echo -e "\n################ $inbamfile needs to be resorted\n" >&2; set -x

				qsub_sortbammayo=$outputlogs/qsub.sortbammayo.$prefix

				echo "#PBS -A $pbsprj" >> $qsub_sortbammayo
				echo "#PBS -N ${pipeid}_sortbamayo_${prefix}" >> $qsub_sortbammayo
				echo "#PBS -l walltime=$pbscpu" >> $qsub_sortbammayo
				echo "#PBS -l nodes=1:ppn=$thr" >> $qsub_sortbammayo
				echo "#PBS -o $outputlogs/log.sortbammayo.${prefix}.ou" >> $qsub_sortbammayo
				echo "#PBS -e $outputlogs/log.sortbammayo.${prefix}.in" >> $qsub_sortbammayo
				echo "#PBS -q $pbsqueue" >> $qsub_sortbammayo
				echo "#PBS -m ae" >> $qsub_sortbammayo
				echo "#PBS -M $email" >> $qsub_sortbammayo
				echo "$scriptdir/sortbammayo.sh $outputalign $tmpbamfile $sortedplain $sorted $sortednodups $runfile $outputlogs/log.sortbammayo.${prefix}.in $outputlogs/log.sortbammayo.${prefix}.ou $email $outputlogs/qsub.sortbammayo.${prefix}" >> $qsub_sortbammayo
				#`chmod a+r $qsub_sortbammayo`
				sortid=`qsub $qsub_sortbammayo`
				#`qhold -h u $sortid`
				echo $sortid >> $TopOutputLogs/pbs.RESORTED
			else
				set +x; echo -e "\n################ $inbamfile DOES NOT need to be resorted. populate folder with symlinks\n" >&2; set -x
				### TODO: header and index files need to be generated afresh

				cd $outputalign 
				ln -s $inbamfile $sortedplain
				ln -s $inbamfile $sorted
				ln -s $inbamfile $sortednodups            

			fi # end of resortbam if stmt
		fi # end of if statement checking for empty line in the SampleName file
	done <  $outputdir/SAMPLENAMES_multiplexed.list
	# end loop over input samples

fi # end if inputformat



set +x; echo -e "\n\n" >&2;
echo "#############################################################################################################" >&2
echo "#####################################  END OF BLOCK FOR              ########################################" >&2
echo "#####################################  BAM files aligned elsewhere   ########################################" >&2
echo "#############################################################################################################" >&2
echo -e "\n\n" >&2; set -x;

# grab job ids to be used for job dependencies
JOBSresorted=$( cat $TopOutputLogs/pbs.RESORTED | sed "s/\.[a-zA-Z]*//" | tr "\n" ":" | sed "s/^://" )


set +x; echo -e "\n\n" >&2;
echo "######################################################################################" >&2
echo "#############   NOW THAT THE INPUT HAVE BEEN CHECKED AND RESORTED,  ##################" >&2
echo "#############   WE CAN PROCEED TO SCHEDULE REAL/RECAL ETC           ##################" >&2
echo "#############   CASE1: MULTIPLEXED CASE2: NONMULTIPLEXED            ##################" >&2
echo "######################################################################################" >&2
echo -e "\n\n" >&2; set -x;


# schedule_realrecal should 

qsub_realrecal=$RealignOutputLogs/qsub.schedule_realrecal
cat $outputdir/qsubGenericHeader > $qsub_realrecal
echo "#PBS -N ${pipeid}_schedule_realrecal" >> $qsub_realrecal
echo "#PBS -l walltime=01:00:00" >> $qsub_realrecal
echo "#PBS -l nodes=1:ppn=1" >> $qsub_realrecal
echo "#PBS -o $RealignOutputLogs/log.schedule_realrecal.ou" >> $qsub_realrecal
echo "#PBS -e $RealignOutputLogs/log.schedule_realrecal.in" >> $qsub_realrecal

# inserting dependencies 
if [ `expr ${#JOBSresorted}` -gt 0 ]
then
	echo "#PBS -W depend=afterok:$JOBSresorted" >> $qsub_realrecal
fi

# choosing the script to run in the qsub script
if [ $analysis != "MULTIPLEXED" ]
then
	set +x; echo -e "\n################ ANALYSIS IS NONMULTIPLEXED\n" >&2; set -x

	echo "$scriptdir/schedule_realrecal_nonmultiplexed.sh $outputdir $runfile $RealignOutputLogs/log.schedule_realrecal.in $RealignOutputLogs/log.schedule_realrecal.ou $email $RealignOutputLogs/qsub.schedule_realrecal" >> $RealignOutputLogs/qsub.schedule_realrecal
else
	set +x; echo -e "\n################ ANALYSIS IS MULTIPLEXED\n" >&2; set -x
	
	echo "$scriptdir/schedule_realrecal_multiplexed.sh $outputdir $runfile $RealignOutputLogs/log.schedule_realrecal.in $RealignOutputLogs/log.schedule_realrecal.ou $email $RealignOutputLogs/qsub.schedule_realrecal" >> $RealignOutputLogs/qsub.schedule_realrecal
fi

#`chmod a+r $qsub_realrecal`               
realrecaljob=`qsub $qsub_realrecal`
# `qhold -h u $realrecaljob` 
echo $realrecaljob >> $TopOutputLogs/pbs.RECALL


set +x; echo -e "\n ###################### now making PBS log files read accessible to the group #################################\n" >&2; set -x
echo `date`
#`chmod -R 770 $outputdir/`
#`chmod -R 770 $TopOutputLogs/`

find $outputdir -name logs -type d | awk '{print "chmod -R g=rwx "$1}' | sh -x

echo `date`
