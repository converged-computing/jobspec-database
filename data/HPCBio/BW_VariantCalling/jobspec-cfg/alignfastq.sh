#!/bin/bash
#
# alignfastq.sh
# align module to be used for input files in fastq format. This module schedules fastqc, alignment and dedup jobs
#redmine=hpcbio-redmine@igb.illinois.edu
redmine=grendon@illinois.edu

if [ $# != 5 ]
then
        MSG="Parameter mismatch"
        echo -e "program=$0 stopped. Reason=$MSG" | mail -s 'Variant Calling Workflow failure message' "$redmine"
        exit 1;
fi

echo -e "\n\n############# BEGIN ALIGNFASTQ PROCEDURE: schedule fastqc, parse sample information and create alignment jobs  ###############\n\n" >&2
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

set +x; echo -e "\n\n" >&2; 
echo "####################################################################################################" >&2
echo "##################################### PARSING RUN INFO FILE ########################################" >&2
echo "##################################### AND SANITY CHECK      ########################################" >&2
echo "####################################################################################################" >&2
echo -e "\n\n" >&2; set -x;

if [ !  -s $runfile ]
then
   MSG="$runfile configuration file not found."
   echo -e "Program $0 stopped. Reason=$MSG" | mail -s "Variant Calling Workflow failure message" "$redmine"
   exit 1;
fi


reportticket=$( cat $runfile | grep -w REPORTTICKET | cut -d '=' -f2 )
run_method=$( cat $runfile | grep -w RUNMETHOD | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
outputdir=$( cat $runfile | grep -w OUTPUTDIR | cut -d '=' -f2 )
nodes=$( cat $runfile | grep -w PBSNODES | cut -d '=' -f2 )
pbsprj=$( cat $runfile | grep -w PBSPROJECTID | cut -d '=' -f2 )
thr=$( cat $runfile | grep -w PBSTHREADS | cut -d '=' -f2 )
refdir=$( cat $runfile | grep -w REFGENOMEDIR | cut -d '=' -f2 )
scriptdir=$( cat $runfile | grep -w SCRIPTDIR | cut -d '=' -f2 )
ref=$( cat $runfile | grep -w REFGENOME | cut -d '=' -f2 )
chunkfastq=$( cat $runfile | grep -w CHUNKFASTQ | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
launcherdir=$( cat $runfile | grep -w LAUNCHERDIR | cut -d '=' -f2 )
aligner=$( cat $runfile | grep -w ALIGNER | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
fastqcdir=$( cat $runfile | grep -w FASTQCDIR | cut -d '=' -f2 )
fastqcflag=$( cat $runfile | grep -w FASTQCFLAG | cut -d '=' -f2 )
fastqcparms=$( cat $runfile | grep -w FASTQCPARMS | cut -d '=' -f2 | tr " " "_" )_-t_${thr}
picardir=$( cat $runfile | grep -w PICARDIR | cut -d '=' -f2 )
samdir=$( cat $runfile | grep -w SAMDIR | cut -d '=' -f2 )
samblasterdir=$( cat $runfile | grep -w SAMBLASTERDIR | cut -d '=' -f2 )
dup=$( cat $runfile | grep -w MARKDUP  | cut -d '=' -f2 )
dupflag=$( cat $runfile | grep -w REMOVE_DUP  | cut -d '=' -f2 )
input_type=$( cat $runfile | grep -w INPUTTYPE | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
paired=$( cat $runfile | grep -w PAIRED | cut -d '=' -f2 )
rlen=$( cat $runfile | grep -w READLENGTH | cut -d '=' -f2 )
multisample=$( cat $runfile | grep -w MULTISAMPLE | cut -d '=' -f2 )
sortool=$( cat $runfile | grep -w SORTMERGETOOL | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
markduplicatestool=$( cat $runfile | grep -w MARKDUPLICATESTOOL | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
analysis=$( cat $runfile | grep -w ANALYSIS | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
profiling=$( cat $runfile | grep -w PROFILING | cut -d '=' -f2 )
profiler=$( cat $runfile | grep -w PROFILER | cut -d '=' -f2 )
cleanupflag=$( cat $runfile | grep -w REMOVETEMPFILES | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
splitAlignDedup=$( cat $runfile | grep -w SPLITALIGNDEDUP | cut -d '=' -f2 ) #binay value YES|NO
run_cmd=$( cat $runfile | grep -w LAUNCHERCMD | cut -d '=' -f2 )   #string value aprun|mpirun
bash_cmd=`which bash`        
QCfile=$outputdir/QC_Results.txt        
truncate -s 0 $QCfile

set +x; echo -e "\n\n\n############ checking workflow scripts directory\n" >&2; set -x;

if [ ! -d $scriptdir ]
then
	MSG="SCRIPTDIR=$scriptdir directory not found"
	echo -e "$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi
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


set +x; echo -e "\n\n\n############# check that the file with sample configuration info exists\n" >&2; set -x

if [ ! -s $outputdir/SAMPLENAMES_multiplexed.list ]
then
	MSG="$outputdir/SAMPLENAMES_multiplexed.list file not found"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi


# The file that contains sampleinfo. This file MUST always exist. It has a variable number of columns.
# 5 columns when input is multiplexed    && paired fastq format
# 3 columns when input is nonmultiplexed && paired fastq foramt
# 2 columns when input is nonmultiplexed && bam

TheInputFile=$outputdir/SAMPLENAMES_multiplexed.list

set +x; echo -e "\n\n\n############# check whether fastq will be chunked\n" >&2; set -x
if [ $chunkfastq != "YES" -a $chunkfastq != "NO" ]
then
	MSG="CHUNKFASTQ variable must be binary YES/NO; incorrect value encountered"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi


set +x; echo -e "\n\n\n############ checking settings for marking of duplicates\n" >&2; set -x
if [ $dup != "1" -a $dup != "0" -a $dup != "YES" -a $dup != "NO" ]
then
	MSG="Invalid value for MARKDUP=$dup"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi
if [ $dup == "1" ]
then
	dup="YES"
fi
if [ $dup == "0" ]
then
	dup="NO"
fi

if [ $dupflag != "1" -a $dupflag != "0" -a $dupflag != "YES" -a $dupflag != "NO" ]
then
	MSG="Invalid value for REMOVE_DUP=$dupflag"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi
if [ $dupflag == "1" ]
then
	dupflag="YES"
fi
if [ $dupflag == "0" ]
then
	dupflag="NO"
fi
        
dupparms=$( echo "dup=${dup}_flag=${dupflag}" )


set +x; echo -e "\n\n\n############ checking FastQC settings\n" >&2; set -x 
if [ $fastqcflag != "1" -a $fastqcflag != "0" -a $fastqcflag != "YES" -a $fastqcflag != "NO" ]
then
	MSG="Invalid value for FASTQCFLAG=$fastqcflag"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi
if [ $fastqcflag == "1" ]
then
	fastqcflag="YES"
fi
if [ $fastqcflag == "0" ]
then
	fastqcflag="NO"
fi


set +x; echo -e "\n\n\n############ checking Cleanup settings\n" >&2; set -x
if [ $cleanupflag != "1" -a $cleanupflag != "0" -a $cleanupflag != "YES" -a $cleanupflag != "NO" ]
then
	MSG="Invalid value for REMOVETEMPFILES=$cleanupflag"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi
if [ $cleanupflag == "1" ]
then
	cleanupflag="YES"
fi
if [ $cleanupflag == "0" ]
then
	cleanupflag="NO"
fi

set +x; echo -e "\n\n\n############ launcher command aprun or mpirun\n" >&2; set -x

if [ $run_cmd == "aprun" ]
then        
	run_cmd="aprun -n "
else        
	run_cmd="mpirun -np " 
fi

set +x; echo -e "\n\n\n############ checking computational tools\n" >&2; set -x
if [ $aligner != "NOVOALIGN" -a $aligner != "BWA_ALN" -a $aligner != "BWA_MEM"]
then
	MSG="ALIGNER=$aligner  is not available at this site"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi
if [ $aligner == "NOVOALIGN" ]
then
	alignerdir=$( cat $runfile | grep -w NOVODIR | cut -d '=' -f2 )
	refindexed=$( cat $runfile | grep -w NOVOINDEX | cut -d '=' -f2 )
	alignparms=$( cat $runfile | grep -w NOVOPARAMS | cut -d '=' -f2 | tr " " "_" )
fi
if [ $aligner == "BWA_ALN" ]
then
	alignerdir=$( cat $runfile | grep -w BWAALNDIR | cut -d '=' -f2 )
	refindexed=$( cat $runfile | grep -w BWAALNINDEX | cut -d '=' -f2 )
	alignparms=$( cat $runfile | grep -w BWAALNPARAMS | cut -d '=' -f2 | tr " " "_" )
fi
if [ $aligner == "BWA_MEM" ]
then
	alignerdir=$( cat $runfile | grep -w BWAMEMDIR | cut -d '=' -f2 )
	refindexed=$( cat $runfile | grep -w BWAMEMINDEX | cut -d '=' -f2 )
	alignparms=$( cat $runfile | grep -w BWAMEMPARAMS | cut -d '=' -f2 | tr " " "_" )
fi

if [ $sortool != "NOVOSORT" -a $sortool != "PICARD" ]
then
	MSG="Invalid value for SORTOOL=$sortool in configuration file"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi

if [ $markduplicatestool != "PICARD" -a $markduplicatestool != "SAMBLASTER" ]
then
	MSG="Invalid value for SORTOOL=$sortool in configuration file"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi

if [ ! -d $alignerdir ]
then
	MSG="$alignerdir aligner directory not found"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi

if [ ! -d $picardir ]
then
	MSG="PICARDIR=$picardir directory not found"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi

if [ ! -d $samdir ]
then
	MSG="SAMDIR=$samdir directory not found"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi

if [ ! -e $profiler ]
then
	MSG="PROFILER=$profiler not found"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	#exit 1;
fi

set +x; echo -e "\n\n\n############ checking presence of references\n" >&2; set -x
if [ ! -s $refdir/$ref ]
then
	MSG="$refdir/$ref reference genome not found"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi


set +x; echo -e "\n\n\n############ checking run method\n" >&2; set -x
if [ $run_method != "LAUNCHER" -a $run_method != "QSUB" -a $run_method != "APRUN" ]
then
	MSG="Invalid value for RUNMETHOD=$run_method"
	echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
	exit 1;
fi


set +x; echo -e "\n\n">&2
echo "############################################################################################################" >&2
echo "###############################  CREATE  DIRECTORY STRUCTURE FOR LOGS. TOP LEVEL ONLY     ##################" >&2
echo "############################################################################################################" >&2
echo -e "\n\n" >&2; set -x;


TopOutputLogs=$outputdir/logs

if [ -d $TopOutputLogs ]
then
	pbsids=""
else
	mkdir -p $TopOutputLogs/align
fi

if [ $run_method == "LAUNCHER" ]
then
	mkdir -p $TopOutputLogs/align
fi
pipeid=$( cat $TopOutputLogs/pbs.CONFIGURE )
truncate -s 0 $TopOutputLogs/pbs.SCHREAL
truncate -s 0 $TopOutputLogs/pbs.ALIGNED
truncate -s 0 $TopOutputLogs/pbs.MARKED
truncate -s 0 $TopOutputLogs/FastqcAnisimov.joblist
truncate -s 0 $TopOutputLogs/align/AlignAnisimov.joblist 
truncate -s 0 $TopOutputLogs/align/MarkdupsAnisimov.joblist

#chunks=`expr $nodes "-" 1`
#if [ $chunks -lt 1 ]
#then
#    chunks=$nodes
#fi
nthreads=`expr $thr "-" 1`
if [ $nthreads -lt 1 ]
then
	nthreads=$thr
fi



set +x; echo -e "\n\n">&2
echo "############################################################################################################" >&2
echo "##################################### ALIGNMENT: LOOP1 OVER SAMPLES ########################################" >&2
echo "############################################################################################################" >&2
echo -e "\n\n" >&2; set -x;


while read SampleLine
do
	set +x; echo -e "\n\n############ processing next sample \n" >&2; set -x;
	# this will evaluate the length of string
	if [ `expr ${#SampleLine}` -lt 1 ]
	then
		set +x; echo -e "\n\n############ skipping empty line \n" >&2; set -x;
	else
		set +x; echo -e "\n\n">&2
		echo -e "#################################### PREP WORK FOR $SampleLine            ########################################\n" >&2
		echo -e "#################################### PARSING READS and CREATING RGLINE    ########################################\n" >&2
		echo -e "\n\n" >&2; set -x;


		if [ $analysis == "MULTIPLEXED" ]
		then
			set +x; echo -e "\n" >&2
			echo -e "\n ###### Parsing SAMPLENAMES_multiplexed.list" >&2
			echo -e "\n ###### Current line has FIVE fields:" >&2
			echo -e "\n ###### col1=sampleid col2=read1 col3=read2 col4=flowcell_and_lane_name col5=lib " >&2
			echo -e "\n ###### filenames of read1 and read2 MUST include full path 
			echo -e "\n ###### The code parses PL and CN from runfile\n" >&2; 
			echo -w "\n" >&2; set -x;

			SampleName=$( echo -e "$SampleLine" | cut -f 4 )
			LeftReadsFastq=$( echo -e "$SampleLine" | cut -f 2 )
			RightReadsFastq=$( echo -e "$SampleLine" | cut -f 3 )
			sID=$( echo -e "$SampleLine" | cut -f 4 )
			sPU=$( echo -e "$SampleLine" | cut -f 4 )
			sSM=$( echo -e "$SampleLine" | cut -f 1 )
			sLB=$( echo -e "$SampleLine" | cut -f 5 )
			sPL=$( cat $runfile | grep -w SAMPLEPL | cut -d '=' -f2 )
			sCN=$( cat $runfile | grep -w SAMPLECN | cut -d '=' -f2 )
		else
			set +x; echo -e "\n" >&2
			echo -e "\n ###### Parsing SAMPLENAMES_multiplexed.list" >&2
			echo -e "\n ###### Current line has THREE fields:" >&2
			echo -e "\n ###### col1=sampleid col2=read1 col3=read2 " >&2
			echo -e "\n ###### The code parses PL LB and CN from runfile\n" >&2; 
			echo -w "\n" >&2; set -x;

			SampleName=$( echo -e "$SampleLine" | cut -f 1 )
			LeftReadsFastq=$( echo -e "$SampleLine" | cut -f 2 )
			RightReadsFastq=$( echo -e "$SampleLine" | cut -f 3 )
			sID=$SampleName
			sPU=$SampleName
			sSM=$SampleName
			sPL=$( cat $runfile | grep -w SAMPLEPL | cut -d '=' -f2 )
			sCN=$( cat $runfile | grep -w SAMPLECN | cut -d '=' -f2 )
			sLB=$( cat $runfile | grep -w SAMPLELB | cut -d '=' -f2 )		
		fi  # end if $analysis

		set +x; echo -e "\n\n checking that it actually worked: we must have RG line and file names of the reads... \n" >&2; set -x;

		if [ `expr ${#sID}` -lt 1 -o `expr ${#sPL}` -lt 1 -o `expr ${#sCN}` -lt 1 ] 
		then
			MSG="ID=$sID PL=$sPL CN=$sCN invalid values. The RG line cannot be formed"
			echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
			exit 1;
		fi


		RGparms=$( echo "ID=${sID}:LB=${sLB}:PL=${sPL}:PU=${sPU}:SM=${sSM}:CN=${sCN}" )

		set +x; echo -e "\n\n">&2
		echo -e "#################################### CHECKING THAT THE INPUT FILES EXIST  ########################################\n" >&2
		echo -e "\n\n" >&2; set -x;

		if [ ! -s $LeftReadsFastq ]
		then
			MSG="$LeftReadsFastq left reads file not found"
			echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
			exit 1;
		fi
		
		echo `date`

		if [ $paired -eq 1 -a ! -s $RightReadsFastq ]
		then
			MSG="$RightReadsFastq right reads file not found"
			echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
			exit 1;
		fi


		set +x; echo -e "\n\n">&2
		echo -e "#################################### CREATING OUTPUT FOLDERS AND FILENAMES        ################################\n" >&2
		echo -e "\n\n" >&2; set -x;

		AlignOutputDir=$outputdir/$SampleName/align
		AlignOutputLogs=$AlignOutputDir/logs
		FastqcOutputDir=$outputdir/$SampleName/fastqc
		FastqcOutputLogs=$FastqcOutputDir/logs

		if [ -d $AlignOutputDir ]
		then
			# perhaps we should stop when a sample is seen more than once. But for now, we simply reset the folder
			set +x; echo -e "\n\n $AlignOutputDir is there; resetting it" >&2; set -x;
			`rm  $AlignOutputDir/*`
			mkdir -p $AlignOutputDir/logs
		else
			mkdir -p $AlignOutputDir/logs
		fi

	        if [ -d $FastqcOutputDir ]
	        then		
			set +x; echo -e "\n\n fastqcflag=$fastqcflag. need to prepare the output folder for these results" >&2; set -x;
			`rm   $FastqcOutputDir/*` 
			mkdir -p $FastqcOutputDir/logs 
                else
			mkdir -p $FastqcOutputDir/logs
                fi 

            
		#`chmod -R 770 $outputdir/$SampleName/`

		# filenames of logs, temporary and output files go here. 

		outputsamfileprefix=$AlignOutputDir/$SampleName
		sortedplain=$outputsamfileprefix.wrg.sorted.bam
		outsortnodup=$outputsamfileprefix.nodups.sorted.bam
		outsortwdup=$outputsamfileprefix.wdups.sorted.bam
		failedlog=$TopOutputLogs/FAILED_align.${sample} 


		set +x; echo -e "\n\n ##################### CHECKING IF WE NEED TO LAUNCH FASTQC CMD                           #######\n\n" >&2; set -x;

		if [ $fastqcflag == "YES" ]
		then
			set +x; echo -e "\n\n" >&2; 
			echo "#################################### FASTQFLAG == YES RUNNING FASTQC ON UNALIGNED READS of $SampleName #####################" >&2
			echo -e "\n\n" >&2; set -x;

			# check that fastqc tool is there
			if [ ! -d $fastqcdir ]
			then
				MSG="FASTQCDIR=$fastqcdir directory not found"
				echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
				exit 1;
			fi

			# gather the input files to run fastqc on                
			if [ $paired -ne 1 ]
			then                
				fastqc_input=$LeftReadsFastq                
			else
				fastqc_input="$LeftReadsFastq:$RightReadsFastq"
			fi
			


			FastqcOutputLogs=$TopOutputLogs

			
			# form the fastqc command in jobfile and then add it to the anisimov launcher 
			
			jobfileFastqc=$FastqcOutputLogs/Fastqc.$SampleName.jobfile
			truncate -s 0 $jobfileFastqc
			echo "$scriptdir/fastqc.sh $runfile $fastqcdir $FastqcOutputDir $fastqcparms $fastqc_input $FastqcOutputLogs/log.Fastqc_${SampleName}.in $FastqcOutputLogs/log.Fastqc_${SampleName}.ou $email $FastqcOutputLogs/qsub.Fastqc_$SampleName > $FastqcOutputLogs/log.Fastqc_${SampleName}.in" > $jobfileFastqc
			jobfilename=$( basename $jobfileFastqc )
			echo "$FastqcOutputLogs $jobfilename" >> $FastqcOutputLogs/FastqcAnisimov.joblist

		else
			set +x; echo -e "\n\n ############ FASTQCFLAG == NO. Quality information for fastq files will NOT be calculated." >&2; set -x;
		fi
            
		## done with generating quality info for each read file


		set +x; echo -e "\n\n" >&2;
		echo -e "#################################### DONE WITH FASTQC BLOCK                           ##################\n\n" >&2
		echo -e "#################################### SELECT TO CHUNK/Not2CHUNK READS for $SampleName  ##################\n\n" >&2; set -x;

		# create new names for chunks of fastq
		# doing this outside the chunkfastq conditional, because
		# non-chunking is handled by creating a symbolic link from chunk 0 to original fastq

		cd $AlignOutputDir
		LeftReadsChunkNamePrefix=leftreads_chunk
		RightReadsChunkNamePrefix=rightreads_chunk
		if [ $chunkfastq == "YES" ]
		then

			set +x; echo -e "\n ######## splitting files into chunks before aligning \n" >&2; set -x;
			## remember that one fastq read is made up of four lines
			NumChunks=`expr $nodes "-" 1`
			if [ $NumChunks -lt 1 ]
			then
				NumChunks=$nodes
			fi
			#if [ $NumChunks -lt 1 ]
			#then
			#    NumChunks=1
			#fi

			NumLinesInLeftFastq=`wc -l $LeftReadsFastq | cut -d ' ' -f 1`
			NumReadsInOriginalFastq=`expr $NumLinesInLeftFastq "/" 4`
			NumReadsPerChunk=`expr $NumReadsInOriginalFastq "/" $NumChunks`
			RemainderReads=`expr $NumReadsInOriginalFastq "%" $NumChunks`
			NumLinesPerChunk=`expr $NumReadsPerChunk "*" 4`

			if [ $RemainderReads -eq 0  ]
			then
				set +x; echo -e "\n # mod is 0; no reads for last chunk file, one idle node \n" >&2; set -x;
				(( NumChunks-- ))
			fi

			set +x; echo -e "\n # splitting read file1=$LeftReadsFastq \n" >&2; set -x;
			`split -l $NumLinesPerChunk -a 2 -d $LeftReadsFastq $LeftReadsChunkNamePrefix`

			exitcode=$?
			if [ $exitcode -ne 0 ]
			then
				MSG="splitting of read file $LeftReadsFastq failed. exitcode=$exitcode"
				echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
				exit $exitcode;
			fi

			if [ $paired -eq 1 ]
			then
				set +x; echo -e "\n # splitting read file2=$RightReadsFastq \n" >&2; set -x;
				`split -l $NumLinesPerChunk -a 2 -d $RightReadsFastq $RightReadsChunkNamePrefix`
				exitcode=$?
				if [ $exitcode -ne 0 ]
				then
					MSG="splitting of read file $RightReadsFastq failed.  exitcode=$exitcode"
					echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
					exit $exitcode;
				fi
			fi
		else

			set +x; echo -e "\n ######## NO splitting of files into chunks before aligning \n" >&2; set -x;            
			NumChunks=0
			set +x; echo -e "\n\n" >&2; 
			echo "# copying original fastq into a single chunk takes too long for whole genome" >&2;
			echo "# instead, we will create symbolic link to the original file" >&2;
			echo "# adding double-0 to the chunk name, because expecting to chunk files into tens of pieces" >&2;
			echo "# need consistency in naming: two digits for chunk number" >&2;
			echo -e "\n\n" >&2; set -x;
			
			#   `cp $LeftReadsFastq ${LeftReadsChunkNamePrefix}0`
			ln -s $LeftReadsFastq leftreads_chunk00

			if [ $paired -eq 1 ]
			then
			#   `cp $RightReadsFastq ${RightReadsChunkNamePrefix}0`
			  ln -s $RightReadsFastq rightreads_chunk00
			fi
		fi

		## done chunking input fastq

		set +x; echo -e "\n\n\n" >&2;
		echo -e "#################################### DONE WITH CHUNKING DATA                 ########################################" >&2
		echo -e "#################################### CREATING ALL QSUBS FOR ALIGMENT         ########################################" >&2
		echo -e "#################################### SELECTING CASE --BASED ON ALIGNER       ########################################" >&2
		echo -e "\n\n" >&2; set -x

		set +x; echo -e "\n\n" >&2;       
		echo -e "#####################################################################################################################" >&2
		echo -e "#####################################                               #################################################" >&2
		echo -e "##################################### ALIGNMENT: LOOP2 OVER CHUNKS   ################################################" >&2
		echo -e "################## IT WILL ITERATE AT LEAST ONCE EVEN IF NO CHUNKING WAS PERFORMED  #################################" >&2
		echo -e "#####################################################################################################################" >&2
		echo -e "\n\n" >&2; set -x;  

		allfiles=""
		for i in $(seq 0 $NumChunks)
		do
			set +x; echo -e "\n ######### step 1: grab  chunk $i from each read file" >&2; set -x
			echo `date`
			fqfiles=""  #list of fastq files to pass to bwa mem
			if (( $i < 10 ))
			then
				Rone=${LeftReadsChunkNamePrefix}0$i
				OutputFileSuffix=0${i}
			else
				Rone=${LeftReadsChunkNamePrefix}$i
				OutputFileSuffix=${i}
			fi
			if [ ! -s $Rone ]
			then
				MSG="chunk $i of read file $LeftReadsFastq file not found"
				echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
				exit 1;
			fi
			fqfiles=$Rone
			if [ $paired -eq 1 ]
			then
				if (( $i < 10 ))
				then
					Rtwo=${RightReadsChunkNamePrefix}0$i
				else
					Rtwo=${RightReadsChunkNamePrefix}$i
				fi
				if [ ! -s $Rtwo ]
				then
					MSG="chunk $i of read file $RightReadsFastq file not found"
					echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n\nDetails:\n\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
					exit 1;
				fi
				fqfiles=${Rone}:$Rtwo
			fi
                
			set +x; echo -e "\n ######### step 2: generate the alignment cmd depending on the aligner " >&2; set -x

			if [ $aligner == "NOVOALIGN"  ]
			then
				set +x; echo -e "\n###############   novoalign is used as aligner. input file in fastq format ################\n" >&2; set -x
				##############################################################
				#this section needs editing
				##############################################################
				
				if [ $paired -eq 1 ]
				then
					echo "$scriptdir/novosplit.sh $alignerdir $alignparms $refdir/$refindexed $AlignOutputDir $outputsamfileprefix.node$OutputFileSuffix.sam $outputsamfileprefix.node$OutputFileSuffix.bam $scriptdir $runfile $paired $AlignOutputDir/$Rone $AlignOutputDir/$Rtwo $AlignOutputLogs/log.novoaln.$SampleName.node$OutputFileSuffix.in $AlignOutputLogs/log.novoaln.$SampleName.node$OutputFileSuffix.ou $email $AlignOutputLogs/qsub.novoaln.$SampleName.node$OutputFileSuffix" >> $AlignOutputDir/logs/novosplit.${SampleName}.node$OutputFileSuffix
				else
					echo "$scriptdir/novosplit.sh $alignerdir $alignparms $refdir/$refindexed $AlignOutputDir $outputsamfileprefix.node$OutputFileSuffix.sam $outputsamfileprefix.node$OutputFileSuffix.bam $scriptdir $runfile $paired $AlignOutputDir/$Rone $AlignOutputLogs/log.novoaln.$SampleName.node$OutputFileSuffix.in $AlignOutputLogs/log.novoaln.$SampleName.node$OutputFileSuffix.ou $email $AlignOutputLogs/qsub.novoaln.$SampleName.node$OutputFileSuffix" >> $AlignOutputDir/logs/novosplit.${SampleName}.node$OutputFileSuffix   
				fi
			fi
			if [ $aligner == "BWA_ALN" ] 
			then
				set +x; echo -e "\n############# bwa is used as aligner. input file format is in fastq. this segment needs to be rewritten ###############\n" >&2; set -x
				
				##############################################################
				#this section needs editing
				##############################################################
				
				echo "$scriptdir/bwaS1.sh $alignerdir $alignparms $refdir/$refindexed $AlignOutputDir $outputsamfileprefix.node$OutputFileSuffix.R1.sai $AlignOutputDir/$Rone $scriptdir $AlignOutputLogs/log.bwar1.$SampleName.node$OutputFileSuffix.in $AlignOutputLogs/log.bwar1.$SampleName.node$OutputFileSuffix.ou $email $AlignOutputLogs/qsub.bwar1.$SampleName.node$OutputFileSuffix" >> $AlignOutputDir/logs/bwar1.${SampleName}.node$OutputFileSuffix

				if [ $paired -eq 1 ]
				then
					set +x; echo -e "\n########## bwa aligner. paired-end reads #################\n" >&2; set -x

					echo "$scriptdir/bwaS1.sh $alignerdir $alignparms $refdir/$refindexed $AlignOutputDir $outputsamfileprefix.node$OutputFileSuffix.R2.sai $AlignOutputDir/$Rtwo $scriptdir $AlignOutputLogs/log.bwar2.$SampleName.node$OutputFileSuffix.in $AlignOutputLogs/log.bwar2.$SampleName.node$OutputFileSuffix.ou $email $AlignOutputLogs/qsub.bwar2.$SampleName.node$OutputFileSuffix" >> $AlignOutputDir/logs/bwar2.${SampleName}.node$OutputFileSuffix

					echo "$scriptdir/bwaS2.sh $alignerdir $refdir/$refindexed $AlignOutputDir $outputsamfileprefix.node$OutputFileSuffix.R1.sai $outputsamfileprefix.node$OutputFileSuffix.R2.sai $AlignOutputDir/$Rone $AlignOutputDir/$Rtwo $outputsamfileprefix.node$OutputFileSuffix.sam $outputsamfileprefix.node$OutputFileSuffix.bam $samdir $AlignOutputLogs/log.bwasampe.$SampleName.node$OutputFileSuffix.in $AlignOutputLogs/log.bwasampe.$SampleName.node$OutputFileSuffix.ou $email $AlignOutputLogs/qsub.bwasampe.$SampleName.node$OutputFileSuffix" >> $AlignOutputDir/logs/bwaS2.${SampleName}.node$OutputFileSuffix

				else
					set +x; echo -e "\n############# bwa aligner. single read #################\n" >&2; set -x

					echo "$scriptdir/bwaS3.sh $alignerdir $refdir/$refindexed $AlignOutputDir $outputsamfileprefix.node$OutputFileSuffix.R1.sai $AlignOutputDir/$Rone $outputsamfileprefix.node$OutputFileSuffix.sam $outputsamfileprefix.node$OutputFileSuffix.bam $samdir $AlignOutputLogs/log.bwasamse.$SampleName.node$OutputFileSuffix.in $AlignOutputLogs/log.bwasamse.$SampleName.node$OutputFileSuffix.ou $email $AlignOutputLogs/qsub.bwasamse.$SampleName.node$OutputFileSuffix" >> $AlignOutputDir/logs/bwasamse.${SampleName}.node$OutputFileSuffix

				fi
			fi
			if [ $aligner == "BWA_MEM" ]
			then
				set +x; echo -e "\n################ bwa mem is used as aligner. input file format is in fastq ###############\n" >&2;
				echo -e "\n####### input files for bwa mem are specified in $fqfiles                          ###############\n" >&2;
				echo -e "\n####### we need to see if we need to split/not split alignemnt and dedup commands  ###############\n" >&2; set -x


				# logs will go to TopOutputLogs if launcher is used, otherwise they will go to sample/align/logs folder

				if [ $run_method == "LAUNCHER" ]
				then
					AlignOutputLogs=$TopOutputLogs/align
				else 
					AlignOutputLogs=$AlignOutputDir/logs			
				fi

				jobfileAlign=$AlignOutputLogs/bwamem.${SampleName}.node$OutputFileSuffix.jobfile
				jobfileMarkdup=$AlignOutputLogs/Markdup.${SampleName}.node$OutputFileSuffix.jobfile
				
				alignedbam=${outputsamfileprefix}.node${OutputFileSuffix}.sorted.bam
				dedupbam=${outputsamfileprefix}.node${OutputFileSuffix}.wdups.sorted.bam

				if [ $splitAlignDedup != "YES" ]
				then
					set +x; echo -e "################# CASE is NOT SPLIT, alignment and markduplicates in one job ############\n" >&2; set -x
					echo "$scriptdir/bwamem_and_markduplicates.sh $AlignOutputDir $fqfiles $alignedbam $dedupbam $RGparms  $runfile $AlignOutputLogs/log.bwamem.$SampleName.node$OutputFileSuffix.in $AlignOutputLogs/log.bwamem.$SampleName.node$OutputFileSuffix.ou $email $jobfileAlign $failedlog > $AlignOutputLogs/log.bwamem.$SampleName.node$OutputFileSuffix.in" > $jobfileAlign
					jobfilename=$( basename $jobfileAlign )
					echo "$AlignOutputLogs $jobfilename" >> $AlignOutputLogs/AlignAnisimov.joblist
				else
					set +x; echo -e "################# CASE is SPLIT, alignment in one job and markduplicates in another job ##\n" >&2; set -x

					echo "$scriptdir/bwamem_only.sh $AlignOutputDir $fqfiles $alignedbam $dedupbam $RGparms  $runfile $AlignOutputLogs/log.bwamem.$SampleName.node$OutputFileSuffix.in $AlignOutputLogs/log.bwamem.$SampleName.node$OutputFileSuffix.ou $email $jobfileAlign $failedlog > $AlignOutputLogs/log.bwamem.$SampleName.node$OutputFileSuffix.in" > $jobfileAlign
					jobfilename=$( basename $jobfileAlign )
					echo "$AlignOutputLogs $jobfilename" >> $AlignOutputLogs/AlignAnisimov.joblist

					echo "$scriptdir/markduplicates.sh $AlignOutputDir $fqfiles $alignedbam $dedupbam $RGparms  $runfile $AlignOutputLogs/log.Markdup.$SampleName.node$OutputFileSuffix.in $AlignOutputLogs/log.Markdup.$SampleName.node$OutputFileSuffix.ou $email $jobfileMarkdup $failedlog > $AlignOutputLogs/log.Markdup.$SampleName.node$OutputFileSuffix.in" > $jobfileMarkdup
					jobfilenameMarkdup=$( basename $jobfileMarkdup )
					echo "$AlignOutputLogs $jobfilenameMarkdup" >> $AlignOutputLogs/MarkdupsAnisimov.joblist
				fi                             


			fi # end bwa-mem

			if (( $i < 10 ))
			then
				allfiles=$allfiles" $outputsamfileprefix.node0$i.bam" # this will be used later for merging
			else
				allfiles=$allfiles" $outputsamfileprefix.node$i.bam" # this will be used later for merging
			fi
			echo `date`
		
		done # end loop over chunks of the current fastq

		set +x; echo -e "\n\n\n" >&2;
		echo -e "####################################################################################################################" >&2
		echo -e "#############                   end loop2 over chunks of the current fastq                     #####################" >&2
		echo -e "####################################################################################################################" >&2
		echo -e "#############               WE ARE STILL INSIDE THE LOOP OVER INPUT FASTQ!!!                   #####################" >&2
		echo -e "####################################################################################################################" >&2
		echo -e "\n\n\n" >&2; set -x


		set +x; echo -e "\n\n\n" >&2;
		echo -e "####################################################################################################################" >&2
		echo -e "############                                                                            ############################" >&2
		echo -e "############   FORM POST-ALIGNMENT QSUBS: MERGING, SORTING, MARKING DUPLICATES          ############################" >&2
		echo -e "############   SKIP THIS BLOCK IF READS WHERE NOT CHUNKED                               ############################" >&2
		echo -e "############                                                                            ############################" >&2
		echo -e "####################################################################################################################" >&2
		echo -e "\n\n\n" >&2; set -x

		if [ $chunkfastq == "YES" ]
		then
		       #ALIGNED=$( cat $AlignOutputLogs/ALIGNED_* | sed "s/\.[a-z]*//" | tr "\n" ":" )
		       #ALIGNED=$( cat $AlignOutputLogs/ALIGNED_* | sed "s/\..*//" | tr "\n" ":" )

			##############################################################
			#this section needs editing
			##############################################################


		       listfiles=$( echo $allfiles  | tr " " ":" | sed "s/::/:/g" )
		       if [ $sortool == "NOVOSORT" ]
		       then
				set +x; echo -e "\n # merging aligned chunks with novosort \n" >&2; set -x

				echo "$scriptdir/mergenovo.sh $AlignOutputDir $listfiles $outsortwdup $outsortnodup $sortedplain $dupparms $RGparms $runfile $AlignOutputLogs/log.sortmerge_novosort.$SampleName.in $AlignOutputLogs/log.sortmerge_novosort.$SampleName.ou $email $AlignOutputLogs/qsub.sortmerge.novosort.$SampleName" >> $AlignOutputDir/logs/mergenovo.${SampleName}
				#`chmod a+r $qsub_sortmerge`
				#mergejob=`qsub $qsub_sortmerge`
				#`qhold -h u $mergejob`
				#echo $mergejob  > $AlignOutputLogs/MERGED_$SampleName
		       else
				set +x; echo -e "\n # merging aligned chunks with picard \n" >&2; set -x 
				qsub_sortmerge=$AlignOutputLogs/qsub.sortmerge.picard.$SampleName
				echo "#PBS -A $pbsprj" >> $qsub_sortmerge
				echo "#PBS -N ${pipeid}_sortmerge_picard_$SampleName" >> $qsub_sortmerge
				echo "#PBS -l walltime=$pbscpu" >> $qsub_sortmerge
				echo "#PBS -l nodes=1:ppn=$thr" >> $qsub_sortmerge
				echo "#PBS -o $AlignOutputLogs/log.sortmerge.picard.$SampleName.ou" >> $qsub_sortmerge
				echo "#PBS -e $AlignOutputLogs/log.sortmerge.picard.$SampleName.in" >> $qsub_sortmerge
				echo "#PBS -q $pbsqueue" >> $qsub_sortmerge
				echo "#PBS -m ae" >> $qsub_sortmerge
				echo "#PBS -M $email" >> $qsub_sortmerge
				echo "#PBS -W depend=afterok:$ALIGNED" >> $qsub_sortmerge
				echo "aprun -n 1 -d $thr $scriptdir/mergepicard.sh $AlignOutputDir $listfiles $outsortwdup $outsortnodup $sortedplain $dupparms $RGparms $runfile $AlignOutputLogs/log.sortmerge.picard.$SampleName.in $AlignOutputLogs/log.sortmerge.picard.$SampleName.ou $email $AlignOutputLogs/qsub.sortmerge.picard.$SampleName" >> $qsub_sortmerge
				#`chmod a+r $qsub_sortmerge`
				mergejob=`qsub $qsub_sortmerge`
				`qhold -h u $mergejob`
				echo $mergejob  > $AlignOutputLogs/MERGED_$SampleName
		       fi  # end if sortool
		fi #end if chunkfastq
		(( inputfastqcounter++ )) # was not initialized at beginning of loop, so starts at zero
	fi # end if empty lines
done <  $TheInputFile # end loop over input fastq

set +x; echo -e "\n\n\n" >&2;
echo "######################################################################################################################" >&2
echo "####################################         end loop1 over input fastq              #################################" >&2
echo "####################################                                                 #################################" >&2
echo "######################################################################################################################" >&2
echo -e "\n\n" >&2; set -x


set +x; echo -e "\n\n\n" >&2
echo "#####################################################################################################################" >&2
echo "#####################################                                        ########################################" >&2
echo "#####################################  SCHEDULE JOBS                         ########################################" >&2
echo "#####################################  CASE1: NO CHUNKING OF INPUT FILES     ########################################" >&2
echo "#####################################################################################################################" >&2
echo -e "\n\n" >&2; set -x



if [ $chunkfastq == "NO" -a $aligner == "BWA_MEM" -a $paired -eq 1 ]
then
 
	set +x; echo -e "\n ### update autodocumentation script ### \n"; set -x;
	echo -e "# @begin $aligner" >> $outputdir/WorkflowAutodocumentationScript.sh
	echo -e "   # @in datafilestoalign @as fastqc_inputs" >> $outputdir/WorkflowAutodocumentationScript.sh
	AlignedFastqPathTemplate="SampleName/align/SampleName.node00.wdups.sorted.bam"
	echo -e "   # @out input_to_realrecal @as fastq_aligned_sorted_markdup @URI ${AlignedFastqPathTemplate}" >> $outputdir/WorkflowAutodocumentationScript.sh
	echo -e "# @end $aligner" >> $outputdir/WorkflowAutodocumentationScript.sh


	if [ $run_method == "LAUNCHER" ]
	then
		# increment so number of nodes = number fo input fastq + 1, even when there is only one input fastq
		# otherwise nowhere for launcher to run, due to the -N 1 option in aprun
		# 
		# In this segment of the code, we need to schedule a minimum of one job and a maximum of three jobs
		# case1: one job only when fastqc is skipped and align-dedup is a single job, no splitting
		# case2: two jobs when fastqc is not skipped and align-dedup is a single job, no splitting 
		# case3: three jobs when when fastqc is not skipped and align-dedup is two jobs, with splitting
		#
		# all logs, joblists, and qsub files for launchers will go to TopOutputLogs
		
		numalignnodes=$(( inputfastqcounter + 1))

		set +x; echo -e "\n # run_method is LAUNCHER. scheduling the FastQC Launcher\n" >&2; set -x

		AlignOutputLogs=$TopOutputLogs/align
		FastqcOutputLogs=$TopOutputLogs
		
		if [ $fastqcflag == "YES" ]
		then
                   
			#No dependencies will be made with this job. Hopefully, it will run last right before the summary script

			FastqcOutputLogs=$TopOutputLogs

			qsubFastqcLauncher=$FastqcOutputLogs/qsub.Fastqc.Anisimov
			echo "#!/bin/bash" > $qsubFastqcLauncher
			echo "#PBS -A $pbsprj" >> $qsubFastqcLauncher
			echo "#PBS -N ${pipeid}_Fastqc_Anisimov" >> $qsubFastqcLauncher
			echo "#PBS -l walltime=$pbscpu" >> $qsubFastqcLauncher
			echo "#PBS -l nodes=$numalignnodes:ppn=$thr" >> $qsubFastqcLauncher
			echo "#PBS -o $FastqcOutputLogs/log.Fastqc.Anisimov.ou" >> $qsubFastqcLauncher
			echo "#PBS -e $FastqcOutputLogs/log.Fastqc.Anisimov.in" >> $qsubFastqcLauncher
			echo "#PBS -q $pbsqueue" >> $qsubFastqcLauncher
			echo "#PBS -m ae" >> $qsubFastqcLauncher
			echo "#PBS -M $email" >> $qsubFastqcLauncher

			echo "$run_cmd $numalignnodes -env OMP_NUM_THREADS=$thr $launcherdir/scheduler.x $FastqcOutputLogs/FastqcAnisimov.joblist $bash_cmd > $FastqcOutputLogs/FastqcAnisimov.joblist.log" >> $qsubFastqcLauncher

			echo "exitcode=\$?" >> $qsubFastqcLauncher
			echo -e "if [ \$exitcode -ne 0 ]\nthen " >> $qsubFastqcLauncher
			echo "   echo -e \"\n\n FastqcAnisimov failed with exit code = \$exitcode \n logfile=$AlignOutputLogs/log.Fastqc.Anisimov.in\n\" | mail -s \"[Task #${reportticket}]\" \"$redmine,$email\"" >> $qsubFastqcLauncher
			echo "   exit 1" >> $qsubFastqcLauncher
			echo "fi" >> $qsubFastqcLauncher

			FastqcAnisimovJoblistId=`qsub $qsubFastqcLauncher`
			echo $FastqcAnisimovJoblistId >> $TopOutputLogs/pbs.FASTQC # so that this job could be released in the next section. Should it be held to begin with?
		fi              

		set +x; echo -e "\n # run_method is LAUNCHER. scheduling the Align Launcher\n" >&2; set -x

		qsubAlignLauncher=$AlignOutputLogs/qsub.align.Anisimov
		echo "#!/bin/bash" > $qsubAlignLauncher
		echo "#PBS -A $pbsprj" >> $qsubAlignLauncher
		echo "#PBS -N ${pipeid}_align_Anisimov" >> $qsubAlignLauncher
		echo "#PBS -l walltime=$pbscpu" >> $qsubAlignLauncher
		echo "#PBS -l nodes=$numalignnodes:ppn=$thr" >> $qsubAlignLauncher
		echo "#PBS -o $AlignOutputLogs/log.align.Anisimov.ou" >> $qsubAlignLauncher
		echo "#PBS -e $AlignOutputLogs/log.align.Anisimov.in" >> $qsubAlignLauncher
		echo "#PBS -q $pbsqueue" >> $qsubAlignLauncher
		echo "#PBS -m ae" >> $qsubAlignLauncher
		echo "#PBS -M $email" >> $qsubAlignLauncher

		echo "$run_cmd $numalignnodes -env OMP_NUM_THREADS=$thr $launcherdir/scheduler.x $AlignOutputLogs/AlignAnisimov.joblist $bash_cmd > $AlignOutputLogs/AlignAnisimov.joblist.log" >> $qsubAlignLauncher

		echo "exitcode=\$?" >> $qsubAlignLauncher
		echo -e "if [ \$exitcode -ne 0 ]\nthen " >> $qsubAlignLauncher
		echo "   echo -e \"\n\n AlignAnisimov failed with exit code = \$exitcode \n logfile=$AlignOutputLogs/log.AlignAnisimov.in\n\" | mail -s \"[Task #${reportticket}]\" \"$redmine,$email\"" >> $qsubAlignLauncher
		echo "   exit 1" >> $qsubAlignLauncher
		echo "fi" >> $qsubAlignLauncher

		AlignAnisimovJoblistId=`qsub $qsubAlignLauncher`
		echo $AlignAnisimovJoblistId >> $TopOutputLogs/pbs.ALIGNED # so that this job could be released in the next section. Should it be held to begin with?

		set +x; echo -e "\n # run_method is LAUNCHER. scheduling the MarkDuplicates Launcher\n" >&2; set -x   

		if [ $splitAlignDedup == "YES" ]
		then

			qsubMarkdupLauncher=$AlignOutputLogs/qsub.Markdup.Anisimov
			echo "#!/bin/bash" > $qsubMarkdupLauncher
			echo "#PBS -A $pbsprj" >> $qsubMarkdupLauncher
			echo "#PBS -N ${pipeid}_Markdup_Anisimov" >> $qsubMarkdupLauncher
			echo "#PBS -l walltime=$pbscpu" >> $qsubMarkdupLauncher
			echo "#PBS -l nodes=$numalignnodes:ppn=$thr" >> $qsubMarkdupLauncher
			echo "#PBS -o $AlignOutputLogs/log.Markdup.Anisimov.ou" >> $qsubMarkdupLauncher
			echo "#PBS -e $AlignOutputLogs/log.Markdup.Anisimov.in" >> $qsubMarkdupLauncher
			echo "#PBS -q $pbsqueue" >> $qsubMarkdupLauncher
			echo "#PBS -m ae" >> $qsubMarkdupLauncher
			echo "#PBS -M $email" >> $qsubMarkdupLauncher
			echo "#PBS -W depend=afterok:$AlignAnisimovJoblistId" >> $qsubMarkdupLauncher 

			echo "$run_cmd $numalignnodes -env OMP_NUM_THREADS=$thr $launcherdir/scheduler.x $AlignOutputLogs/MarkdupsAnisimov.joblist $bash_cmd > $AlignOutputLogs/MarkdupsAnisimov.joblist.log" >> $qsubMarkdupLauncher

			echo "exitcode=\$?" >> $qsubMarkdupLauncher
			echo -e "if [ \$exitcode -ne 0 ]\nthen " >> $qsubMarkdupLauncher
			echo "   echo -e \"\n\n MarkdupsAnisimov failed with exit code = \$exitcode \n logfile=$AlignOutputLogs/log.Markdup.Anisimov.in\n\" | mail -s \"[Task #${reportticket}]\" \"$redmine,$email\"" >> $qsubMarkdupLauncher
			echo "   exit 1" >> $qsubMarkdupLauncher
			echo "fi" >> $qsubMarkdupLauncher

			MarkDupAnisimovJoblistId=`qsub $qsubMarkdupLauncher`
			#`qhold -h u $MarkDupAnisimovJoblistId`
			#echo $MarkDupAnisimovJoblistId >> $TopOutputLogs/pbs.ALIGNED # so that this job could be released in the next section. Should it be held to begin with?
			echo $MarkDupAnisimovJoblistId >> $TopOutputLogs/pbs.MARKED # so that summaryok and start_realrecal_block.sh could depend on this job, in case when there is no merging: a sigle chunk                     
		fi
		# end case run_method=LAUNCHER

	else
		set +x; echo -e "\n # run_method is not LAUNCHER. scheduling qsubs inside a loop, one per sample\n" >&2; set -x

		# In this segment of the code, we need to schedule one-to-three jobs per sample
		# case1: one job  per sample only when fastqc is skipped and align-dedup is a single job, no splitting
		# case2: two jobs  per sample when fastqc is not skipped and align-dedup is a single job, no splitting 
		# case3: three jobs  per sample when when fastqc is not skipped and align-dedup is two jobs, with splitting
		#
		# all logs, joblists, and qsub files for launchers will go to either sample/align/logs  or   sample/fastqc/logs

		while read SampleLine
		do              
			set +x; echo -e "\n # run_method is not LAUNCHER. parsing $SampleLine and setup files and paths for qsub\n" >&2; set -x

            	 
			SampleName=$( echo -e "$SampleLine" | cut -f 1 )

			AlignOutputDir=$outputdir/$SampleName/align
			AlignOutputLogs=$AlignOutputDir/logs
			FastqcOutputDir=$outputdir/$SampleName/fastqc
			FastqcOutputLogs=$$outputdir/logs

			jobfileFastqc=$( ls  $FastqcOutputLogs/Fastqc.$SampleName.jobfile | tr "\n" " " )
			jobfileAlign=$( ls   $AlignOutputLogs/bwamem.${SampleName}.*.jobfile | tr "\n" " " )
			jobfileMarkdup=$( ls $AlignOutputLogs/Markdup.${SampleName}.*.jobfile | tr "\n" " " )

			
			set +x; echo -e "\n # run_method is not LAUNCHER. scheduling the FastQC qsub job\n" >&2; set -x

			if [ $fastqcflag == "YES" ]
			then

				#No dependencies will be made with this job. Hopefully, it will run last right before the summary script

				qsubFastqc=$FastqcOutputLogs/qsub.Fastqc.${SampleName}
				echo "#!/bin/bash" > $qsubFastqc
				echo "#PBS -A $pbsprj" >> $qsubFastqc
				echo "#PBS -N ${pipeid}_Fastqc_${SampleName}" >> $qsubFastqc
				echo "#PBS -l walltime=$pbscpu" >> $qsubFastqc
				echo "#PBS -l nodes=1:ppn=$thr" >> $qsubFastqc
				echo "#PBS -o $FastqcOutputLogs/log.Fastqc.${SampleName}.ou" >> $qsubFastqc
				echo "#PBS -e $FastqcOutputLogs/log.Fastqc.${SampleName}.in" >> $qsubFastqc
				echo "#PBS -q $pbsqueue" >> $qsubFastqc
				echo "#PBS -m ae" >> $qsubFastqc
				echo "#PBS -M $email" >> $qsubFastqc
				cat  $jobfileFastqc  >> $qsubFastqc
				echo "exitcode=\$?" >> $qsubFastqc
				echo -e "if [ \$exitcode -ne 0 ]\nthen " >> $qsubFastqc
				echo "   echo -e \"\n\n Fastqc_${SampleName} failed with exit code = \$exitcode \n logfile=$FastqcOutputLogs/log.Fastqc.${SampleName}.in\n\" | mail -s \"[Task #${reportticket}]\" \"$redmine,$email\"" >> $qsubFastqc
				echo "   exit 1" >> $qsubFastqc
				echo "fi" >> $qsubFastqc

				FastqcJobId=`qsub $qsubFastqc`
				echo $FastqcJobId >> $TopOutputLogs/pbs.FASTQC # so that this job could be released in the next section. Should it be held to begin with?
			fi              


			set +x; echo -e "\n # run_method is not LAUNCHER. scheduling alignment qsubs\n" >&2; set -x

			qsubAlign=$AlignOutputLogs/qsub.align.${SampleName}
			echo "#!/bin/bash" > $qsubAlign
			echo "#PBS -A $pbsprj" >> $qsubAlign
			echo "#PBS -N ${pipeid}_align_${SampleName}" >> $qsubAlign
			echo "#PBS -l walltime=$pbscpu" >> $qsubAlign
			echo "#PBS -l nodes=1:ppn=$thr" >> $qsubAlign
			echo "#PBS -o $AlignOutputLogs/log.align.${SampleName}.ou" >> $qsubAlign
			echo "#PBS -e $AlignOutputLogs/log.align.${SampleName}.in" >> $qsubAlign
			echo "#PBS -q $pbsqueue" >> $qsubAlign
			echo "#PBS -m ae" >> $qsubAlign
			echo "#PBS -M $email" >> $qsubAlign
			cat $jobfileAlign >> $qsubAlign
			echo "exitcode=\$?" >> $qsubAlign
			echo -e "if [ \$exitcode -ne 0 ]\nthen " >> $qsubAlign
			echo "   echo -e \"\n\n Align${SampleName} failed with exit code = \$exitcode \n logfile=$AlignOutputLogs/log.Align${SampleName}.in\n\" | mail -s \"[Task #${reportticket}]\" \"$redmine,$email\"" >> $qsubAlign
			echo "   exit 1" >> $qsubAlign
			echo "fi" >> $qsubAlign

			AlignJobId=`qsub $qsubAlign`
			
			echo $AlignJobId >> $TopOutputLogs/pbs.ALIGNED # so that this job could be released in the next section. Should it be held to begin with?

			set +x; echo -e "\n # run_method is not LAUNCHER. scheduling the MarkDuplicates Launcher\n" >&2; set -x   

			if [ $splitAlignDedup == "YES" ]
			then
				qsubMarkdup=$AlignOutputLogs/qsub.Markdup.${SampleName}
				echo "#!/bin/bash" > $qsubMarkdup
				echo "#PBS -A $pbsprj" >> $qsubMarkdup
				echo "#PBS -N ${pipeid}_Markdup_${SampleName}" >> $qsubMarkdup
				echo "#PBS -l walltime=$pbscpu" >> $qsubMarkdup
				echo "#PBS -l nodes=1:ppn=$thr" >> $qsubMarkdup
				echo "#PBS -o $AlignOutputLogs/log.Markdup.${SampleName}.ou" >> $qsubMarkdup
				echo "#PBS -e $AlignOutputLogs/log.Markdup.${SampleName}.in" >> $qsubMarkdup
				echo "#PBS -q $pbsqueue" >> $qsubMarkdup
				echo "#PBS -m ae" >> $qsubMarkdup
				echo "#PBS -M $email" >> $qsubMarkdup
				echo "#PBS -W depend=afterok:$AlignJobId" >> $qsubMarkdup 
				cat $jobfileMarkdup >> $qsubMarkdup
				echo "exitcode=\$?" >> $qsubMarkdup
				echo -e "if [ \$exitcode -ne 0 ]\nthen " >> $qsubMarkdup
				echo "   echo -e \"\n\n Markdups_${SampleName} failed with exit code = \$exitcode \n logfile=$AlignOutputLogs/log.Markdup.${SampleName}.in\n\" | mail -s \"[Task #${reportticket}]\" \"$redmine,$email\"" >> $qsubMarkdup
				echo "   exit 1" >> $qsubMarkdup
				echo "fi" >> $qsubMarkdup

				MarkDupJobId=`qsub $qsubMarkdup`
				echo $MarkDupJobId >> $TopOutputLogs/pbs.MARKED # so that summaryok and start_realrecal_block.sh could depend on this job, in case when there is no merging: a sigle chunk                     
			fi

		done < $TheInputFile # done looping over samples
		# end case run_method=!LAUNCHER              
	fi  ## end if run_method
fi ## end if bwa_mem and non-split input


set +x; echo -e "\n\n\n" >&2
echo "#####################################################################################################################" >&2
echo "#####################################  SCHEDULE JOBS                         ########################################" >&2
echo "#####################################  CASE2: CHUNKING OF INPUT FILES        ########################################" >&2
echo "#####################################################################################################################" >&2
echo -e "\n\n" >&2; set -x

if [ $chunkfastq == "YES" -a $aligner == "NOVOALIGN" -a $sortool == "NOVOSORT" ]
then

	set +x; echo -e "\n ### update autodocumentation script ### \n"; set -x;
	echo -e "# @begin chunk_fastq" >> $outputdir/WorkflowAutodocumentationScript.sh
	echo -e "   # @in datafilestoalign @as fastqc_inputs" >> $outputdir/WorkflowAutodocumentationScript.sh
	InputFastqPathTemplate="SampleName/align/{left/right}reads_chunk{number}"
	echo -e "   # @out chunks @as chunked_fastq @URI ${InputFastqPathTemplate}" >> $outputdir/WorkflowAutodocumentationScript.sh
	echo -e "# @end $chunk_fastq" >> $outputdir/WorkflowAutodocumentationScript.sh

	echo -e "# @begin ${aligner}" >> $outputdir/WorkflowAutodocumentationScript.sh
	echo -e "   # @in chunks @as chunked_fastq " >> $outputdir/WorkflowAutodocumentationScript.sh
	AlignedFastqPathTemplate="/SampleName/align/{left/right}reads.node{number}.bam"
	echo -e "   # @out alignment_output @as aligned_fastq @URI ${AlignedFastqPathTemplate}" >> $outputdir/WorkflowAutodocumentationScript.sh
	echo -e "# @end ${aligner}" >> $outputdir/WorkflowAutodocumentationScript.sh

	echo -e "# @begin merge_${sortool}_${markduplicatestool}" >> $outputdir/WorkflowAutodocumentationScript.sh
	echo -e "   # @in alignment_output @as aligned_fastq" >> $outputdir/WorkflowAutodocumentationScript.sh
	MergedFastqPathTemplate="SampleName/align/SampleName.wdups.sorted.bam"
	echo -e "   # @out input_to_realrecal @as fastq_aligned_sorted_markdup @URI ${MergedFastqPathTemplate}" >> $outputdir/WorkflowAutodocumentationScript.sh
	echo -e "# @end merge_${sortool}_${markduplicatestool}" >> $outputdir/WorkflowAutodocumentationScript.sh

	##############################################################
	#this section needs editing
	##############################################################
	
	case $run_method in
	"LAUNCHER")
              # clear out the joblists
              truncate -s 0 $AlignOutputLogs/novosplit.AnisimovJoblist
              truncate -s 0 $AlignOutputLogs/mergenovo.AnisimovJoblist

              while read SampleName
              do
            	 if [ ! -s $outputdir/SAMPLENAMES_multiplexed.list ]
                 then
                     SampleName=$( echo $SampleLine )
	         else
                     SampleName=$( echo -e "$SampleLine" | cut -f 2 )
                 fi
                 AlignOutputDir=$outputdir/${SampleName}/align
                 
                 for i in $(seq 0 $NumChunks)
                 do

                    if (( $i < 10 ))
                    then
                       OutputFileSuffix=0${i}
                    else
                       OutputFileSuffix=${i}
                    fi

                    # creating a qsub out of the job file
                    # need to prepend "nohup" and append log file name, so that logs are properly created when Anisimov launches these jobs
                    novosplit_log=${AlignOutputDir}/logs/log.novosplit.${SampleName}.node$OutputFileSuffix.in
                    awk -v awkvar_novosplitlog=$novosplit_log '{print "nohup "$0" > "awkvar_novosplitlog}' $AlignOutputDir/logs/novosplit.${SampleName}.node${OutputFileSuffix} > $AlignOutputDir/logs/jobfile.novosplit.${SampleName}.node${OutputFileSuffix}
                    echo "$AlignOutputDir/logs/ jobfile.novosplit.${SampleName}.node${OutputFileSuffix}" >> $AlignOutputLogs/novosplit.AnisimovJoblist
                 done # done going through chunks

                 mergenovo_log=${AlignOutputDir}/logs/log.mergenovo.${SampleName}.in
                 awk -v awkvar_mergenovolog=$mergenovo_log '{print "nohup "$0" > "awkvar_mergenovolog}' $AlignOutputDir/logs/mergenovo.${SampleName} > $AlignOutputDir/logs/jobfile.mergenovo.${SampleName}
                 echo "$AlignOutputDir/logs/ jobfile.mergenovo.${SampleName}" >> $AlignOutputLogs/mergenovo.AnisimovJoblist
              done < $TheInputFile # done looping over samples


              set +x; echo -e "\n\n ########### scheduling Anisimov Launcher joblists #############\n\n" >&2; set -x
              qsub_novosplit_anisimov=$AlignOutputLogs/qsub.novosplit.AnisimovLauncher
              qsub_mergenovo_anisimov=$AlignOutputLogs/qsub.mergenovo.AnisimovLauncher

              # appending the generic header to the qsub
              cat $outputdir/qsubGenericHeader > $qsub_novosplit_anisimov
              cat $outputdir/qsubGenericHeader > $qsub_mergenovo_anisimov


              ###############################
              set +x; echo -e "\n ################# constructing qsub for novosplit\n" >&2; set -x
              echo "#PBS -N ${pipeid}_novoalign_Anisimov" >> $qsub_novosplit_anisimov
              echo "#PBS -l walltime=$pbscpu" >> $qsub_novosplit_anisimov
              echo "#PBS -o $AlignOutputLogs/log.novosplit.Anisimov.ou" >> $qsub_novosplit_anisimov
              echo "#PBS -e $AlignOutputLogs/log.novosplit.Anisimov.in" >> $qsub_novosplit_anisimov

              # number of nodes required for alignment is equal to the number of samples times number of chunks into which every sample is broken
              # counter started at zero, so in the end reflects the true number of fsatq samples
              # for some stupid reason NumChunks is actually from 0 to number_of_chunks-1, so we have to increment it now
              (( NumChunks++ ))
              NumberOfNodes=$(( inputfastqcounter * NumChunks )) 
              (( NumberOfNodes++ )) # plus one for the launcher
              echo -e "#PBS -l nodes=$NumberOfNodes:ppn=$thr\n" >> $qsub_novosplit_anisimov
              echo "$eun_cmd $NumberOfNodes -env OMP_NUM_THREADS=$thr ~anisimov/scheduler/scheduler.x $AlignOutputLogs/novosplit.AnisimovJoblist $bash_cmd > $AlignOutputLogs/novosplit.AnisimovLauncher.log" >> $qsub_novosplit_anisimov

              ### iForge - specific use of launcher - not sure how to handle the switch, so keeping it commented out for now
              ### echo "module load intel/12.0.4" >> $qsub_novosplit_anisimov
              ### echo "module load openmpi-1.4.3-intel-12.0.4" >> $qsub_novosplit_anisimov
              #echo "cat $PBS_NODEFILE | sort -u | awk -v n=1 '{for(i=0;i<n;i++) print \$0}' > ${AlignOutputLogs}/HOSTLIST" >> $qsub_novosplit_anisimov
              ### echo "mpiexec -n $NumberOfNodes  --pernode -machinefile \$PBS_NODEFILE --mca btl tcp,self  ${launcherdir}/scheduler.x $AlignOutputLogs/novosplit.AnisimovJoblist $bash_cmd > $AlignOutputLogs/novosplit.AnisimovLauncher.log" >> $qsub_novosplit_anisimov

              novosplit_job=`qsub $qsub_novosplit_anisimov`
              `qhold -h u $novosplit_job`
              echo $novosplit_job >> $TopOutputLogs/pbs.ALIGNED # so that this job could be released in the next section. Should it be held to begin with?


              ###############################
              set +x; echo -e "\n ################# constructing qsub for mergenovo\n" >&2; set -x
              echo "#PBS -N ${pipeid}_mergenovo_Anisimov" >> $qsub_mergenovo_anisimov
              echo "#PBS -l walltime=$pbscpu" >> $qsub_mergenovo_anisimov
              echo "#PBS -o $AlignOutputLogs/log.mergenovo.Anisimov.ou" >> $qsub_mergenovo_anisimov
              echo "#PBS -e $AlignOutputLogs/log.mergenovo.Anisimov.in" >> $qsub_mergenovo_anisimov
              # add dependency on novosplit job
              echo -e "#PBS -W depend=afterok:$novosplit_job" >> $qsub_mergenovo_anisimov

              # number of nodes required for mergenovo is equal to the number of samples plus one for the launcher
              NumberOfNodes=$(( inputfastqcounter + 1 )) # counter started at zero, so in the end reflects the true number of fsatq samples
              echo -e "#PBS -l nodes=$NumberOfNodes:ppn=$thr\n" >> $qsub_mergenovo_anisimov
              echo "$run_cmd $NumberOfNodes -env OMP_NUM_THREADS=$thr $launcherdir/scheduler.x $AlignOutputLogs/mergenovo.AnisimovJoblist $bash_cmd > $AlignOutputLogs/mergenovo.AnisimovLauncher.log" >> $qsub_mergenovo_anisimov

              mergenovo_job=`qsub $qsub_mergenovo_anisimov`
              `qhold -h u $mergenovo_job`
              echo $mergenovo_job > $TopOutputLogs/pbs.MERGED # so that this job could be released in the next section, and start_realrecal_block.sh could depend on it 

           ;;           
           "APRUN")
              while read SampleName
              do
            	 if [ ! -s $outputdir/SAMPLENAMES_multiplexed.list ]
                 then
                     SampleName=$( echo $SampleLine )
	         else
                     SampleName=$( echo -e "$SampleLine" | cut -f 2 )
                 fi
                 AlignOutputDir=$outputdir/${SampleName}/align

                 # ADD LOOP OVER CHUNKS
                 for i in $(seq 0 $NumChunks)
                 do

                    if (( $i < 10 ))
                    then
                       OutputFileSuffix=0${i}
                    else
                       OutputFileSuffix=${i}
                    fi

                    set +x; echo -e "\n # scheduling qsubs\n" >&2; set -x
                    qsub_novosplit=$AlignOutputDir/logs/qsub.novosplit.${SampleName}.node${OutputFileSuffix}
                    # appending the generic header to the qsub
                    cat $outputdir/qsubGenericHeader > $qsub_novosplit


                    ###############################
                    set +x; echo -e "\n # constructing qsub for novosplit\n" >&2; set -x
                    echo "#PBS -N ${pipeid}_novoalign.${SampleName}.node${OutputFileSuffix}" >> $qsub_novosplit
                    echo "#PBS -l walltime=$pbscpu" >> $qsub_novosplit
                    echo "#PBS -o $AlignOutputDir/logs/log.novosplit.${SampleName}.node${OutputFileSuffix}.ou" >> $qsub_novosplit
                    echo "#PBS -e $AlignOutputDir/logs/log.novosplit.${SampleName}.node${OutputFileSuffix}.in" >> $qsub_novosplit
                    echo -e "#PBS -l nodes=1:ppn=$thr\n" >> $qsub_novosplit

                    # using sed to edit the first and only line in each jobfile to add the relevant scheduling commands
                    sed "1!b;s/^/$run_cmd 1 -N 1 -d $thr /" $AlignOutputDir/logs/novosplit.${SampleName}.node$OutputFileSuffix >> $qsub_novosplit 

                    novosplit_job=`qsub $qsub_novosplit`
                    `qhold -h u $novosplit_job`
                    echo $novosplit_job >> $TopOutputLogs/pbs.ALIGNED_${SampleName} # so that this job could be released in the next section. Should it be held to begin with?

                 done # done looping over chunks of a sample
                 cat $AlignOutputLogs/ALIGNED_$SampleName >> $TopOutputLogs/pbs.ALIGNED
                 alignids=$( cat $TopOutputLogs/ALIGNED_$SampleName | sed "s/\..*//" | tr "\n" ":" )

                 qsub_mergenovo=$AlignOutputDir/logs/qsub.mergenovo.${SampleName}
                 # appending the generic header to the qsub
                 cat $outputdir/qsubGenericHeader > $qsub_mergenovo




                 ###############################
                 set +x; echo -e "\n # constructing qsub for mergenovo\n" >&2; set -x
                 echo "#PBS -N ${pipeid}_mergenovo" >> $qsub_mergenovo
                 echo "#PBS -l walltime=$pbscpu" >> $qsub_mergenovo
                 echo "#PBS -o $AlignOutputLogs/log.mergenovo.ou" >> $qsub_mergenovo
                 echo "#PBS -e $AlignOutputLogs/log.mergenovo.in" >> $qsub_mergenovo
                 # add dependency on novosplit job
                 echo -e "#PBS -W depend=afterok:$alignids" >> $qsub_mergenovo
                 echo -e "#PBS -l nodes=1:ppn=$thr\n" >> $qsub_mergenovo

                 # using sed to edit the first and only line in each jobfile to add the relevant scheduling commands
                 sed "1!b;s/^/$run_cmd 1 -N 1 -d $thr /" $AlignOutputDir/logs/mergenovo.${SampleName} >> $qsub_mergenovo 

                 mergenovo_job=`qsub $qsub_mergenovo`
                 `qhold -h u $mergenovo_job`
                 echo $mergenovo_job > $TopOutputLogs/pbs.MERGED # so that this job could be released in the next section, and start_realrecal_block.sh could depend on it

                 # release all held novosplit jobs for this sample
                 `qrls -h u $alignids`

              done < $TheInputFile # done looping over samples
           ;;
           "QSUB")
              while read SampleName
              do
            	 if [ ! -s $outputdir/SAMPLENAMES_multiplexed.list ]
                 then
                     SampleName=$( echo $SampleLine )
	         else
                     SampleName=$( echo -e "$SampleLine" | cut -f 2 )
                 fi
                 AlignOutputDir=$outputdir/${SampleName}/align


                 for i in $(seq 0 $NumChunks)
                 do

                    if (( $i < 10 ))
                    then
                       OutputFileSuffix=0${i}
                    else
                       OutputFileSuffix=${i}
                    fi

                    set +x; echo -e "\n # scheduling qsubs\n" >&2; set -x
                    qsub_novosplit=$AlignOutputDir/logs/qsub.novosplit.${SampleName}.node${OutputFileSuffix}

                    # appending the generic header to the qsub
                    cat $outputdir/qsubGenericHeader > $qsub_novosplit


                    ###############################
                    set +x; echo -e "\n # constructing qsub for novosplit\n" >&2; set -x
                    echo "#PBS -N ${pipeid}_novoalign.${SampleName}.node${OutputFileSuffix}" >> $qsub_novosplit
                    echo "#PBS -l walltime=$pbscpu" >> $qsub_novosplit
                    echo "#PBS -o $AlignOutputDir/logs/log.novosplit.${SampleName}.node${OutputFileSuffix}.ou" >> $qsub_novosplit
                    echo "#PBS -e $AlignOutputDir/logs/log.novosplit.${SampleName}.node${OutputFileSuffix}.in" >> $qsub_novosplit
                    echo -e "#PBS -l nodes=1:ppn=$thr\n" >> $qsub_novosplit
                    cat $AlignOutputDir/logs/novosplit.${SampleName}.node${OutputFileSuffix} >> $qsub_novosplit
                    novosplit_job=`qsub $qsub_novosplit`
                    `qhold -h u $novosplit_job`
                    echo $novosplit_job >> $TopOutputLogs/pbs.ALIGNED # so that this job could be released in the next section. Should it be held to begin with?

                 done # done looping over chunks of a sample
                 cat $AlignOutputLogs/ALIGNED_$SampleName >> $TopOutputLogs/pbs.ALIGNED
                 alignids=$( cat $TopOutputLogs/ALIGNED_$SampleName | sed "s/\..*//" | tr "\n" ":" )

                 qsub_mergenovo=$AlignOutputDir/logs/qsub.mergenovo.${SampleName}
                 # appending the generic header to the qsub
                 cat $outputdir/qsubGenericHeader > $qsub_mergenovo


                 ###############################
                 set +x; echo -e "\n # constructing qsub for mergenovo\n" >&2; set -x
                 echo "#PBS -N ${pipeid}_mergenovo" >> $qsub_mergenovo
                 echo "#PBS -l walltime=$pbscpu" >> $qsub_mergenovo
                 echo "#PBS -o $AlignOutputLogs/log.mergenovo.ou" >> $qsub_mergenovo
                 echo "#PBS -e $AlignOutputLogs/log.mergenovo.in" >> $qsub_mergenovo
                 # add dependency on novosplit job
                 echo -e "#PBS -W depend=afterok:$alignids" >> $qsub_mergenovo
                 echo -e "#PBS -l nodes=1:ppn=$thr\n" >> $qsub_mergenovo

                 # using sed to edit the first and only line in each jobfile to add the relevant scheduling commands
                 sed "1!b;s/^/$run_cmd 1 -env OMP_NUM_THREADS=$thr /" $AlignOutputDir/logs/mergenovo.${SampleName} >> $qsub_mergenovo

                 mergenovo_job=`qsub $qsub_mergenovo`
                 `qhold -h u $mergenovo_job`
                 echo $mergenovo_job > $TopOutputLogs/pbs.MERGED # so that this job could be released in the next section, and start_realrecal_block.sh could depend on it

                 # release all held novosplit jobs for this sample
                 `qrls -h u $alignids`


              done < $TheInputFile # done looping over samples
           ;;
           esac
fi


set +x; echo -e "\n\n\n" >&2
echo "#####################################################################################################################" >&2
echo "###############     WRAP UP ALIGNMENT BLOCK                                  ########################################" >&2
echo "###############     ALL QSUB SCRIPTS BELOW WILL RUN AFTER ALIGNMENT IS DONE  ########################################" >&2
echo "#####################################################################################################################" >&2
echo -e "\n\n" >&2; set -x;

markedids=$( cat "$TopOutputLogs/pbs.MARKED" | sed "s/\.[a-zA-Z]*//" | tr "\n" " " ) #for release comd
mergeids=$( cat "$TopOutputLogs/pbs.MERGED" | sed "s/\.[a-zA-Z]*//" | tr "\n" " " )  #for release comd
alignids=$( cat "$TopOutputLogs/pbs.ALIGNED" | sed "s/\.[a-zA-Z]*//" | tr "\n" " " ) #for release comd
fastqcids=$( cat "$TopOutputLogs/pbs.FASTQC" | sed "s/\.[a-zA-Z]*//" | tr "\n" " " ) #for release comd
#pbsids=$( cat "$TopOutputLogs/pbs.MARKED" "$TopOutputLogs/pbs.ALIGNED" | sed "s/\.[a-zA-Z]*//" | tr "\n" ":" | sed "s/^://" | sed "s/:$//" ) #for job dependency argument
pbsids=$( cat "$TopOutputLogs/pbs.MARKED" "$TopOutputLogs/pbs.ALIGNED" | tr "\n" ":" | sed "s/^://" | sed "s/:$//" ) #for job dependency argument

## generating summary redmine email if analysis ends here
set +x; echo -e "\n # wrap up and produce summary table if analysis ends here or call realign if analysis continues \n" >&2; set -x;

if [ $analysis == "ALIGNMENT" -o $analysis == "ALIGN" -o $analysis == "ALIGN_ONLY" ]
then
	set +x; echo -e "\n ###### ANALYSIS = $analysis ends here. Wrapping up and quitting\n" >&2; set -x;
	# release all held jobs
	`qrls -h u $alignids`
	`qrls -h u $mergeids`
	`qrls -h u $fastqcids`
	`qrls -h u $markedids`    

	lastjobid=""
	cleanjobid=""

	if [ $cleanupflag == "YES" ]
	then 
		set +x; echo -e "\n ###### Removing temporary files  ######\n" >&2; set -x;
		qsub_cleanup=$TopOutputLogs/qsub.cleanup.align
		echo "#PBS -A $pbsprj" >> $qsub_cleanup
		echo "#PBS -N ${pipeid}_cleanup_aln" >> $qsub_cleanup
		echo "#PBS -l walltime=$pbscpu" >> $qsub_cleanup
		echo "#PBS -l nodes=1:ppn=1" >> $qsub_cleanup
		echo "#PBS -o $TopOutputLogs/log.cleanup.align.ou" >> $qsub_cleanup
		echo "#PBS -e $TopOutputLogs/log.cleanup.align.in" >> $qsub_cleanup
		echo "#PBS -q $pbsqueue" >> $qsub_cleanup
		echo "#PBS -m ae" >> $qsub_cleanup
		echo "#PBS -M $email" >> $qsub_cleanup
		echo "#PBS -W depend=afterok:$pbsids" >> $qsub_cleanup
		echo "$scriptdir/cleanup.sh $outputdir $analysis $TopOutputLogs/log.cleanup.align.in $TopOutputLogs/log.cleanup.align.ou $email $TopOutputLogs/qsub.cleanup.align"  >> $qsub_cleanup
		#`chmod a+r $qsub_cleanup`
		cleanjobid=`qsub $qsub_cleanup`
		echo $cleanjobid >> $outputdir/logs/pbs.CLEANUP
	fi

	`sleep 10s`  # to avoid choking the scheduler
	
	set +x; echo -e "\n ###### Generating Summary report   ######\n" >&2; set -x;
	qsub_summary=$TopOutputLogs/qsub.summary.aln.allok
	echo "#PBS -A $pbsprj" >> $qsub_summary
	echo "#PBS -N ${pipeid}_summaryok" >> $qsub_summary
	echo "#PBS -l walltime=01:00:00" >> $qsub_summary # 1 hour should be more than enough
	echo "#PBS -l nodes=1:ppn=1" >> $qsub_summary
	echo "#PBS -o $TopOutputLogs/log.summary.aln.ou" >> $qsub_summary
	echo "#PBS -e $TopOutputLogs/log.summary.aln.in" >> $qsub_summary
	echo "#PBS -q $pbsqueue" >> $qsub_summary
	echo "#PBS -m ae" >> $qsub_summary
	echo "#PBS -M $email" >> $qsub_summary
	echo "#PBS -W depend=afterok:$pbsids" >> $qsub_summary
	echo "$scriptdir/summary.sh $runfile $email exitok $reportticket"  >> $qsub_summary
	#`chmod a+r $qsub_summary`
	lastjobid=`qsub $qsub_summary`
	echo $lastjobid >> $TopOutputLogs/pbs.SUMMARY

	# if at least one job failed, the summary script will be executed anyway with the following bock

	qsub_summaryany=$TopOutputLogs/qsub.summary.aln.afterany
	echo "#PBS -A $pbsprj" >> $qsub_summaryany
	echo "#PBS -N ${pipeid}_summary_afterany" >> $qsub_summaryany
	echo "#PBS -l walltime=01:00:00" >> $qsub_summaryany # 1 hour should be more than enough
	echo "#PBS -l nodes=1:ppn=1" >> $qsub_summaryany
	echo "#PBS -o $TopOutputLogs/log.summary.aln.afterany.ou" >> $qsub_summaryany
	echo "#PBS -e $TopOutputLogs/log.summary.aln.afterany.in" >> $qsub_summaryany
	echo "#PBS -q $pbsqueue" >> $qsub_summaryany
	echo "#PBS -m ae" >> $qsub_summaryany
	echo "#PBS -M $email" >> $qsub_summaryany
	echo "#PBS -W depend=afternotok:$lastjobid" >> $qsub_summaryany
	echo "$scriptdir/summary.sh $runfile $email exitnotok $reportticket"  >> $qsub_summaryany
	#`chmod a+r $qsub_summaryany`
	badjobid=`qsub $qsub_summaryany`
	echo $badjobid >> $TopOutputLogs/pbs.SUMMARY

fi

if [ $analysis == "REALIGNMENT" -o $analysis == "REALIGN" -o $analysis == "MULTIPLEXED" ]
then
	set +x; echo -e "\n ###### ANALYSIS == $analysis. Pipeline execution continues with realignment   ###### \n" >&2; set -x;
	qsub_realign=$TopOutputLogs/qsub.start_realrecal_block
	echo "#PBS -A $pbsprj" >> $qsub_realign
	echo "#PBS -N ${pipeid}_START_REALRECAL_BLOCK" >> $qsub_realign
	echo "#PBS -l walltime=01:00:00" >> $qsub_realign # 1 hour should be more than enough
	echo "#PBS -l nodes=1:ppn=1" >> $qsub_realign
	echo "#PBS -o $TopOutputLogs/start_realrecal_block.ou" >> $qsub_realign
	echo "#PBS -e $TopOutputLogs/start_realrecal_block.in" >> $qsub_realign
	echo "#PBS -q $pbsqueue" >> $qsub_realign
	echo "#PBS -m ae" >> $qsub_realign
	echo "#PBS -W depend=afterok:$pbsids" >> $qsub_realign
	echo "#PBS -M $email" >> $qsub_realign
	echo "$scriptdir/start_realrecal_block.sh $runfile $TopOutputLogs/start_realrecal_block.in $TopOutputLogs/start_realrecal_block.ou $email $TopOutputLogs/qsub.start_realrecal_block" >> $qsub_realign
	#`chmod a+r $qsub_realign` 
	`qsub $qsub_realign >> $TopOutputLogs/pbs.SCHREAL`

	# need to release jobs here or realignment will not start
	`qrls -h u $alignids`
	`qrls -h u $mergeids`
	`qrls -h u $fastqcids`
	`qrls -h u $markedids`

	echo `date`
fi

#`chmod -R 770 $AlignOutputDir`
#`chmod -R 770 $TopOutputLogs`
echo `date`

