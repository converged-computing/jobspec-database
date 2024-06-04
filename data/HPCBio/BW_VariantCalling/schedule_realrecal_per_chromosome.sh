#!/bin/bash
# written in collaboration with Mayo Bioinformatics core group
#
#  script to realign and recalibrate the aligned file(s)
#redmine=hpcbio-redmine@igb.illinois.edu
redmine=grendon@illinois.edu
if [ $# != 6 ]
then
   MSG="parameter mismatch."
   echo -e "program=$0 stopped. Reason=$MSG" | mail -s 'Variant Calling Workflow failure message' "$redmine"
   exit 1;
fi

umask 0027
set -x
echo `date`
scriptfile=$0
outputdir=$1
runfile=$2
elog=$3
olog=$4
email=$5
qsubfile=$6
LOGS="jobid:${PBS_JOBID}\nqsubfile=$qsubfile\nerrorlog=$elog\noutputlog=$olog"

if [ ! -s $runfile ]
then
   MSG="$runfile configuration file not found"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s 'Variant Calling Workflow failure message' "$redmine,$email"
   exit 1;
fi



set +x; echo -e "\n\n" >&2; 
echo -e "####################################################################################################" >&2
echo -e "#####################################                       ########################################" >&2
echo -e "##################################### PARSING RUN INFO FILE ########################################" >&2
echo -e "##################################### AND SANITY CHECK      ########################################" >&2
echo -e "####################################################################################################" >&2
echo -e "\n\n" >&2; set -x;




reportticket=$( cat $runfile | grep -w REPORTTICKET | cut -d '=' -f2 )
pbsprj=$( cat $runfile | grep -w PBSPROJECTID | cut -d '=' -f2 )
thr=$( cat $runfile | grep -w PBSTHREADS | cut -d '=' -f2 )
input_type=$( cat $runfile | grep -w INPUTTYPE | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
analysis=$( cat $runfile | grep -w ANALYSIS | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
run_method=$( cat $runfile | grep -w RUNMETHOD | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
launchercmd=$( cat $runfile | grep -w LAUNCHERCMD | cut -d '=' -f2 )
outputdir=$( cat $runfile | grep -w OUTPUTDIR | cut -d '=' -f2 )
scriptdir=$( cat $runfile | grep -w SCRIPTDIR | cut -d '=' -f2 )
refdir=$( cat $runfile | grep -w REFGENOMEDIR | cut -d '=' -f2 )
ref=$( cat $runfile | grep -w REFGENOME | cut -d '=' -f2 )
picardir=$( cat $runfile | grep -w PICARDIR | cut -d '=' -f2 )
samdir=$( cat $runfile | grep -w SAMDIR | cut -d '=' -f2 )
gatk=$( cat $runfile | grep -w GATKDIR | cut -d '=' -f2 )
tabixdir=$( cat $runfile | grep -w TABIXDIR | cut -d '=' -f2 )
vcftoolsdir=$( cat $runfile | grep -w VCFTOOLSDIR | cut -d '=' -f2 )
dbSNP=$( cat $runfile | grep -w DBSNP | cut -d '=' -f2 )
indeldir=$( cat $runfile | grep -w INDELDIR | cut -d '=' -f2 )
indelfile=$( cat $runfile | grep -w INDELFILE | cut -d '=' -f2 )
snpdir=$( cat $runfile | grep -w SNPDIR | cut -d '=' -f2 )
targetdir=$( cat $runfile | grep -w TARGETREGIONSDIR | cut -d '=' -f2 )
targetfile=$( cat $runfile | grep -w TARGETREGIONSFILE | cut -d '=' -f2 )
realignparams=$( cat $runfile | grep -w REALIGNPARMS | cut -d '=' -f2 )
omnisites=$( cat $runfile | grep -w OMNISITES | cut -d '=' -f2 ) 
recalibrator=$( cat $runfile | grep -w RECALIBRATOR | cut -d '=' -f2 )
runverify=$( cat $runfile | grep -w RUNVERIFYBAM | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
multisample=$( cat $runfile | grep -w MULTISAMPLE | cut -d '=' -f2 )
samples=$( cat $runfile | grep -w SAMPLENAMES | cut -d '=' -f2 )
chrindex=$( cat $runfile | grep -w CHRINDEX | cut -d '=' -f2 )
indices=$( echo $chrindex | sed 's/:/ /g' )
sPL=$( cat $runfile | grep -w SAMPLEPL | cut -d '=' -f2 )
sCN=$( cat $runfile | grep -w SAMPLECN | cut -d '=' -f2 )
sLB=$( cat $runfile | grep -w SAMPLELB | cut -d '=' -f2 )
javadir=$( cat $runfile | grep -w JAVADIR | cut -d '=' -f2 )
skipvcall=$( cat $runfile | grep -w SKIPVCALL | cut -d '=' -f2 )
cleanupflag=$( cat $runfile | grep -w REMOVETEMPFILES | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )

set +x; echo -e "\n\n#############      checking analysis ###############\n\n" >&2; set -x;

if [ $analysis == "MULTIPLEXED" ]
then
   MSG="ANALYSIS=$analysis Program=$scriptfile Invalid pipeline program for this type of analysis. This program is for the NON-MULTIPLEXED case only"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi

set +x; echo -e "\n\n#############      checking input type ###############\n\n" >&2; set -x;

if [ $input_type == "GENOME" -o $input_type == "WHOLE_GENOME" -o $input_type == "WHOLEGENOME" -o $input_type == "WGS" ]
then
   input_type="WGS"
   pbscpu=$( cat $runfile | grep -w PBSCPUOTHERWGEN | cut -d '=' -f2 )
   pbsqueue=$( cat $runfile | grep -w PBSQUEUEWGEN | cut -d '=' -f2 )
elif [ $input_type == "EXOME" -o $input_type == "WHOLE_EXOME" -o $input_type == "WHOLEEXOME" -o $input_type == "WES" ]
then
   input_type="WES"
   pbscpu=$( cat $runfile | grep -w PBSCPUOTHEREXOME | cut -d '=' -f2 )
   pbsqueue=$( cat $runfile | grep -w PBSQUEUEEXOME | cut -d '=' -f2 )
else
   MSG="Invalid value for INPUTTYPE=$input_type in configuration file."
   echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi


set +x; echo -e "\n\n#############      checking cleanup options   ###############\n\n" >&2; set -x;


if [ $cleanupflag != "1" -a $cleanupflag != "0" -a $cleanupflag != "YES" -a $cleanupflag != "NO" ]
then
   MSG="Invalid value for REMOVETEMPFILES=$cleanupflag"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
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



set +x; echo -e "\n\n#############skip/include variant calling module ###############\n\n" >&2; set -x;

if [ $skipvcall != "1" -a $skipvcall != "0" -a $skipvcall != "YES" -a $skipvcall != "NO" ]
then
   MSG="Invalid value for SKIPVCALL=$skipvcall"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi
if [ $skipvcall == "1" ]
then
   skipvcall="YES"
fi
if [ $skipvcall == "0" ]
then
   skipvcall="NO"
fi

set +x; echo -e "\n\n#############skip/include QC step with verifyBamID  o###############\n\n" >&2; set -x;

if [ $runverify != "1" -a $runverify != "0" -a $runverify != "YES" -a $runverify != "NO" ]
then
   MSG="Invalid value for RUNVERIFYBAM=$runverify"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi
if [ $runverify == "1" ]
then
   runverify="YES"
fi
if [ $runverify == "0" ]
then
   runverify="NO"
fi   

if [ $launchercmd == "aprun" ]
then        
     launchercmd="aprun -n "
else        
     launchercmd="mpirun -np " 
fi

set +x; echo -e "\n\n#############version of java for gatk ###############\n\n" >&2; set -x;

if [ -z $javadir ]
then
   MSG="Value for JAVADIR must be specified in configuration file"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi

set +x; echo -e "\n\n###########checking directories  and  tools ###############\n\n" >&2; set -x;


if [ ! -d $outputdir ]
then
   MSG="$outputdir ROOT directory for this run of the pipeline not found"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi

if [ ! -d $picardir ]
then
   MSG="$picardir picard directory not found"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi
if [ ! -d $samdir ]
then
   MSG="$samdir samtools directory not found"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi
if [ ! -d $gatk ]
then
   MSG="$gatk GATK directory not found"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi

set +x; echo -e "\n\n###########checking callsets needed for analyses ###############\n\n" >&2; set -x;

if [ ! -d $refdir ]
then
   MSG="$refdir reference genome directory not found"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi
if [ ! -s $refdir/$ref ]
then
   MSG="$ref reference genome not found"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi

if [ ! -s $refdir/$dbSNP ]
then
   MSG="$dbSNP DBSNP file not found"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi   
if [ ! -d $refdir/$indeldir ]
then
   MSG="$indeldir indel directory not found"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
   exit 1;
fi

if [ ! -s $indelfile ]
then
   MSG="$indelfile INDELS file not found"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi   

if [ $runverify == "YES" -a ! -s $omnisites ]
then
   MSG="OMNISITES=$omnisites file not found. A file must be specified if this parameter has been specified too RUNVERIFYBAM=$runverify"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi

if [ `expr ${#targetfile}` -gt 1 -a $input_type == "WES" -a ! -s $targetfile ]
then
then
   MSG="TARGETFILE=$targetfile file not found. A file must be specified if this parameter has been specified too INPUTTYPE==$input_type"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi 


set +x; echo -e "\n\n" >&2;   
echo "####################################################################################################" >&2
echo "####################################################################################################" >&2
echo "###########################      params ok. Now          creating log folders        ###############" >&2
echo "####################################################################################################" >&2
echo "####################################################################################################" >&2
echo -e "\n\n" >&2; set -x;  
  

TopOutputLogs=$outputdir/logs
RealignOutputLogs=$outputdir/logs/realign
VcallOutputLogs=$outputdir/logs/variant
if [ ! -d $TopOutputLogs ]
then
   MSG="$TopOutputLogs realign directory not found"
   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
   exit 1;
fi
if [ ! -d $RealignOutputLogs ]
then
   mkdir -p $RealignOutputLogs
fi
if [ ! -d $VcallOutputLogs ]
then
   mkdir -p $VcallOutputLogs
fi

pipeid=$( cat $TopOutputLogs/CONFIGUREpbs )


set +x; echo -e "\n\n" >&2; 
echo "####################################################################################################" >&2
echo "####################################################################################################" >&2
echo "###########################   generating regions, targets, known/knownSites           ############" >&2
echo "####################################################################################################" >&2
echo "####################################################################################################" >&2
echo "target is the array of targeted regions for WES ONLY which will be used by all GATK commands"  >&2
echo "region is the array of known variants per chr  >&2
echo "realparms is the array with indels per chr which will be used for  gatk-IndelRealigner" >&2
echo "####################################################################################################" >&2
echo -e "\n\n" >&2; set -x; 

echo `date`
i=1

for chr in $indices
do
	cd $refdir/$snpdir
	region[$i]=$( find $PWD -type f -name "${chr}.*.vcf.gz" | sed "s/^/:knownSites:/g" | tr "\n" ":" )
	cd $refdir/$indeldir
	realparms[$i]=$( find $PWD -type f -name "${chr}.*.vcf" | tr "\n" ":" )

	if [ -d $targetdir -a $input_type == "WES" ]
	then
   		set +x; echo -e "\n\n###### let's check to see if we have a bed file for this chromosme ##############\n\n" >&2; set -x;

   		if [ -s $targetdir/${chr}.bed ]
   		then
       			target[$i]=$targetdir/${chr}.bed
   		fi
	fi
	(( i++ ))
done

echo `date`

set +x; echo -e "\n\n" >&2;
echo -e "\n####################################################################################################" >&2              
echo -e "\n################ checking that aligned files exist. Creating output folders          ###############" >&2
echo -e "\n"############### ALso creating RGPAMS array with the RG line for all samples         ###############" >&2     
echo -e "\n####################################################################################################" >&2
echo -e "\n\n" >&2; set -x;


if [ -s $outputdir/SAMPLENAMES_multiplexed.list ]
then 
	TheInputFile=$outputdir/SAMPLENAMES_multiplexed.list
else
	TheInputFile=$outputdir/SAMPLENAMES.list
fi


while read SampleLine
do
	if [ `expr ${#SampleLine}` -gt 1 ]
	then

  		set +x; echo -e "\n\n###### processing next non-empty line in SAMPLENAMES_multiplexed.list ##############\n\n" >&2; set -x;

  		sample=$( echo "$SampleLine" | cut -f 1 )

  		set +x; echo -e "\n\n###### checking aligned bam for realigning-recalibrating sample: $sample      ##############\n\n" >&2; set -x;

  		alignedfile=`find $outputdir/$sample/align -name "*.wdups.sorted.bam"`

	        set +x; echo -e "\n\n###### # now check that there is only one bam file      ##############\n\n" >&2; set -x;

	        aligned_bam=$outputdir/${sample}/align/${alignedfile}
	        aligned_bam=( $aligned_bam ) # recast variable as an array and count the number of members
	        if [ ${#aligned_bam[@]} -ne 1 ]
                then
                    MSG="more than one bam file found in $outputdir/${sample}/align"
                    echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | mail -s "[Task #${reportticket}]" "$redmine,$email"
                    exit 1;
                fi	  

  		set +x; echo -e "\n\n###### create folders for the analyses     ##############\n\n" >&2; set -x;

  		if [ -d $outputdir/${sample}/realign ]
  		then
      			echo -e "creating output folders for sample=$sample"

      			mkdir -p $outputdir/${sample}/realign/logs
      			mkdir -p $outputdir/${sample}/variant/logs
  		fi

   		set +x; echo -e "\n\n###### resetting logs                    ##############\n\n" >&2; set -x;

  		rm $outputdir/${sample}/realign/logs/*
  		rm $outputdir/${sample}/variant/logs/*

	fi  # end processing non-empty lines
done  <  $TheInputFile
# end loop over samples



set +x; echo -e "\n\n" >&2; 
echo "####################################################################################################" >&2
echo "####################################################################################################" >&2
echo "########################   NESTED LOOP1 to generate cmds for anisimov joblists       ###############" >&2
echo "########################   outer loop by sample; inner loop by chromosome            ###############" >&2
echo "####################################################################################################" >&2
echo "####################################################################################################" >&2
echo -e "\n\n" >&2; set -x;


samplecounter=1 
while read SampleLine
do
	if [ `expr ${#SampleLine}` -gt 1 ]
	then
          
	      set +x; echo -e "\n\n###### processing next non-empty line in SAMPLENAMES_multiplexed.list ##############\n\n" >&2; set -x;


	      sample=$( echo "$SampleLine" | cut -f 1 )

	      cd $outputdir/${sample}/align
	      aligned_bam=`find ./ -name "*.wdups.sorted.bam"`    
              header=${aligned_bam}.header
 
              set +x; echo -e "\n\n###### now putting together the RG line for RGparms                ##############\n\n" >&2; set -x;
 
              #truncate -s 0 ${aligned_bam}.RGline
	      #`grep "RG.*ID" $header | cut -f 2` >> ${aligned_bam}.RGline
              RGline=${aligned_bam}.RGline
              #RGline=$( sed -i 's/ID/RGID/' "$RGline" )
              #RGline=$( sed -i 's/ / RG/g' "$RGline" )
              #RGparms=$( cat "$RGline" )
              
              if [ `expr ${#RGline}` -lt 1 -o ! -s $RGline ]
              then
                 echo -e "RGparms line needs to be recreated from scratch" 
                 RGparms=$( grep "^@RG" *wdups*header | sed 's/@RG//' | tr ":" "=" | tr " " ":" | tr "\t" ":" | sed "s/ //g" )
              else 
                 RGparms=$( cat $RGline )
              fi

 
              set +x; echo -e "\n\n###### now defining paths to output directories          ##############\n\n" >&2; set -x;
              
              RealignOutputDir=$outputdir/${sample}/realign
              VcallOutputDir=$outputdir/${sample}/variant
              RealignLog=$outputdir/${sample}/realign/logs
              recalibratedfile=$RealignOutputDir/${sample}.recalibrated.calmd.bam
              listRealignedFiles=$RealignOutputLogs/listRealignedFiles.${sample}
              realignFailedlog=$RealignOutputLogs/FAILED_realign.${sample}              
              vcallFailedlog=$VcallOutputLogs/FAILED_vcallgatk.${sample}
              
              
              truncate -s 0 $RealignOutputLogs/verifySample.${sample}.AnisimovJoblist
              truncate -s 0 $RealignOutputLogs/realign.${sample}.AnisimovJoblist
              truncate -s 0 $RealignOutputLogs/recalibrate.${sample}.AnisimovJoblist
              truncate -s 0 $listRealignedFiles
              truncate -s 0 $realignFailedlog
              truncate -s 0 $vcallFailedlog              
              truncate -s 0 $VcallOutputLogs/vcallgatk.${sample}.AnisimovJoblist

              
              set +x; echo -e "\n\n###### now entering the inner loop by chr                                ##############\n\n" >&2; set -x;

	      chromosomecounter=1
	      for chr in $indices
	      do
	      
                  set +x; echo -e "\n\n###### generating realignment calls for sample=$sample chr=${chr} with bam=$bamfile ########\n\n" >&2; set -x;
		  echo `date`
		  
                  realignOutputfile=${chr}.realign.${sample}.bam
                  echo -n ":$realignOutputfile" >> $listRealignedFiles
                  
                  set +x; echo -e "\n\n###### if WES and target file exist then include it in the parms list ########\n\n" >&2; set -x;
                  
                  target=${target[$chromosomecounter]}
                  
                  if [ `expr ${value}` -lt 1 -a $input_type == "WES"  ]
                  then
                      target="NOTARGET"         
                  fi

                  if [ $run_method != "LAUNCHER" ]
                  then
                      realignFailedlog=$RealignOutputDir/logs/log.realign.$sample.$chr.in
                      vcallFailedlog=$VcallOutputDir/logs/log.vcallgatk.${sample}.${chr}.in
                  fi

                  echo "$scriptdir/realign.sh $RealignOutputDir $realignOutputfile $aligned_bam $chr $RGparms $target ${realparms[$chromosomecounter]} $runfile $RealignOutputDir/logs/log.realign.$sample.$chr.in $RealignOutputDir/logs/log.realign.$sample.$chr.ou $email $RealignOutputDir/logs/realign.${sample}.${chr} $realignFailedlog" > $RealignOutputDir/logs/realign.${sample}.${chr}
 
                  if [ $skipvcall == "NO" ]
                  then
                      set +x; echo -e "\n\n###### generating variant calls for sample=$sample chr=${chr} with bam=$bamfile ########\n\n" >&2; set -x;
		      echo "$scriptdir/vcallgatk.sh $sample $VcallOutputDir  $RealignOutputDir $recalibratedfile $chr $target $runfile $VcallOutputDir/logs/log.vcallgatk.${sample}.${chr}.in $VcallOutputDir/logs/log.vcallgatk.${sample}.${chr}.ou $email $VcallOutputDir/logs/vcallgatk.${sample}.${chr} $vcallFailedlog" > $VcallOutputDir/logs/vcallgatk.${sample}.${chr}
                  fi

		  ((  chromosomecounter++ )) 
                  set +x; echo -e "\n\n###### bottom of the loop over chrs                             ##############\n\n" >&2; set -x;
		  
	      done # done going through chromosomes 

              set +x; echo -e "\n\n###### Exiting the inner loop by chr. We still need to put together the recalibrate cmd ##############\n\n" >&2; set -x;

              echo "nohup $scriptdir/recalibrate.sh $RealignOutputDir $sample $listRealignedFiles $recalibratedfile $RGparms $runfile $RealignOutputDir/logs/log.recalibrate.$sample.in $RealignOutputDir/logs/log.recalibrate.$sample.ou $email $RealignOutputDir/logs/recalibrate.${sample}" > $RealignOutputDir/logs/recalibrate.${sample}
 	      
              (( samplecounter++ ))
	fi # done processing non-empty lines
          
	echo -e "\n\n###### bottom of the loop over samples   
          
done <  $TheInputFile
# end loop over samples

# at the end of this set of nested loops, the variables chromosomecounter and samplecounter
# reflect, respectively, number_of_chromosomes+1 and number_of_samples+1,
# which is exactly the number of nodes required for anisimov launcher 



set +x; echo -e "\n\n" >&2; 
echo -e "\n####################################################################################################" >&2
echo -e "\n##########     END NESTED LOOP1 to generate cmds for anisimov joblists                   ###########" >&2
echo -e "\n####################################################################################################" >&2
echo -e "\n\n" >&2; set -x;


#set +x; echo -e "\n ### update autodocumentation script ### \n"; set -x;
#echo -e "# @begin RealignRecalibrate_per_chromosome" >> $outputdir/WorkflowAutodocumentationScript.sh
#echo -e "   # @in sample_chr @as aligned_bam_per_chromosome" >> $outputdir/WorkflowAutodocumentationScript.sh
#RealignedBAMTemplate="{SampleName}/realign/{chromosome}.realrecal.${SampleName}.output.bam"
#echo -e "   # @out realrecal  @as  realigned_bam_per_chromosome @URI ${RealignedBAMTemplate}" >> $outputdir/WorkflowAutodocumentationScript.sh
#echo -e "# @end RealignRecalibrate_per_chromosome" >> $outputdir/WorkflowAutodocumentationScript.sh

#echo -e "# @begin VariantCalling_per_chromosome" >> $outputdir/WorkflowAutodocumentationScript.sh
#echo -e "   # @in  realrecal  @as  realigned_bam_per_chromosome" >> $outputdir/WorkflowAutodocumentationScript.sh
#VariantTemplate=${RealignedBAMTemplate}.raw.all.vcf
#echo -e "   # @out vcf @as output_variants @URI ${VariantTemplate}" >> $outputdir/WorkflowAutodocumentationScript.sh
#echo -e "# @end VariantCalling_per_chromosome" >> $outputdir/WorkflowAutodocumentationScript.sh

#echo -e "# @out vcf @as output_variants @URI sample_name/variant/chr_name.vcf" >> $outputdir/WorkflowAutodocumentationScript.sh
#WorkflowName=`basename $outputdir`
#echo -e "# @end $WorkflowName" >> $outputdir/WorkflowAutodocumentationScript.sh



set +x; echo -e "\n\n" >&2; 
echo "####################################################################################################" >&2
echo "####################################################################################################" >&2
echo "###################################   NEXT: JOB SCHEDULING CASES                        ############" >&2
echo "####################################################################################################" >&2
echo "####################################################################################################" >&2
echo -e "\n\n" >&2; set -x;


case $run_method in
"LAUNCHER")

        set +x; echo -e "\n\n" >&2; 
        echo "####################################################################################################" >&2
	echo -e "CASE2: ANISIMOV LAUNCHER. " >&2
        echo "####################################################################################################" >&2
        echo -e "\n\n" >&2; set -x;	      
      
      	while read SampleLine
      	do
           if [ `expr ${#SampleLine}` -gt 1 ]
           then
               set +x; echo -e "\n\n###### processing next non-empty line                               ##############\n\n" >&2; set -x;
                  
	       sample=$( echo "$SampleLine" | cut -f 1 )
	         
	       ## defining paths and reset joblists
	         
               RealignOutputDir=$outputdir/$sample/realign
               VcallOutputDir=$outputdir/${sample}/variant  
                          
	       set +x; echo -e "\n\n ###### loop2 by chromosome to populate the joblist by writing ONE line for each job call to a sample-chr pair ###### \n\n" set +x;        
                 
               for chr in $indices
               do
         
                     # creating a qsub out of the job file
                     # need to prepend "nohup" and append log file name, so that logs are properly created when Anisimov launches these jobs 

                     realign_log=$RealignOutputDir/logs/log.realign.${sample}.$chr.in

		     truncate -s 0 $RealignOutputDir/logs/jobfile.realign.${sample}.${chr}
                     truncate -s 0 $RealignOutputDir/logs/log.realign.${sample}.$chr.in

                     awk -v awkvar_realignlog=$realign_log '{print "nohup "$0" > "awkvar_realignlog}' $RealignOutputDir/logs/realign.${sample}.${chr} > $RealignOutputDir/logs/jobfile.realign.${sample}.${chr}
                     echo "$RealignOutputDir/logs/ jobfile.realign.${sample}.${chr}" >> $RealignOutputLogs/realign.${sample}.AnisimovJoblist

                     if [ $skipvcall == "NO" ]
                     then
                         vcall_log=$VcallOutputDir/logs/log.vcallgatk.${sample}.${chr}.in

                         truncate -s 0 $VcallOutputDir/logs/log.vcallgatk.${sample}.${chr}.in
                         truncate -s 0 $VcallOutputDir/logs/jobfile.vcallgatk.${sample}.${chr}

                         awk -v awkvar_vcalllog=$vcall_log '{print "nohup "$0" > "awkvar_vcalllog}' $VcallOutputDir/logs/vcallgatk.${sample}.${chr} > $VcallOutputDir/logs/jobfile.vcallgatk.${sample}.${chr}
                         echo "$VcallOutputDir/logs/ jobfile.vcallgatk.${sample}.${chr}" >> $VcallOutputLogs/vcallgatk.${sample}.AnisimovJoblist
                     fi
               done # end loop2 by chromosome


               set +x; echo -e "\n\n ###### putting together the other pieces of the qsub file and then scheduling Anisimov Launcher joblists   ###### \n\n\n" >&2; set -x;

               set +x; echo -e "\n\n######  constructing the realign qsub for sample $sample      ##############\n\n" >&2; set -x;
               
               qsub_realign_anisimov=$RealignOutputLogs/qsub.realign.${sample}.AnisimovLauncher
               cat $outputdir/qsubGenericHeader > $qsub_realign_anisimov
               
               echo "#PBS -N ${pipeid}_realign_${sample}" >> $qsub_realign_anisimov
               echo "#PBS -l walltime=$pbscpu" >> $qsub_realign_anisimov
               echo "#PBS -o $RealignOutputLogs/log.realign.${sample}.ou" >> $qsub_realign_anisimov
               echo "#PBS -e $RealignOutputLogs/log.realign.${sample}.in" >> $qsub_realign_anisimov
               # realign and varcall scripts use multithreaded processes,
               # so we will give each chromosome its own node  +1 for the launcher
               # chromosmecounter is the variable that has this value already setup
               echo "#PBS -l nodes=$chromosomecounter:ppn=$thr" >> $qsub_realign_anisimov

               # the actual command
               echo "$launchercmd $chromosomecounter -N 1 -d $thr ~anisimov/scheduler/scheduler.x $RealignOutputLogs/realign.${sample}.AnisimovJoblist /bin/bash > $RealignOutputLogs/realign.${sample}.AnisimovLauncher.log" >> $qsub_realign_anisimov

               set +x; echo -e "\n\n######  constructing the recalibrate qsub for sample $sample. No launcher  ##############\n\n" >&2; set -x;

               qsub_recalibrate=$RealignOutputLogs/qsub.recalibrate.${sample}
               cat $outputdir/qsubGenericHeader > $qsub_recalibrate
  
               echo "#PBS -N ${pipeid}_recalibrate_${sample}" >> $qsub_recalibrate
               echo "#PBS -l walltime=$pbscpu" >> $qsub_recalibrate
               echo "#PBS -o $RealignOutputLogs/log.recalibrate.${sample}.ou" >> $qsub_recalibrate
               echo "#PBS -e $RealignOutputLogs/log.recalibrate.${sample}.in" >> $qsub_recalibrate
               echo "#PBS -l nodes=1:ppn=$thr" >> $qsub_recalibrate
               echo $RealignOutputDir/logs/recalibrate.${sample}  >> $qsub_recalibrate


               set +x; echo -e "\n\n######  constructing the qsub for varcall for sample $sample  ##############\n\n" >&2; set -x;

               if [ $skipvcall == "NO" ]
               then

                     qsub_vcallgatk_anisimov=$VcallOutputLogs/qsub.vcalgatk.${sample}.AnisimovLauncher
                     cat $outputdir/qsubGenericHeader > $qsub_vcallgatk_anisimov
               
                     echo "#PBS -N ${pipeid}_vcallgatk_${sample}" >> $qsub_vcallgatk_anisimov
                     echo "#PBS -l walltime=$pbscpu" >> $qsub_vcallgatk_anisimov
                     echo "#PBS -o $VcallOutputLogs/log.vcallgatk.${sample}.ou" >> $qsub_vcallgatk_anisimov
                     echo -e "#PBS -e $VcallOutputLogs/log.vcallgatk.${sample}.in\n" >> $qsub_vcallgatk_anisimov
                     # the dependency on realrecal job will be added when it is scheduled in the loop below
                     echo -e "#PBS -l nodes=$chromosomecounter:ppn=$thr\n" >> $qsub_vcallgatk_anisimov
                     # the actual command
                    echo "$launchercmd  $chromosomecounter -N 1 -d 32 ~anisimov/scheduler/scheduler.x $VcallOutputLogs/vcallgatk.${sample}.AnisimovJoblist /bin/bash > $VcallOutputLogs/vcallgatk.${sample}.AnisimovLauncher.log" >> $qsub_vcallgatk_anisimov
               fi

           fi # end non-empty line
      	done <  $TheInputFile
      	# end loop over samples. Still inside case=LAUNCHER


	set +x; echo -e "\n\n" >&2; 
	echo "####################################################################################################" >&2
	echo "####################################################################################################" >&2
	echo "####              arranging execution order with job dependencies            #######################" >&2
	echo "####################################################################################################" >&2      
	echo "####################################################################################################" >&2
	echo -e "\n\n" >&2; set -x;


	# reset the lists
	truncate -s 0 $RealignOutputLogs/RECALIBRATEpbs
	truncate -s 0 $RealignOutputLogs/REALIGNpbs
	if [ $skipvcall == "NO" ]
	then
	 	truncate -s 0 $VcallOutputLogs/VCALGATKpbs
	fi
	cd $RealignOutputLogs # so that whatever temp folders and pbs notifications would go there


	while read SampleLine
	do
           if [ `expr ${#SampleLine}` -gt 1 ]
           then
               set +x; echo -e "\n\n###### processing next non-empty line          ##############\n\n" >&2; set -x;
	       sample=$( echo "$SampleLine" | cut -f 1 )

               RealignOutputDir=$outputdir/$sample/realign
               VcallOutputDir=$outputdir/${sample}/variant  
               RealignLog=$RealignOutputDir/logs

               set +x; echo -e "\n\n######  launching realign                       ##############\n\n" >&2; set -x;

               realign_job=`qsub $RealignOutputLogs/qsub.realign.${sample}.AnisimovLauncher`
               #`qhold -h u $realign_job`
               echo $realign_job >> $RealignOutputLogs/REALIGNpbs
               echo $realign_job >> $TopOutputLogs/pbs.REALIGN


               set +x; echo -e "\n\n######  launching recalibrate                       ##############\n\n" >&2; set -x;
               sed -i "2i #PBS -W depend=afterok:$realign_job" $RealignOutputLogs/qsub.recalibrate.${sample}
               recalibrate_job=`qsub $RealignOutputLogs/qsub.recalibrate.${sample}` 
               #`qhold -h u $recalibrate_job`
               echo $recalibrate_job>> $RealignOutputLogs/RECALIBRATEpbs
               echo $recalibrate_job >> $TopOutputLogs/pbs.RECALIBRATE

               set +x; echo -e "\n\n######  launching varcall                       ##############\n\n" >&2; set -x;
      
               if [ $skipvcall == "NO" ]
               then
                     sed -i "2i #PBS -W depend=afterok:$recalibrate_job" $VcallOutputLogs/qsub.vcalgatk.${sample}.AnisimovLauncher
                     vcallgatk_job=`qsub $VcallOutputLogs/qsub.vcalgatk.${sample}.AnisimovLauncher`
                     #`qhold -h u $vcallgatk_job`
                     echo $vcallgatk_job >> $VcallOutputLogs/VCALGATKpbs
		     echo $vcallgatk_job >> $TopOutputLogs/pbs.VARCALL
               fi
           fi # done processing non-empty line    
      done <  $TheInputFile

;;
"APRUN")
        set +x; echo -e "\n\n" >&2; 
        echo "####################################################################################################" >&2
	echo -e "CASE1: BLUE WATERS LAUNCHER. " >&2
        echo "####################################################################################################" >&2
        echo -e "\n\n" >&2; set -x;	

      truncate -s 0 $RealignOutputLogs/VERIFYXSAMPLEpbs
      truncate -s 0 $RealignOutputLogs/REALRECALpbs
      truncate -s 0 $VcallOutputLogs/VCALGATKpbs

      echo -e "\n\n ###### loop1 by sample  ###### \n\n"
      while read SampleLine
      do

          if [ `expr ${#SampleLine}` -gt 1 ]
          then
              echo -e "\n\n ###### processing non-empty line $SampleLine ###### \n\n"
                  
	      sample=$( echo "$SampleLine" | cut -f 1 )

              RealignOutputDir=$outputdir/$sample/realign
              VcallOutputDir=$outputdir/${sample}/variant 

              set +x; echo -e "\n\n" >&2; 
              echo "####################################################################################################" >&2
              echo -e "collecting all jobs for sample= $sample    " >&2
              echo "####################################################################################################" >&2
              echo -e "\n\n" >&2; set -x;

              for chr in $indices
              do

                 set +x; echo -e "\n\n###### constructing  realrecal qsub for for sample $sample chr $chr ##############\n\n" >&2; set -x;

                 qsub_realrecal=$RealignOutputDir/logs/qsub.realrecal.${sample}.${chr}


                 cat $outputdir/qsubGenericHeader > $qsub_realrecal
                 echo "#PBS -N ${pipeid}_realrecal.${sample}.${chr}" >> $qsub_realrecal
                 echo "#PBS -l walltime=$pbscpu" >> $qsub_realrecal
                 echo "#PBS -o $RealignOutputDir/logs/log.realrecal.${sample}.${chr}.ou" >> $qsub_realrecal
                 echo "#PBS -e $RealignOutputDir/logs/log.realrecal.${sample}.${chr}.in" >> $qsub_realrecal
                 echo "#PBS -W depend=afterok:$split_job" >> $qsub_realrecal
                 echo -e "#PBS -l nodes=1:ppn=$thr\n" >> $qsub_realrecal

                 echo "aprun -n 1 -N 1 -d $thr /bin/bash $RealignOutputDir/logs/realrecal.${sample}.${chr}" >> $qsub_realrecal

                 realrecal_job=`qsub $qsub_realrecal`
                 `qhold -h u $realrecal_job`
                 #`qrls -h u $split_job` 
                 echo $realrecal_job >> $RealignOutputLogs/REALRECALpbs
                 echo $realrecal_job >> $TopOutputLogs/pbs.REALRECAL


                 if [ $skipvcall == "NO" ]
                 then
                 
                     set +x; echo -e "\n\n###### constructing  varcall qsub for for sample $sample chr $chr ##############\n\n" >&2; set -x;  
                     
                     qsub_vcall=$VcallOutputDir/logs/qsub.vcallgatk.${sample}.${chr}

                     cat $outputdir/qsubGenericHeader > $qsub_vcall
                     echo "#PBS -N ${pipeid}_vcall_${SampleName}.${chr}" >> $qsub_vcall
                     echo "#PBS -l walltime=$pbscpu" >> $qsub_vcall
                     echo "#PBS -o $VcallOutputDir/logs/log.vcall.${sample}.${chr}.ou" >> $qsub_vcall
                     echo "#PBS -e $VcallOutputDir/logs/log.vcall.${sample}.${chr}.in" >> $qsub_vcall
                     echo "#PBS -W depend=afterok:$realrecal_job" >> $qsub_vcall
                     echo -e "#PBS -l nodes=1:ppn=$thr\n" >> $qsub_vcall
   
                     echo "aprun -n 1 -d $thr /bin/bash $VcallOutputDir/logs/vcallgatk.${SampleName}.${chr}" >> $qsub_vcall
   
                     vcall_job=`qsub $qsub_vcall`
                     #`qrls -h u $realrecal_job`
                     echo $vcall_job >> $VcallOutputLogs/VCALGATKpbs
                     echo $vcall_job >> $TopOutputLogs/pbs.VARCALL
                 fi
                 set +x; echo -e "\n\n###### bottom of the loop over chrs                              ##############\n\n" >&2; set -x;
                 
              done # going through chromosomes for this sample
         fi  # non-empty line of file
         set +x; echo -e "\n\n###### bottom of the loop over samples                              ##############\n\n" >&2; set -x;
         
      done <  $TheInputFile
      # end loop over samples
;;
"QSUB")
	# will add later
;;   
esac


set +x; echo -e "\n\n" >&2; 
echo "####################################################################################################" >&2
echo "####################################################################################################" >&2
echo "###########################    wrap up and produce summary table        ############################" >&2
echo "####################################################################################################" >&2
echo "####################################################################################################" >&2
echo -e "\n\n" >&2; set -x;

if [ $skipvcall == "NO" ]
then
      summarydependids=$( cat $VcallOutputLogs/VCALGATKpbs | sed "s/\..*//" | tr "\n" ":" )
else
      summarydependids=$( cat $RealignOutputLogs/RECALIBRATEpbs | sed "s/\..*//" | tr "\n" ":" )
fi

lastjobid=""
cleanjobid=""

if [ $cleanupflag == "YES" ]
then
       qsub_cleanup=$TopOutputLogs/qsub.cleanup
       echo "#PBS -A $pbsprj" >> $qsub_cleanup
       echo "#PBS -N ${pipeid}_cleanup" >> $qsub_cleanup
       echo "#PBS -l walltime=$pbscpu" >> $qsub_cleanup
       echo "#PBS -l nodes=1:ppn=1" >> $qsub_cleanup
       echo "#PBS -o $TopOutputLogs/log.cleanup.ou" >> $qsub_cleanup
       echo "#PBS -e $TopOutputLogs/log.cleanup.in" >> $qsub_cleanup
       echo "#PBS -q $pbsqueue" >> $qsub_cleanup
       echo "#PBS -m a" >> $qsub_cleanup
       echo "#PBS -M $email" >> $qsub_cleanup
       echo "#PBS -W depend=afterok:$summarydependids" >> $qsub_cleanup
       echo "$scriptdir/cleanup.sh $outputdir $analysis $TopOutputLogs/log.cleanup.in $TopOutputLogs/log.cleanup.ou $email $TopOutputLogs/qsub.cleanup" >> $qsub_cleanup
       #`chmod a+r $qsub_cleanup`
       cleanjobid=`qsub $qsub_cleanup`
       echo $cleanjobid >> $outputdir/logs/CLEANUPpbs
fi

`sleep 30s`
qsub_summary=$TopOutputLogs/qsub.summary.allok
echo "#PBS -A $pbsprj" >> $qsub_summary
echo "#PBS -N ${pipeid}_summaryok" >> $qsub_summary
echo "#PBS -l walltime=01:00:00" >> $qsub_summary # 1 hour should be more than enough
echo "#PBS -l nodes=1:ppn=1" >> $qsub_summary
echo "#PBS -o $TopOutputLogs/log.summary.ou" >> $qsub_summary
echo "#PBS -e $TopOutputLogs/log.summary.in" >> $qsub_summary
echo "#PBS -q $pbsqueue" >> $qsub_summary
echo "#PBS -m a" >> $qsub_summary
echo "#PBS -M $email" >> $qsub_summary
if [ `expr ${#cleanjobid}` -gt 0 ]
then
       echo "#PBS -W depend=afterok:$cleanjobid" >> $qsub_summary
else
       echo "#PBS -W depend=afterok:$summarydependids" >> $qsub_summary
fi
echo "$scriptdir/summary.sh $runfile $email exitok $reportticket"  >> $qsub_summary
#`chmod a+r $qsub_summary`
lastjobid=`qsub $qsub_summary`
echo $lastjobid >> $TopOutputLogs/SUMMARYpbs

if [ `expr ${#lastjobid}` -lt 1 ]
then
       echo "at least one job aborted"
       qsub_summary=$TopOutputLogs/qsub.summary.afterany
       echo "#PBS -A $pbsprj" >> $qsub_summary
       echo "#PBS -N ${pipeid}_summary_afterany" >> $qsub_summary
       echo "#PBS -l walltime=01:00:00" >> $qsub_summary # 1 hour should be more than enough
       echo "#PBS -l nodes=1:ppn=1" >> $qsub_summary
       echo "#PBS -o $TopOutputLogs/log.summary.afterany.ou" >> $qsub_summary
       echo "#PBS -e $TopOutputLogs/log.summary.afterany.in" >> $qsub_summary
       echo "#PBS -q $pbsqueue" >> $qsub_summary
       echo "#PBS -m a" >> $qsub_summary
       echo "#PBS -M $email" >> $qsub_summary
       echo "#PBS -W depend=afterany:$summarydependids" >> $qsub_summary
       echo "$scriptdir/summary.sh $runfile $email exitnotok $reportticket"  >> $qsub_summary
       #`chmod a+r $qsub_summary`
       badjobid=`qsub $qsub_summary`
       echo $badjobid >> $TopOutputLogs/SUMMARYpbs
fi


# release all jobs now. 
# todo: the lines below have to be redone: place them in a loop to release all samples, instead of those jobs for the last sample

#realignids=$( cat $RealignOutputLogs/REALIGNpbs | sed "s/\..*//" | tr "\n" " " )
#recalids=$( cat $RealignOutputLogs/RECALIBRATEpbs | sed "s/\..*//" | tr "\n" " " )
#`qrls -h u $realignids`
#`qrls -h u $recalids`

#if [ $skipvcall == "NO" ]
#then
#   vcalids=$( cat $VcallOutputLogs/VCALGATKpbs | sed "s/\..*//" | tr "\n" " " )
#   `qrls -h u $vcalids`
#fi

