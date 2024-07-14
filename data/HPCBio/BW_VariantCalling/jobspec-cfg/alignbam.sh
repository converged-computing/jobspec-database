#!/bin/bash
#
# alignbam.sh
# align module to be used for input files in bam format
#redmine=hpcbio-redmine@igb.illinois.edu
##redmine=grendon@illinois.edu
if [ $# != 5 ]
then
        MSG="Parameter mismatch"
        echo -e "jobid:${PBS_JOBID}\nprogram=$0 stopped at line=$LINENO.\nReason=$MSG"
        #echo -e "jobid:${PBS_JOBID}\nprogram=$0 stopped at line=$LINENO.\nReason=$MSG" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$grendon@illinois.edu""
        exit 1;
else 
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
           echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
           #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
           exit 1;
        fi

        set +x; echo -e "\n\n" >&2; 
        echo "####################################################################################################" >&2
        echo "##################################### PARSING RUN INFO FILE ########################################" >&2
        echo "##################################### AND SANITY CHECK      ########################################" >&2
        echo "####################################################################################################" >&2
        echo -e "\n\n" >&2; set -x;

	outputdir=$( cat $runfile | grep -w OUTPUTDIR | cut -d '=' -f2 )
        nodes=$( cat $runfile | grep -w PBSNODES | cut -d '=' -f2 )
        pbsprj=$( cat $runfile | grep -w PBSPROJECTID | cut -d '=' -f2 )
        thr=$( cat $runfile | grep -w PBSTHREADS | cut -d '=' -f2 )
        refdir=$( cat $runfile | grep -w REFGENOMEDIR | cut -d '=' -f2 )
        scriptdir=$( cat $runfile | grep -w SCRIPTDIR | cut -d '=' -f2 )
        ref=$( cat $runfile | grep -w REFGENOME | cut -d '=' -f2 )
        aligner=$( cat $runfile | grep -w ALIGNER | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
        analysis=$( cat $runfile | grep -w ANALYSIS | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
        picardir=$( cat $runfile | grep -w PICARDIR | cut -d '=' -f2 )
        samdir=$( cat $runfile | grep -w SAMDIR | cut -d '=' -f2 )
        epilogue=$( cat $runfile | grep -w EPILOGUE  | cut -d '=' -f2 )
        dup=$( cat $runfile | grep -w MARKDUP  | cut -d '=' -f2 )
        dupflag=$( cat $runfile | grep -w REMOVE_DUP  | cut -d '=' -f2 )
        type=$( cat $runfile | grep -w INPUTTYPE | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
        inputdir=$( cat $runfile | grep -w INPUTDIR | cut -d '=' -f2 )
        paired=$( cat $runfile | grep -w PAIRED | cut -d '=' -f2 )
        samplefileinfo=$( cat $runfile | grep -w SAMPLEFILENAMES | cut -d '=' -f2 )
        rlen=$( cat $runfile | grep -w READLENGTH | cut -d '=' -f2 )
        multisample=$( cat $runfile | grep -w MULTISAMPLE | cut -d '=' -f2 )
        samples=$( cat $runfile | grep -w SAMPLENAMES | cut -d '=' -f2 | tr ":" "\n")
        sortool=$( cat $runfile | grep -w SORTMERGETOOL | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
        cleanupflag=$( cat $runfile | grep -w REMOVETEMPFILES | cut -d '=' -f2 | tr '[a-z]' '[A-Z]' )
        revertsam=$( cat $runfile | grep -w REVERTSAM | cut -d '=' -f2 )
        igvdir=$( cat $runfile | grep -w IGVDIR | cut -d '=' -f2 )
        sortooldir=$( cat $runfile | grep -w NOVODIR | cut -d '=' -f2 )

        if [ $input_type == "GENOME" -o $input_type == "WHOLE_GENOME" -o $input_type == "WHOLEGENOME" -o $input_type == "WGS" ]
        then
            pbscpu=$( cat $runfile | grep -w PBSCPUALIGNWGEN | cut -d '=' -f2 )
            pbsqueue=$( cat $runfile | grep -w PBSQUEUEWGEN | cut -d '=' -f2 )
        else
            if [ $input_type == "EXOME" -o $input_type == "WHOLE_EXOME" -o $input_type == "WHOLEEXOME" -o $input_type == "WES" ]
            then
		pbscpu=$( cat $runfile | grep -w PBSCPUALIGNEXOME | cut -d '=' -f2 )
		pbsqueue=$( cat $runfile | grep -w PBSQUEUEEXOME | cut -d '=' -f2 )
            else
		MSG="Invalid value for INPUTTYPE=$input_type in configuration file."
		echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
		#echo -e "program=$0 stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
                exit 1;
            fi
        fi

        if [ $revertsam != "1" -a $revertsam != "0" -a $revertsam != "YES" -a $revertsam != "NO" ]
        then
           MSG="Invalid value for REVERTSAM=$revertsam"
            echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
            #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
            exit 1;
        else
            if [ $revertsam == "YES" ]
            then
                $revertsam="1"
            fi
            if [ $revertsam == "NO" ]
            then
                $revertsam="0"
            fi
        fi

        if [ $cleanupflag != "1" -a $cleanupflag != "0" -a $cleanupflag != "YES" -a $cleanupflag != "NO" ]
        then
           MSG="Invalid value for REMOVETEMPFILES=$cleanupflag"
            echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
            #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
            exit 1;
        else
            if [ $cleanupflag == "1" ]
            then
                $cleanupflag="YES"
            fi
            if [ $cleanupflag == "0" ]
            then
                $cleanupflag="NO"
            fi
        fi


        if [ $dup != "1" -a $dup != "0" -a $dup != "YES" -a $dup != "NO" ]
        then
           MSG="Invalid value for MARKDUP=$dup"
            echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
            #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
            exit 1;
        else
            if [ $dup == "1" ]
            then
                $dup="YES"
            fi
            if [ $dup == "0" ]
            then
                $dup="NO"
            fi
        fi

        if [ $dupflag != "1" -a $dupflag != "0" -a $dupflag != "YES" -a $dupflag != "NO" ]
        then
           MSG="Invalid value for REMOVE_DUP=$dupflag"
            echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
            #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
            exit 1;
        else
            if [ $dupflag == "1" ]
            then
                $dupflag="YES"
            fi
            if [ $dupflag == "0" ]
            then
                $dupflag="NO"
            fi
        fi


	dupparms=$( echo "dup=${dup}_flag=${dupflag}" )

        if [ $multisample != "1" -a $multisample != "0" -a $multisample != "YES" -a $multisample != "NO" ]
        then
            MSG="Invalid value for MULTISAMPLE=$multisample"
            echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
            #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
            exit 1;
        else
            if [ $multisample == "1" ]
            then
                $multisample="YES"
            fi
            if [ $multisample == "0" ]
            then
                $multisample="NO"
            fi
        fi

        if [ $aligner != "NOVOALIGN" -a $aligner != "BWA" ]
        then
            MSG="ALIGNER=$aligner  is not available at this site"
            echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
            #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
            exit 1;
        fi

        if [ $aligner == "NOVOALIGN" ]
        then
            alignerdir=$( cat $runfile | grep -w NOVODIR | cut -d '=' -f2 )
            refindexed=$( cat $runfile | grep -w NOVOINDEX | cut -d '=' -f2 )
            alignparms=$( cat $runfile | grep -w NOVOPARAMS | cut -d '=' -f2 | tr " " "_" )_-c_${thr}
        fi
        if [ $aligner == "BWA" ]
        then
            alignerdir=$( cat $runfile | grep -w BWADIR | cut -d '=' -f2 )
            refindexed=$( cat $runfile | grep -w BWAALNINDEX | cut -d '=' -f2 )
            alignparms=$( cat $runfile | grep -w BWAPARAMS | cut -d '=' -f2 | tr " " "_" )_-t_${thr}
        fi
        if [ -z $epilogue ]
        then
           MSG="Value for EPILOGUE must be specified in configuration file"
           echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
           #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
           exit 1;
        else
           `chmod 740 $epilogue`
        fi

        if [ -z $sortool ]
        then
           MSG="Value for SORTOOL must be specified in configuration file"
           echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
           #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
           exit 1;
        else
           if [ $sortool != "NOVOSORT" -a $sortool != "PICARD" ]
           then
               MSG="Invalid value for SORTOOL=$sortool in configuration file"
               echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
               #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
               exit 1;
           fi
        fi      
        if [ ! -d $scriptdir ]
        then
           MSG="SCRIPTDIR=$scriptdir directory not found"
           echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
           #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
           exit 1;
        fi

        if [ ! -d $outputdir ]
        then
           mkdir -p $outputdir
        fi

        if [ ! -s $refdir/$ref ]
        then
           MSG="$refdir/$ref reference genome not found"
           echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
           #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
           exit 1;
        fi
        if [ ! -s $refdir/$refindexed ]
        then
           MSG="$refdir/$refindexed index for reference genome not found"
           echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
           #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
           exit 1;
        fi
        if [ ! -d $inputdir ]
        then
           MSG="INPUTDIR=$inputdir input directory not found"
           echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
           #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""           
	   exit 1;
        fi
        if [ ! -d $alignerdir ]
        then
           MSG="$alignerdir aligner directory not found"
           echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
           #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
           exit 1;
        fi
        if [ ! -d $picardir ]
        then
           MSG="PICARDIR=$picardir directory not found"
           echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
           #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
           exit 1;
        fi
        if [ ! -d $samdir ]
        then
           MSG="SAMDIR=$samdir directory not found"
           echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
           #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
           exit 1;
        fi
        if [ ! -s $samplefileinfo ]
        then
           MSG="SAMPLEFILENAMES=$samplefileinfo file not found"
           echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
           #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$email""
           exit 1;
        fi
        igv=$outputdir/$igvdir
        extradir=$outputdir/extrareads

        set +x; echo -e "\n\n" >&2; 
        echo "####################################################################################################" >&2
        echo "##################################### simple QC test on input files    #############################" >&2
        echo "####################################################################################################" >&2
        echo -e "\n\n" >&2; set -x;

        numsamples=0
        for name in $samples
        do
            countnames=$( cat $samplefileinfo | grep $name -c )
            if [ $countnames -lt 1 ]
            then
              MSG="No samples found in SAMPLEFILENAMES=$samplefileinfo."
              echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
              #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$redmine,$email""
              exit 1;
            fi
            let numsamples+=1
        done
        if [ $numsamples -gt 1 -a $multisample == "YES" ]
        then
            echo "multiple samples to be aligned."
        else
           if [ $numsamples -eq 1 -a $multisample == "NO" ]
           then
              echo "single sample to be aligned."
           else
              MSG="mismatch between number of samples found=$numsamples and vaalue of parameter MULTISAMPLE=$multisample in configuration file."
              echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
              #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$redmine,$email""
              exit 1;
	   fi
        fi

        oualigndir=$outputdir/align
        output_logs=$outputdir/logs
        pipeid=$( cat $output_logs/pbs.CONFIGURE )
        
        if [ -d $oualigndir ]
        then
           echo "$oualigndir is there; resetting it"
           `rm -r $oualigndir/*`
        else
           mkdir -p $oualigndir
        fi

        if [ -d $output_logs ]
        then
           echo "$output_logs is there; resetting it"
           #`rm -r $output_logs/*`
           pbsids=""
        else
           mkdir -p $output_logs
        fi

        set +x; echo -e "\n\n">&2
        echo "############################################################################################################" >&2
        echo "##############################         MAIN ALIGNMENT LOOP                       ###########################" >&2
        echo "############################################################################################################" >&2
        echo -e "\n\n" >&2; set -x;

        allfiles=""
        while read sampledetail
        do
          echo "processing next line in samplefile..."
          if [ `expr ${#sampledetail}` -lt 7 ]
          then
              set +x; echo -e "\n\n########################          skipping empty line                     #######################\n\n" >&2; set -x;
          else

              set +x; echo -e "\n\n" >&2; 
              echo "####################################################################################################" >&2
              echo "##################################### processing $sampledetail         #############################" >&2
              echo "####################################################################################################" >&2
              echo -e "\n\n" >&2; set -x;
              
              dirname=$( echo $sampledetail | cut -d ':' -f2 | cut -d '=' -f1 )
              BAM=$( echo $sampledetail | cut -d '=' -f2 )
              outputalign=$oualigndir/$dirname
              outputlogs=$output_logs/align

              if [ ! -d $outputalign ]
              then
		mkdir $outputalign
		outputsam=$outputalign/$dirname
	      else
		outputsam=$outputalign/$dirname
	      fi
              if [ ! -d $outputlogs ]
              then
 		mkdir $outputlogs
	      fi
	      #`chmod -R 770 $outputalign/`
	      #`chmod -R 770 $outputlogs/`

              cd $outputalign
              sortedplain=$outputsam.wrg.sorted.bam
              outsortnodup=$outputsam.nodups.sorted.bam
              outsortwdup=$outputsam.wdups.sorted.bam
              sID=$dirname
              sPU=$dirname
              sSM=$dirname
              sPL=$( cat $runfile | grep -w SAMPLEPL | cut -d '=' -f2 )
              sCN=$( cat $runfile | grep -w SAMPLECN | cut -d '=' -f2 )
              sLB=$( cat $runfile | grep -w SAMPLELB | cut -d '=' -f2 )
              RGparms=$( echo "ID=${sID}:LB=${sLB}:PL=${sPL}:PU=${sPU}:SM=${sSM}:CN=${sCN}" )

              set +x; echo -e "\n\n" >&2; 
              echo "####################################################################################################" >&2
              echo "##################################### aligning  $sampledetail         #############################" >&2
              echo "####################################################################################################" >&2
              echo -e "\n\n" >&2; set -x;
               
              if [ $aligner == "NOVOALIGN"  ]
	      then
                echo "novoalign is used as aligner. input file in bam format"
                qsub=$outputlogs/qsub.novoalnbam.$dirname
                echo "#PBS -A $pbsprj" >> $qsub
                echo "#PBS -N ${pipeid}_novoalnbam_${dirname}" >> $qsub
		echo "#PBS -l epilogue=$epilogue" >> $qsub
		echo "#PBS -l walltime=$pbscpu" >> $qsub
		echo "#PBS -l nodes=1:ppn=$thr" >> $qsub
		echo "#PBS -o $outputlogs/log.novoalnbam.$dirname.ou" >> $qsub
		echo "#PBS -e $outputlogs/log.novoalnbam.$dirname.in" >> $qsub
                echo "#PBS -q $pbsqueue" >> $qsub
                echo "#PBS -m ae" >> $qsub
                echo "#PBS -M $email" >> $qsub
		echo "aprun -n 1 -d $thr $scriptdir/novobam.sh $alignerdir $alignparms $thr $refdir/$refindexed $outputalign $outputsam.sam $outputsam.bam $samdir $paired $BAM $outputlogs/log.novoalnbam.$dirname.in $outputlogs/log.novoalnbam.$dirname.ou $email $outputlogs/qsub.novoalnbam.$dirname" >> $qsub
                
                #`chmod a+r $qsub`
                jobnovo=`qsub $qsub`
                `qhold -h u $jobnovo`
		echo $jobnovo >> $outputlogs/ALIGNED_$dirname
	      else
                echo "bwa is used as aligner. input file format is BAM"
                readparm="-b1"
                qsub1=$outputlogs/qsub.bwabamr1.$dirname
                echo "#PBS -N ${pipeid}_bwabamr1_${dirname}" >> $qsub1
		echo "#PBS -o $outputlogs/log.bwabamr1.$dirname.ou" >> $qsub1
		echo "#PBS -e $outputlogs/log.bwabamr1.$dirname.in" >> $qsub1
                echo "#PBS -A $pbsprj" >> $qsub1
		echo "#PBS -l epilogue=$epilogue" >> $qsub1
		echo "#PBS -l walltime=$pbscpu" >> $qsub1
		echo "#PBS -l nodes=1:ppn=$thr" >> $qsub1
                echo "#PBS -q $pbsqueue" >> $qsub1
                echo "#PBS -m ae" >> $qsub1
                echo "#PBS -M $email" >> $qsub1
		echo "aprun -n 1 -d $thr $scriptdir/bwabamS1.sh $alignerdir $alignparms $readparm $refdir/$refindexed $outputalign $outputsam.R1.sai $BAM $outputlogs/log.bwabamr1.$dirname.in $outputlogs/log.bwabamr1.$dirname.ou $email $outputlogs/qsub.bwabamr1.$dirname" >> $qsub1
                
                #`chmod a+r $qsub1`
                jobr1=`qsub $qsub1`
                `qhold -h u $jobr1`
                echo $jobr1 >> $outputlogs/ALIGNED_$dirname
                if [ $paired -eq 1 ]
                then
                    echo "bwa aligner. paired-end reads"
                    readparm="-b2"
		    qsub2=$outputlogs/qsub.bwabamr2.$dirname
		    echo "#PBS -N ${pipeid}_bwabamr2_${dirname}" >> $qsub2
		    echo "#PBS -o $outputlogs/log.bwabamr2.$dirname.ou" >> $qsub2
		    echo "#PBS -e $outputlogs/log.bwabamr2.$dirname.in" >> $qsub2
		    echo "#PBS -A $pbsprj" >> $qsub2
		    echo "#PBS -l epilogue=$epilogue" >> $qsub2
		    echo "#PBS -l walltime=$pbscpu" >> $qsub2
		    echo "#PBS -l nodes=1:ppn=$thr" >> $qsub2
		    echo "#PBS -q $pbsqueue" >> $qsub2
		    echo "#PBS -m ae" >> $qsub2
		    echo "#PBS -M $email" >> $qsub2
		    echo "aprun -n 1 -d $thr $scriptdir/bwabamS1.sh $alignerdir $alignparms $readparm $refdir/$refindexed $outputalign $outputsam.R2.sai $BAM $outputlogs/log.bwabamr2.$dirname.in $outputlogs/log.bwabamr2.$dirname.ou $email $outputlogs/qsub.bwabamr2.$dirname" >> $qsub2
                   
		    #`chmod a+r $qsub2`
                    jobr2=`qsub $qsub2`
                    `qhold -h u $jobr2`
		    echo $jobr2 >> $outputlogs/ALIGNED_$dirname

		    qsub3=$outputlogs/qsub.bwabamsampe.$dirname
		    echo "#PBS -N ${pipeid}_bwabamsampe_$dirname" >> $qsub3
		    echo "#PBS -o $outputlogs/log.bwabamsampe.${dirname}.ou" >> $qsub3
		    echo "#PBS -e $outputlogs/log.bwabamsampe.${dirname}.in" >> $qsub3
		    echo "#PBS -A $pbsprj" >> $qsub3
		    echo "#PBS -l epilogue=$epilogue" >> $qsub3
		    echo "#PBS -l walltime=$pbscpu" >> $qsub3
		    echo "#PBS -l nodes=1:ppn=$thr" >> $qsub3
		    echo "#PBS -q $pbsqueue" >> $qsub3
		    echo "#PBS -m ae" >> $qsub3
		    echo "#PBS -M $email" >> $qsub3
		    echo "#PBS -W depend=afterok:$jobr1:$jobr2" >> $qsub3
		    echo "aprun -n 1 -d $thr $scriptdir/bwabamS2.sh $alignerdir $refdir/$refindexed $outputalign $outputsam.R1.sai $outputsam.R2.sai $BAM $BAM  $outputsam.sam $outputsam.bam $samdir $outputlogs/log.bwabamsampe.${dirname}.in $outputlogs/log.bwabamsampe.${dirname}.ou $email $outputlogs/qsub.bwabamsampe.$dirname" >> $qsub3
                    
		    #`chmod a+r $qsub3`
                    jobsampe=`qsub $qsub3`
		    `qhold -h u $jobsampe`
		    echo $jobsampe >> $outputlogs/ALIGNED_$dirname
                else
                    echo "bwa aligner. single read"
		    qsub3=$outputlogs/qsub.bwabamsamse.$dirname
		    echo "#PBS -N ${pipeid}_bwabamsamse_$dirname" >> $qsub3
		    echo "#PBS -o $outputlogs/log.bwabamsamse.${dirname}.ou" >> $qsub3
		    echo "#PBS -e $outputlogs/log.bwabamsamse.${dirname}.in" >> $qsub3
		    echo "#PBS -A $pbsprj" >> $qsub3
		    echo "#PBS -l epilogue=$epilogue" >> $qsub3
		    echo "#PBS -l walltime=$pbscpu" >> $qsub3
		    echo "#PBS -l nodes=1:ppn=$thr" >> $qsub3
		    echo "#PBS -q $pbsqueue" >> $qsub3
		    echo "#PBS -m ae" >> $qsub3
		    echo "#PBS -M $email" >> $qsub3
		    echo "#PBS -W depend=afterok:$jobr1" >> $qsub3
		    echo "aprun -n 1 -d $thr $scriptdir/bwabamS3.sh $alignerdir $refdir/$refindexed $outputalign $outputsam.R1.sai $BAM  $outputsam.sam $outputsam.bam $samdir $outputlogs/log.bwabamsamse.${dirname}.in $outputlogs/log.bwabamsamse.${dirname}.ou $email $outputlogs/qsub.bwabamsamse.$dirname" >> $qsub3
                    
		    #`chmod a+r $qsub3`
                    jobsamse=`qsub $qsub3`
		    `qhold -h u $jobsamse`
                    echo $jobsamse >> $outputlogs/ALIGNED_$dirname
                fi
            fi
	    echo `date`


            set +x; echo -e "\n\n" >&2; 
            echo "####################################################################################################" >&2
            echo "##################################### merging chunks                   #############################" >&2
            echo "####################################################################################################" >&2
            echo -e "\n\n" >&2; set -x;

            listfiles="$outputsam.bam"
	    ALIGNED=$( cat $outputlogs/ALIGNED_$dirname | sed "s/\.[a-z]*//" | tr "\n" ":" )

            if [ $sortool == "NOVOSORT" ]
            then
		qsub1=$outputlogs/qsub.novosort.inbam.$dirname
		echo "#PBS -A $pbsprj" >> $qsub1
		echo "#PBS -N ${pipeid}_mergenovo_inbam.$dirname" >> $qsub1
		echo "#PBS -l epilogue=$epilogue" >> $qsub1
		echo "#PBS -l walltime=$pbscpu" >> $qsub1
		echo "#PBS -l nodes=1:ppn=$thr" >> $qsub1
		echo "#PBS -o $outputlogs/log.novosort.inbam.${dirname}.ou" >> $qsub1
		echo "#PBS -e $outputlogs/log.novosort.inbam.${dirname}.in" >> $qsub1
		echo "#PBS -q $pbsqueue" >> $qsub1
		echo "#PBS -m ae" >> $qsub1
		echo "#PBS -M $email" >> $qsub1
		echo "#PBS -W depend=afterok:$ALIGNED" >> $qsub1
		echo "aprun -n 1 -d $thr $scriptdir/mergenovo.sh $outputalign $listfiles $outsortwdup $outsortnodup $sortedplain $dupparms $RGparms $runfile $output_logs/log.novosort.inbam.${dirname}.in $output_logs/log.novosort.inbam.${dirname}.ou $email $output_logs/qsub.merge.novosort.inbam.$dirname" >> $qsub1
		#`chmod a+r $qsub1`
		mergejob=`qsub $qsub1`
                `qhold -h u $mergejob`
		echo $mergejob  >> $outputlogs/MERGED_$dirname
            else
		qsub1=$outputlogs/qsub.sortmerge.picard.inbam.$dirname
		echo "#PBS -A $pbsprj" >> $qsub1
		echo "#PBS -N ${pipeid}_sortmergepicard_$dirname" >> $qsub1
		echo "#PBS -l epilogue=$epilogue" >> $qsub1
		echo "#PBS -l walltime=$pbscpu" >> $qsub1
		echo "#PBS -l nodes=1:ppn=$thr" >> $qsub1
		echo "#PBS -o $outputlogs/log.sortmerge.picard.inbam.${dirname}.ou" >> $qsub1
		echo "#PBS -e $outputlogs/log.sortmerge.picard.inbam.${dirname}.in" >> $qsub1
		echo "#PBS -q $pbsqueue" >> $qsub1
		echo "#PBS -m ae" >> $qsub1
		echo "#PBS -M $email" >> $qsub1
		echo "#PBS -W depend=afterok:$ALIGNED" >> $qsub1
		echo "aprun -n 1 -d $thr $scriptdir/mergepicard.sh $outputalign $listfiles $outsortwdup $outsortnodup $sortedplain $dupparms $RGparms $runfile $outputlogs/log.sortmerge.picard.inbam.${dirname}.in $outputlogs/log.sortmerge.picards.inbam.${dirname}.ou $email $outputlogs/qsub.sortmerge.picard.inbam.$dirname" >> $qsub1
		#`chmod a+r $qsub1`
		mergejob=`qsub $qsub1`
		`qhold -h u $mergejob`
		echo $mergejob  >> $outputlogs/MERGED_$dirname
            fi


            set +x; echo -e "\n\n" >&2; 
            echo "####################################################################################################" >&2
            echo "##############  extracting regions that will not be included in analysis         ###################" >&2
            echo "####################################################################################################" >&2
            echo -e "\n\n" >&2; set -x;

	    echo `date`
	    echo "extract reads specified in CHRINDEX param"
	    qsub5=$outputlogs/qsub.extractreadsbam.$dirname
	    echo "#PBS -A $pbsprj" >> $qsub5
	    echo "#PBS -N ${pipeid}_extrabam_${dirname}" >> $qsub5
            echo "#PBS -l epilogue=$epilogue" >> $qsub5
	    echo "#PBS -l walltime=$pbscpu" >> $qsub5
	    echo "#PBS -l nodes=1:ppn=$thr" >> $qsub5
	    echo "#PBS -o $outputlogs/log.extractreadsbam.$dirname.ou" >> $qsub5
	    echo "#PBS -e $outputlogs/log.extractreadsbam.$dirname.in" >> $qsub5
	    echo "#PBS -q $pbsqueue" >> $qsub5
	    echo "#PBS -m ae" >> $qsub5
	    echo "#PBS -M $email" >> $qsub5
	    echo "#PBS -W depend=afterok:$mergejob" >> $qsub5
	    echo "aprun -n 1 -d $thr $scriptdir/extract_reads_bam.sh $outputalign $outsortwdup $runfile $outputlogs/log.extractreadsbam.$dirname.in $outputlogs/log.extractreadsbam.$dirname.ou $email  $outputlogs/qsub.extractreadsbam.$dirname $igv $extradir" >> $qsub5
	    #`chmod a+r $qsub5`
	    extrajob=`qsub $qsub5`
            `qhold -h u $extrajob`
            echo $extrajob >> $output_logs/pbs.EXTRACTREADS
            cat $outputlogs/MERGED_$dirname >> $output_logs/pbs.MERGED
	    cat $outputlogs/ALIGNED_$dirname >> $output_logs/pbs.ALIGNED
          fi
	done < $samplefileinfo


     set +x; echo -e "\n\n" >&2; 
     echo "####################################################################################################" >&2
     echo "#####################################          OUT OF THE LOOP         #############################" >&2
     echo "####################################################################################################" >&2
     echo "#####################################     WRAP-UP IF ANALYSIS ENDS HERE         ####################" >&2
     echo "####################################################################################################" >&2
     echo -e "\n\n" >&2; set -x;

     pbsids=$( cat $output_logs/pbs.MERGED | sed "s/\.[a-z]*//" | tr "\n" ":" )
     extraids=$( cat $output_logs/pbs.EXTRACTREADS | sed "s/\.[a-z]*//" | tr "\n" " " )
     mergeids=$( echo $pbsids | tr ":" " " )
     alignids=$( cat $output_logs/pbs.ALIGNED | sed "s/\.[a-z]*//" | tr "\n" " " )
     echo $pbsids >> $output_logs/ALIGN_NCSA_jobids

     if [ $analysis == "ALIGNMENT" -o $analysis == "ALIGN" -o $analysis == "ALIGN_ONLY" ]
     then
        echo "analysis ends here, produce summary report for redmine and end pipeline"
        `qrls -h u $alignids`
        `qrls -h u $mergeids`
        `qrls -h u $extraids`
        
	 lastjobid=""
         cleanupjobid=""
         if [ $cleanupflag == "YES" ]
         then 
	     qsub6=$output_logs/qsub.cleanup.align
	     echo "#PBS -A $pbsprj" >> $qsub6
	     echo "#PBS -N ${pipeid}_cleanup_aln" >> $qsub6
	     echo "#PBS -l epilogue=$epilogue" >> $qsub6
	     echo "#PBS -l walltime=$pbscpu" >> $qsub6
	     echo "#PBS -l nodes=1:ppn=1" >> $qsub6
	     echo "#PBS -o $output_logs/log.cleanup.align.ou" >> $qsub6
	     echo "#PBS -e $output_logs/log.cleanup.align.in" >> $qsub6
	     echo "#PBS -q $pbsqueue" >> $qsub6
	     echo "#PBS -m ae" >> $qsub6
	     echo "#PBS -M $email" >> $qsub6
	     echo "#PBS -W depend=afterok:$pbsids" >> $qsub6
	     echo "aprun -n 1 -d 1 $scriptdir/cleanup.sh $outputdir $analysis $output_logs/log.cleanup.align.in $output_logs/log.cleanup.align.ou $email $output_logs/qsub.cleanup.align"  >> $qsub6
	     #`chmod a+r $qsub6`
	     cleanjobid=`qsub $qsub6`
	     echo $cleanjobid >> $outputdir/logs/pbs.CLEANUP
         fi

         `sleep 30s`

	 qsub4=$output_logs/qsub.summary.aln.allok
	 echo "#PBS -A $pbsprj" >> $qsub4
	 echo "#PBS -N ${pipeid}_summaryok" >> $qsub4
	 echo "#PBS -l epilogue=$epilogue" >> $qsub4
	 echo "#PBS -l walltime=$pbscpu" >> $qsub4
	 echo "#PBS -l nodes=1:ppn=1" >> $qsub4
	 echo "#PBS -o $output_logs/log.summary.aln.ou" >> $qsub4
	 echo "#PBS -e $output_logs/log.summary.aln.in" >> $qsub4
	 echo "#PBS -q $pbsqueue" >> $qsub4
	 echo "#PBS -m ae" >> $qsub4
	 echo "#PBS -M $email" >> $qsub4
         if [ `expr ${#cleanjobid}` -gt 1 ]
         then
	     echo "#PBS -W depend=afterok:$cleanjobid" >> $qsub4
         else
	     echo "#PBS -W depend=afterok:$pbsids" >> $qsub4
         fi
	 echo "aprun -n 1 -d 1 $scriptdir/summary.sh $runfile $email exitok"  >> $qsub4
	 #`chmod a+r $qsub4`
	 lastjobid=`qsub $qsub4`
	 echo $lastjobid >> $output_logs/pbs.SUMMARY

	 if [ `expr ${#lastjobid}` -lt 1 ]
	 then
             echo "at least one job aborted"
	     qsub5=$output_logs/qsub.summary.aln.afterany
	     echo "#PBS -A $pbsprj" >> $qsub5
	     echo "#PBS -N ${pipeid}_summary_afterany" >> $qsub5
	     echo "#PBS -l epilogue=$epilogue" >> $qsub5
	     echo "#PBS -l walltime=$pbscpu" >> $qsub5
	     echo "#PBS -l nodes=1:ppn=1" >> $qsub5
	     echo "#PBS -o $output_logs/log.summary.aln.afterany.ou" >> $qsub5
	     echo "#PBS -e $output_logs/log.summary.aln.afterany.in" >> $qsub5
	     echo "#PBS -q $pbsqueue" >> $qsub5
	     echo "#PBS -m ae" >> $qsub5
	     echo "#PBS -M $email" >> $qsub5
	     echo "#PBS -W depend=afterany:$pbsids" >> $qsub5
	     echo "aprun -n 1 -d 1 $scriptdir/summary.sh $runfile $email exitnotok"  >> $qsub5
	     #`chmod a+r $qsub5`
	     badjobid=`qsub $qsub5`
	     echo $badjobid >> $output_logs/pbs.SUMMARY
	 fi
     fi

     set +x; echo -e "\n\n" >&2; 
     echo "####################################################################################################" >&2
     echo "########################     CALL THE NEXT MODULE IF ANALYSIS CONTINUES         ####################" >&2
     echo "####################################################################################################" >&2
     echo -e "\n\n" >&2; set -x;

     if [ $analysis == "REALIGNMENT" -o $analysis == "REALIGN" ]
     then
            echo " analysis continues with realignment"
	    qsub2=$output_logs/qsub.start_realrecal_block
	    echo "#PBS -A $pbsprj" >> $qsub2
	    echo "#PBS -N ${pipeid}_START_REALRECAL_BLOCK" >> $qsub2
	    echo "#pbs -l epilogue=$epilogue" >> $qsub2
	    echo "#PBS -l walltime=$pbscpu" >> $qsub2
	    echo "#PBS -l nodes=1:ppn=1" >> $qsub2
	    echo "#PBS -o $output_logs/start_realrecal_block.ou" >> $qsub2
	    echo "#PBS -e $output_logs/start_realrecal_block.in" >> $qsub2
	    echo "#PBS -q $pbsqueue" >> $qsub2
	    echo "#PBS -m ae" >> $qsub2
	    echo "#PBS -M $email" >> $qsub2
            echo "#PBS -W depend=afterany:$pbsids" >> $qsub2
	    echo "aprun -n 1 -d 1 $scriptdir/start_realrecal_block.sh $runfile $output_logs/start_realrecal_block.in $output_logs/start_realrecal_block.ou $email $output_logs/qsub.start_realrecal_block" >> $qsub2
	    #`chmod a+r $qsub2` 
	    `qsub $qsub2 >> $output_logs/pbs.REALRECAL`

            # releasing held jobs; else the above job dependency will never be met
            `qrls -h u $alignids`
            `qrls -h u $mergeids`
            `qrls -h u $extraids`
	    echo `date`
      fi

     #`chmod -R 770 $outputdir`
     #`chmod -R 770 $output_logs`
     echo `date`

fi
