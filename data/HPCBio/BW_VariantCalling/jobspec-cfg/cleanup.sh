#!/bin/bash
redmine=hpcbio-redmine@igb.illinois.edu
##redmine=lmainzer@igb.illinois.edu
##redmine=grendon@illinois.edu
if [ $# != 6 ]
then
        MSG="parameter mismatch"
        echo -e "jobid:${PBS_JOBID}\nprogram=$0 stopped at line=$LINENO.\nReason=$MSG" 
        #echo -e "jobid:${PBS_JOBID}\nprogram=$0 stopped at line=$LINENO.\nReason=$MSG" | ssh iforge "mailx -s '[Support #200] variant identification pipeline' "$redmine""
        exit 1;
else
	set -x
	echo `date`
        umask 0027
        scriptfile=$0
        outputdir=$1
        analysis=$2
        elog=$3
        olog=$4
        email=$5
        qsubfile=$6
        LOGS="jobid:${PBS_JOBID}\nqsubfile=$qsubfile\nerrorlog=$elog\noutputlog=$olog"


        if [ ! -d $outputdir ]
	then
            MSG="$outputdir results directory not found"
	   echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" 
	   #echo -e "program=$scriptfile stopped at line=$LINENO.\nReason=$MSG\n$LOGS" | ssh iforge  "mailx -s '[Support #200] variant identification pipeline' "$redmine,$email""
            exit 1;	    
        fi

        cd $outputdir

        if [ $analysis == "ALIGN" -o $analysis == "ALIGNMENT" ]
        then
            echo "analysis=$analysis. cleanup of align directory only"
            `find ./align -name "read*" | awk '{print "rm "$1}' | sh -x`
            `find ./align -name "*.node*" | awk '{print "rm "$1}' | sh -x`
            `find ./input -name "*.preprocessed.fastq" | awk '{print "rm "$1}' | sh -x`
        fi
        if [ $analysis == "REALIGNONLY" -o $analysis == "REALIGN_ONLY" ]
        then
            echo "analysis=$analysis. cleanup of realign directory only"
            `find ./realign -name "*.bed" | awk '{print "rm "$1}' | sh -x`
            `find ./realign -name "*.wdups.sorted.*" | awk '{print "rm "$1}' | sh -x`
            `find ./realign -name "realign.chr*" | awk '{print "rm "$1}' | sh -x`
            `find ./realign -name "recal.chr*" | awk '{print "rm "$1}' | sh -x`
        fi
        if [ $analysis == "REALIGN" -o $analysis == "REALIGNMENT" ]
        then
            echo "analysis=$analysis. cleanup of align and realign directories"
            `find ./align -name "read*" | awk '{print "rm "$1}' | sh -x`
            `find ./align -name "*.node*" | awk '{print "rm "$1}' | sh -x`
            `find ./align -name "*.wrg.*" | awk '{print "rm "$1}' | sh -x`
            `find ./realign -name "*.bed" | awk '{print "rm "$1}' | sh -x`
            `find ./realign -name "*.wdups.sorted.*" | awk '{print "rm "$1}' | sh -x`
            `find ./realign -name "realign.chr*" | awk '{print "rm "$1}' | sh -x`
            `find ./realign -name "recal.chr*" | awk '{print "rm "$1}' | sh -x`
        fi
        if [ $analysis == "VCALL" -o $analysis == "VCALL_ONLY" ]
        then
            cd $outputdir/variant
            `rm *.vcf`
            `rm *.vcf.idx`
            `rm *.bed`
        fi
        echo `date`
fi
