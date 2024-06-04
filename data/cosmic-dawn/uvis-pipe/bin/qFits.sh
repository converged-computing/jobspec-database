#!/bin/sh
#PBS -S /bin/sh
#PBS -N qFits_@FILTER@_@ID@
#PBS -o qFits_@ID@.out
#PBS -j oe
#PBS -l nodes=1:ppn=7,walltime=24:00:00
#-----------------------------------------------------------------------------
# pseudo qFits: 
# requires: astromatic s/w, dfits, python
# Dec.2019: add ldac saturation flagging ... early version
#-----------------------------------------------------------------------------
# Nov.2022: revisions for DR6:
# 
# 1. get sky stats (mean/medi/casu, rms) of input file;
#    compare rms of top and bottom chips to look for anomalies
#    ==> v##_##_sky-stats.dat, v##_##_sky_histo.png
# 2. SExtractor to identify cosmics
#    ==> v##_##_cosmic.fits, for use in building weight, then deleted
# 3. WW to build weight file
#    ==> v##_##_weight.fits, 
#        v##_##_wgtflag.fits ... don't recall why, immediately deleted
# 4. SExtractor for PSFEx
#    ==> v##_##_psfex.ldac
# 5. extract useful data (fwhm, elli, Nstars) from xml file
#    ==> v##_##_psfex.xml
#    ==> v##_##_psfex.dat
#    delete v##_##_psfex.ldac
# 6. SExtractor for scamp
#    ==> v##_##.ldac 
#        v##_##_objects.fits   SEx OBJECTS checkimage for saturation studies
# 7. extract number of stars used (per chip)
#    ==> v##_##_nstars.dat
# 8. determine saturation level (per chip): extract sky level and max pix value
#    ==> v##_##_saturation-data.dat
#        v##_##_saturation-data.png
# 9. actual flagging of saturation levels in ldac fils
#    NOT YET FINALISED
# 
#-----------------------------------------------------------------------------

set -u  
umask 022
export PATH="~/bin:$PATH:/softs/dfits/bin:/softs/astromatic/bin"

module () {  eval $(/usr/bin/modulecmd bash $*); }
module purge ; module load intelpython/3-2020.2
export LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH}

sdate=$(date "+%s")
ecode=0

ec() { echo "$(date "+[%d.%h.%y %T"]) $1 "; }    # echo with date
ecn() { echo -n "$(date "+[%d.%h.%y %T"]) $1 "; }    # echo with date

#-----------------------------------------------------------------------------
# Some variables
#-----------------------------------------------------------------------------
module=qFits
uvis=/home/moneti/softs/uvis-pipe
bindir=$uvis/bin
pydir=$uvis/python                # python scripts
qFitsdir=$uvis/config/qFits        # config dir
export confdir=$qFitsdir

export PYTHONPATH=$pydir

#-----------------------------------------------------------------------------

node=$(hostname)   # NB: compute nodes don't have .iap.fr in name

# check  if run via shell or via qsub:
if [[ "$0" =~ "$module" ]]; then
    ec "$module: running as shell script on $node"
	list=$1
	WRK=$WRK
	FILTER=$FILTER
	if [[ "${@: -1}" =~ 'dry' ]]; then dry=T; else dry=F; fi
else
    ec "$module: running via qsub (from pipeline) on $node with 8 threads"
	dry=@DRY@
	WRK=@WRK@
	list=@LIST@
	FILTER=@FILTER@
fi

verb=" -VERBOSE_TYPE QUIET"
quiet=" -VERBOSE_TYPE QUIET"
satlevel=28000       # Saturation level for SExtractor

#-----------------------------------------------------------------------------
cd $WRK/images

if [ $? -ne 0 ]; then ec "#### ERROR: $WRK/images not found ... quitting"; exit 10; fi
if [ ! -s $list ]; then ec "#### ERROR: $list not found in $WRK/images ... quitting"; exit 10; fi

nl=$(cat $list | wc -l)
ec "=================================================================="
ec " >>>>  Begin qFits $list with $nl entries on $node  <<<<"
ec "------------------------------------------------------------------"
ecn " In     "; echo $WRK
ecn " Using  "; ww -v
ecn " Using  "; sex -v
ecn " Using  "; psfex -v
ecn " pydir is   "; echo $pydir
ecn " bindir is  "; echo $bindir
ecn " confdir is "; echo $qFitsdir
ec "------------------------------------------------------------------"

cd $WRK/images
info=$(mktemp)
logfile=$(echo $list | cut -d\. -f1).log

for ima in $(cat $list); do

   bdate=$(date "+%s")

   root=${ima%.fits}
   if [ ! -e $ima ]; then ln -s origs/$ima . ; fi

   grep $ima ../FileInfo.dat | tr -s ' ' > $info  # get associated files
   flat=../calib/$(cut -d\  -f4 $info)          # flatfield
   norm=${flat%.fits}_norm.fits                 # normalised flat
   bpm=/n09data/UltraVista/DR6/bpms/$(cut -d\  -f5 $info)           # bpm mask

   pb=0
   ec " File is $ima"
   if [ ! -s $flat ]; then ec "#### ERROR: $flat not found;"; pb=1; else ec " Flat is $flat"; fi
   if [ ! -s $norm ]; then ec "#### ERROR: $norm not found;"; pb=1; else ec " Norm is $norm"; fi
#   if [ ! -s $bpm  ]; then ec "#### ERROR: $bpm  not found;"; pb=1; else ec " bpm  is $bpm "; fi
   if [ $pb -ge 1 ]; then ec " ... quitting ..."; exit 10; fi

   #-----------------------------------------------------------------------------
   # 1. stats of sky 
   #-----------------------------------------------------------------------------

   datfile=${root}_sky-stats.dat
   pngfile=${root}_sky-histo.png
   
   if [ ! -e origs/$datfile ]; then
	  statscomm="$pydir/sky_stats.py $ima"
      ec ""; ec ">>>> 1. sky stats on "$ima
	   
	  if [ $dry == 'T' ]; then
		 echo $statscomm
		 echo " ## DRY MODE - do nothing ## "
      else
      	 $statscomm
		 mv $datfile $pngfile origs
	  fi
   else
	  ec "# Attn: found $datfile ... skip sky stats"
   fi

   #-----------------------------------------------------------------------------
   # 2. sex to get cosmics and ww for weight
   #-----------------------------------------------------------------------------

   cosmic=${root}_cosmic.fits       # ; touch $cosmic
   weight=${root}_weight.fits       # output ... later
   wflag=${root}_wgtflag.fits
   
   if [ ! -e weights/$weight ]  ; then    # cosmics and ww to build weight file
       args=" -c sex_cosmic.config -PARAMETERS_NAME sex_cosmic.param \
         -FILTER_NAME vircam.ret  -CHECKIMAGE_NAME $cosmic  -CATALOG_TYPE NONE \
         -SATUR_KEY TOTO -SATUR_LEVEL $satlevel -WRITE_XML N  "

      coscomm="sex ${root}.fits $args  $quiet"
      ec ""; ec ">>>> 1. SEx for cosmics for "$ima
	  logfile=${root}_se_cosmic.log
	  errfile=${root}_se_cosmic.err


	  if [ $dry == 'T' ]; then
		 echo $coscomm
		 echo " ## DRY MODE - do nothing ## "
      else
      	 $coscomm 1> $errfile 2> $logfile
		 if [ $? -ne 0 ]; then 
			ec "#### ERROR in SEx for cosmics for $root .... continue ####"
			echo $coscomm
			ecode=1
		 else
      	    ec " ==> $cosmic built ..."
		    # Clean up logfile:
		    grep -v -e Setting -e Line: $logfile > x$root; mv x$root $logfile
		    if [ ! -s $errfile ]; then rm $errfile; fi
		    if [ ! -s $logfile ]; then rm $logfile; fi  
		 fi
      fi

	  # Now setup for ww
      # arguments
      args=" -c ww_weight.config  -OUTWEIGHT_NAME $weight -OUTFLAG_NAME $wflag\
         -WEIGHT_NAMES $norm,$ima,$cosmic,$bpm  -WRITE_XML N \
         -WEIGHT_MIN 0.7,10,-0.1,0.5 -WEIGHT_MAX 1.3,30000,0.1,1.5  \
         -WEIGHT_OUTFLAGS 0,1,2,4 " 
      
      wwcomm=" ww $args $quiet" 
	  ec ""; ec ">>>> 2. WW for weight for "$ima
	  logfile=${root}_weight.log
	  errfile=${root}_weight.err

      if [ $dry == 'T' ]; then
		 echo "$wwcomm"
		 echo " ## DRY MODE - do nothing ## "
      else
		 echo "$wwcomm" > $logfile 
      	 $wwcomm    1>> $logfile 2> $errfile   
		 if [ $? -ne 0 ]; then 
			ec "#### ERROR in WW for $root ... continue ####"
			echo $wwcomm
			ecode=2
		 else
      	    ec " ==> $(ls $weight) built ... " 
		    ecn ""; $pydir/cp_astro_kwds.py -i $ima -s _weight # >> $logfile
		    if [ ! -s $errfile ]; then rm -f $errfile; fi 
		    if [ ! -s $logfile ]; then rm -f $logfile; fi  
		    rm -f $wflag $cosmic
		 fi
	  fi
   else
	  ec "# Attn: found $weight ... skip WW for weight file"
	  weight=weights/$weight    # change link to weight file
   fi

   #-----------------------------------------------------------------------------
   # 3. SExtractor for psfex and PSFEx
   #-----------------------------------------------------------------------------

   pdac=${root}_psfex.ldac          #   ; touch $pdac
   psfx=${root}_psfex.xml           #   ; touch $psfx

#   satlevel=28000
   if [ ! -e xml/$psfx ]; then 
      args=" -c sex_psfex.config  -PARAMETERS_NAME sex_psfex.param   \
             -BACK_SIZE 128 -BACK_FILTERSIZE 3  -CATALOG_NAME $pdac  -CHECKIMAGE_TYPE NONE \
             -WEIGHT_IMAGE $weight   -WRITE_XML N \
             -STARNNW_NAME default.nnw  -FILTER_NAME gauss_3.0_7x7.conv  \
             -DETECT_THRESH 10. -ANALYSIS_THRESH 5. -SATUR_KEY TOTO -SATUR_LEVEL $satlevel "

      psexcomm="sex $ima $args $verb"
      ec ""; ec ">>>> 3. SEx for PSFEx for "$ima
	  logfile=${root}_se_psfex.log
	  errfile=${root}_se_psfex.err

      if [ $dry == 'T' ]; then
		 echo $psexcomm
		 echo " ## DRY MODE - do nothing ## "
      else
		 echo $psexcomm >> $logfile
      	 $psexcomm  1> $errfile 2>> $logfile
		 if [ $? -ne 0 ]; then 
			ec "#### ERROR in SEx for PSFEx for $root ... continue ####"
			echo $psexcomm
			ecode=3
		 else
      	    ec " ==> $pdac built ..."    
		    # Clean up logfile:
			grep -v -e Setting -e Line: $logfile > x$root; mv x$root $logfile
			if [ ! -s $errfile ]; then rm $errfile; fi  
		 fi
      fi

      #-----------------------------------------------------------------------------
      # 4. PSFEx - for PSF stats only; don't need more
      #-----------------------------------------------------------------------------

      args=" -c psfex.config  -WRITE_XML Y -XML_NAME $psfx \
             -CHECKPLOT_TYPE NONE  -CHECKIMAGE_TYPE NONE  -NTHREADS 2"
      psfcomm="psfex $pdac  $args " # $verb"
      ec ""; ec ">>>> 4. PSFEx for "$ima
	  logfile=${root}_psfex.log

      if [ $dry == 'T' ]; then
		 echo $psfcomm
		 echo " ## DRY MODE - do nothing ## "
      else
		 echo $psfcomm > $logfile
         $psfcomm  1> $logfile 2>&1  # $logfile
		 if [ $? -ne 0 ]; then 
			ec "#### ERROR in PSFEx for $root ... continue ####" 
			echo $psfcomm
			mv $logfile logs
			ecode=4
		 else
		    # fix an error in the xml file
		    sed -i 's/NStars_Accepted_Mean" datatype="int"/NStars_Accepted_Mean" datatype="float"/' $psfx
      	    ec " ==> $psfx  built ... "   
		    if [ ! -s $logfile ]; then rm -f $logfile; fi  
		    $pydir/psfxml2dat.py $psfx
		    mv $psfx ${psfx%.xml}.dat xml   
		    rm $pdac ${root}_psfex.psf  $logfile			
		 fi
      fi
   else
	  ec "# Attn: found $psfx ..... skip PSFEx"
   fi

   #-----------------------------------------------------------------------------
   # 5. SExtractor - for scamp
   #    NB: output ldac file is named _orig.ldac and it mod is set to 444; hence
   #        must be deleted prior to eventual rerun of this step.
   #-----------------------------------------------------------------------------

   ldac=${root}_orig.ldac                 # 20.feb.23: set _orig in name of output
   datfile=ldacs/${root}_nstars.dat
   objects=objects/${root}_objects.fits          # not used
   satfile=objects/${root}_saturation-data.dat   # not used
   config=$qFitsdir/sex_scamp_dr6.config

   satlevel=28000   

   if [ ! -e ldacs/$ldac ]; then 
      args=" -CATALOG_NAME $ldac  -WEIGHT_IMAGE $weight \
        -DETECT_THRESH 5.  -SATUR_LEVEL $satlevel "
	    checks=" -CHECKIMAGE_TYPE OBJECTS  -CHECKIMAGE_NAME $objects"
      
      sexcomm="sex $ima -c $config  $args -VERBOSE_TYPE NORMAL -MEMORY_BUFSIZE 4096"  #$verb" 
      ec ""; ec ">>>> 5. SEx for scamp for "$ima
	  logfile=${root}_se_scamp.log
	  errfile=${root}_se_scamp.err 
	  
      if [ $dry == 'T' ]; then
		 ec "$sexcomm"
		 echo " ## DRY MODE - do nothing ## "
       else
		 ec "$sexcomm" >> $logfile
      	 $sexcomm  1> $errfile 2>> $logfile
		 if [ $? -ne 0 ]; then 
			ec "#### ERROR in SEx for scamp for $root ... continue ####"
			echo $sexcomm
			ecode=5
		 else
      	    ec " ==> $ldac built ... " 
		    # Clean up logfile:
			grep -v -e Setting -e Line: $logfile > x$root; mv x$root $logfile
			if [ ! -s $errfile ]; then rm $errfile; fi  
			
		    # build Nstars files  
			echo "## Num stars in $ldac" > $datfile
			dfits -x 0 $ldac | grep NAXIS2 | \
				awk 'BEGIN{printf "Nstars "}{getline; printf "%7i ",$3}; END{print " "}' >> $datfile

#			$pydir/ldac2region.py $ldac; mv ${root}.reg regs 

            # weight file no longer needed; should be sym link - remove it
			if [ -h $weight ]; then 
			   rm $weight          # rm it if symbolic link
			fi

			chmod 444 $ldac; mv $ldac ldacs     ###### do it here for now ....
		 fi
      fi
   else
	  ec "# Attn: found $ldac ... skip SEx for scamp"
   fi


   doFlag=True    # run python script to flag saturated source in _orig.ldac
   if [ $doFlag == "True" ]; then 
      #-----------------------------------------------------------------------------
      # 6. Flag saturated sources in ldac for scamp
	  #    NB. ldac file should be in ldacs dir by now.  
      #-----------------------------------------------------------------------------
      odac=ldacs/$ldac                        # original (_orig.ldac) ldac file
	  ldac=ldacs/${root}.ldac                 # v20??????_??????.ldac base name; which is flagged in place
	  # copy the original to its base name, and that's what we work on
      if [ ! -e $ldac ]; then cp $odac $ldac ; chmod 644 $ldac; fi 

#      sdac=ldacs/${ldac%.ldac}.ldac      # for output file
      # make a backup copy of the original

      if [ ! -e $ldac ]; then 
   		  satcomm="$pydir/flag_saturation.py  $ldac " 
		  ec ""; ec ">>>> 6. Flag saturated sources in "$ldac
   		  ec "$satcomm"
          if [ $dry == 'T' ]; then
   			  echo " ## DRY MODE - do nothing ## "
          else
   			  $satcomm 
   			  if [ $? -ne 0 ]; then ec "ERROR ... quitting"; exit 6; fi
   			  ec " ==> ${ldac} built ... "  #; sleep 1
#			  mv $sdac 
          fi
      else
   		  ec "# Attn: found ${ldac} ... skip flag saturation in ldac"
      fi
   else
	   ec "# Attn: flag saturation in ldacs not yet implemented"
   fi

   #-----------------------------------------------------------------------------
   # Summary files
   #-----------------------------------------------------------------------------

   doSumm=True
   if [ $doSumm == "True" ]; then 
      #-----------------------------------------------------------------------------
      # Build summary dat file
      #-----------------------------------------------------------------------------
      
      if [ $dry == 'T' ]; then
      	  echo " ## DRY MODE - do nothing ## "
      else
      	  summary=qFits/${root}_qFits.dat
      	  if [ ! -e $summary ]; then
            echo "#  qFits summary file $root, filter $FILTER " > $summary
            echo "chip        1       2       3       4       5       6       7       8       9      10      11      12      13      14      15      16" >> $summary
            echo "------------------------------------------------------- sky stats: kappa = 4 ---------------------------------------------------------" >> $summary
            grep -v -e chip -e filter origs/${root}_sky-stats.dat >> $summary
            echo "------------------------------------------------------- PSF info: fwhm, elli ---------------------------------------------------------" >> $summary
            grep -v PSF xml/${root}_psfex.dat >> $summary
            echo "------------------------------------------------- Nstars and saturation info info ----------------------------------------------------" >> $summary
            grep Nstars ldacs/${root}_nstars.dat  >> $summary
#            grep satlev objects/${root}_saturation-data.dat  >> $summary
            echo "--------------------------------------------------------------------------------------------------------------------------------------" >> $summary
      	  fi
      fi
   fi
   #-----------------------------------------------------------------------------
   # Clean up and finish
   #-----------------------------------------------------------------------------
   if [ $dry == 'T' ]; then
	   echo " ## DRY MODE - do nothing ## "
   else
       if [ -e $ima ]; then rm $ima; fi
       mv ${root}*.log logs 2>/dev/null
   fi

   edate=$(date "+%s"); dt=$(($edate - $bdate))
   ec "------------------------------------------------------------------"
   ec " >>>> Done - runtime: $dt sec  <<<<"
   ec "------------------------------------------------------------------"
   ec ""
done


#-----------------------------------------------------------------------------

edate=$(date "+%s"); dt=$(($edate - $sdate))
rm $info
ec " >>>> qFits finished - total runtime: $dt sec  <<<<"
ec "------------------------------------------------------------------"
echo ""
exit $ecode

#-----------------------------------------------------------------------------
# to cleanup:
rm qFits*.* images/v20*_0????_*.* images/v20*_00???.ldac images/v20*_00???.head
