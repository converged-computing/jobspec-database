#!/bin/bash 

##SBATCH --account=chem-var
##SBATCH --qos=debug
##SBATCH --ntasks=40
##SBATCH --cpus-per-task=10
##SBATCH --time=5
##SBATCH --job-name="bashtest"
##SBATCH --exclusive

###############################################################
### Environmental variables defined in .xml file
set -x
HOMEgfs=${HOMEgfs:-"/home/Bo.Huang/JEDI-2020/expRuns/exp_UFS-Aerosols/UFS-Aerosols_JEDI-AeroDA-1C192-20C192_NRT"}
HOMEjedi=${HOMEjedi:-"/scratch1/BMC/gsd-fv3-dev/MAPP_2018/bhuang/JEDI-2020/JEDI-FV3/expCodes/fv3-bundle/V20230312/build/"}
EXPDIR=${EXPDIR:-"/home/Bo.Huang/JEDI-2020/UFS-Aerosols_NRTcyc/UFS-Aerosols_JEDI-AeroDA-1C192-20C192_NRT/dr-work/"}
TASKRC=${TASKRC:-"/home/Bo.Huang/JEDI-2020/UFS-Aerosols_NRTcyc/UFS-Aerosols_JEDI-AeroDA-1C192-20C192_NRT/dr-work/TaskRecords/cmplCycle_misc.rc"}
PSLOT=${PSLOT:-"UFS-Aerosols_JEDI-AeroDA-1C192-20C192_NRT"}
CDUMP=${CDUMP:-"gdas"}
CDATE=${CDATE:-"2023062500"}
CYCINTHR=${CYCINTHR:-"6"}
CASE=${CASE_OBS:-"C192"}

AODTYPE=${AODTYPE:-"NOAA_VIIRS"}
AODSAT=${AODSAT:-"npp j01"}
OBSDIR_NESDIS=${OBSDIR_NESDIS:-"/scratch2/BMC/public/data/sat/nesdis/viirs/aod/conus/"}
OBSDIR_NRT=${OBSDIR_NRT:-"/scratch1/BMC/gsd-fv3-dev/MAPP_2018/bhuang/JEDI-2020/JEDI-FV3/NRTdata_UFS-Aerosols/AODObs"}
MISS_NOAA_NPP_RECORD=${MISS_NOAA_NPP_RECORD:-"${HOMEgfs}/dr-work/record.miss_NOAAVIIRSnpp"}
MISS_NOAA_J01_RECORD=${MISS_NOAA_J01_RECORD:-"${HOMEgfs}/dr-work/record.miss_NOAAVIIRSj01"}
NDATE=${NDATE:-"/scratch2/NCEPDEV/nwprod/NCEPLIBS/utils/prod_util.v1.1.0/exec/ndate"}
mkdir -p ${OBSDIR_NRT}

#Define JEDI -related executables
VIIRSIODA1_EXEC=${HOMEgfs}/exec/viirs2ioda.x
#VIIRSIODA1_EXEC=/scratch1/BMC/gsd-fv3-dev/MAPP_2018/pagowski/exec/viirs2ioda.x
IODA1_IODA2_EXEC=${HOMEjedi}/bin/ioda-upgrade-v1-to-v2.x
IODA2_IODA3_EXEC=${HOMEjedi}/bin/ioda-upgrade-v2-to-v3.x
IODA2_IODA3_OBSYAML=${HOMEjedi}/../fv3-bundle/ioda/share/ioda/yaml/validation/ObsSpace.yaml

AODOUTDIR=${OBSDIR_NRT}/${CDATE}/
[[ ! -d ${AODOUTDIR} ]] && mkdir -p ${AODOUTDIR}

. ${HOMEjedi}/jedi_module_base.hera.sh
module load nco
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOMEjedi}/lib/"
status=$?
[[ $status -ne 0 ]] && exit $status

export LD_LIBRARY_PATH="/home/Mariusz.Pagowski/MAPP_2018/libs/fortran-datetime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/scratch2/BMC/wrfruc/Samuel.Trahan/viirs-thinning/mpiserial/exec:$PATH"
# Make sure we have the required executables
for exe in mpiserial ncrcat ; do
    if ( ! which "$exe" ) ; then
         echo "Error: $exe is not in \$PATH. Go find it and rerun." 1>&2
         #if [[ $ignore_errors == NO ]] ; then exit 1 ; fi
    fi
done
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # Must match --cpus-per-task in job card
export OMP_STACKSIZE=128M # Should be enough; increase it if you hit the stack limit.

STMP="/scratch2/BMC/gsd-fv3-dev/NCEPDEV/stmp3/$USER/"
RUNDIR="$STMP/RUNDIRS/$PSLOT"
DATA="$RUNDIR/$CDATE/$CDUMP/prepaodobs_NOAAVIIRS_$$"


[[ ! -d $DATA ]] && mkdir -p $DATA
cd $DATA || exit 10

FIX_SELF=${HOMEgfs}/fix_self/

ITILE=1
while [ ${ITILE} -le 6 ]; do
    ln -sf ${FIX_SELF}/grid_spec/${CASE}/${CASE}_grid_spec.tile${ITILE}.nc ${DATA}/grid_spec.tile${ITILE}.nc
    ITILE=$((ITILE+1))
done

FV3GRID=${DATA}

RES=`echo $CASE | cut -c2-4`
YY=`echo "${CDATE}" | cut -c1-4`
MM=`echo "${CDATE}" | cut -c5-6`
DD=`echo "${CDATE}" | cut -c7-8`
HH=`echo "${CDATE}" | cut -c9-10`

HALFCYCLE=$(( CYCINTHR/2 ))
STARTOBS=$(${NDATE} -${HALFCYCLE} ${CDATE})
ENDOBS=$(${NDATE} ${HALFCYCLE} ${CDATE})

STARTYY=`echo "${STARTOBS}" | cut -c1-4`
STARTMM=`echo "${STARTOBS}" | cut -c5-6`
STARTDD=`echo "${STARTOBS}" | cut -c7-8`
STARTHH=`echo "${STARTOBS}" | cut -c9-10`
STARTYMD=${STARTYY}${STARTMM}${STARTDD}
STARTYMDHMS=${STARTYY}${STARTMM}${STARTDD}${STARTHH}0000

ENDYY=`echo "${ENDOBS}" | cut -c1-4`
ENDMM=`echo "${ENDOBS}" | cut -c5-6`
ENDDD=`echo "${ENDOBS}" | cut -c7-8`
ENDHH=`echo "${ENDOBS}" | cut -c9-10`
ENDYMD=${ENDYY}${ENDMM}${ENDDD}
ENDYMDHMS=${ENDYY}${ENDMM}${ENDDD}${ENDHH}0000

for sat in ${AODSAT}; do
    FINALFILEv1_tmp="${AODTYPE}_AOD_${sat}.${CDATE}.iodav1.tmp.nc"
    FINALFILEv1="${AODTYPE}_AOD_${sat}.${CDATE}.iodav1.nc"
    FINALFILEv2="${AODTYPE}_AOD_${sat}.${CDATE}.iodav2.nc"
    FINALFILEv3="${AODTYPE}_AOD_${sat}.${CDATE}.iodav3.nc"
    [[ -f ${FINALFILEv1_tmp} ]] && /bin/rm -rf ${FINALFILEv1_tmp}
    [[ -f ${FINALFILEv1} ]] && /bin/rm -rf ${FINALFILEv1}
    [[ -f ${FINALFILEv2} ]] && /bin/rm -rf ${FINALFILEv2}
    [[ -f ${FINALFILEv3} ]] && /bin/rm -rf ${FINALFILEv3}
    declare -a usefiles # usefiles is now an array
    usefiles=() # clear the list of files
    allfiles=`ls -1 ${OBSDIR_NESDIS}/*_${sat}_s${STARTYMD}*_*.nc ${OBSDIR_NESDIS}/*_${sat}_*_e${ENDYMD}*_*.nc | sort -u`
    for f in ${allfiles}; do
        # Match the _s(number) start time and make sure it is after the time of interest
	if ! [[ $f =~ ^.*_s([0-9]{14}) ]] || ! (( BASH_REMATCH[1] >= STARTYMDHMS )) ; then
            echo "Skip; too early: $f"
        # Match the _e(number) end time and make sure it is after the time of interest
        elif ! [[ $f =~ ^.*_e([0-9]{14}) ]] || ! (( BASH_REMATCH[1] <= ENDYMDHMS )) ; then
            echo "Skip; too late:  $f"
        else
            echo "Using this file: $f"
            usefiles+=("$f") # Append the file to the usefiles array
        fi
    done
    echo "${usefiles[*]}" | tr ' ' '\n'
    
    # Make sure we found some files.
    echo "Found ${#usefiles[@]} files between $STARTOBS and $ENDOBS."
    if ! (( ${#usefiles[@]} > 0 )) ; then
        echo "Error: no files found for specified time range in ${OBSDIR_NESDIS}" 1>&2
    exit 1
    fi
    
    # Prepare the list of commands to run.
    [[ -f cmdfile ]] && /bin/rm -rf cmdfile
    cat /dev/null > cmdfile
    file_count=0
    for f in "${usefiles[@]}" ; do
        fout=$( basename "$f" )
        echo "${VIIRSIODA1_EXEC}" "${CDATE}" "$FV3GRID" "$f" "$fout" >> cmdfile
        file_count=$(( file_count + 1 ))
    done
    
    # Run many tasks in parallel via mpiserial.
    mpiserial_flags='-m '
    echo "Now running executable ${VIIRSIODA1_EXEC}"
    if ( ! srun  -l mpiserial $mpiserial_flags cmdfile ) ; then
        echo "At least one of the files failed. See prior logs for details." 1>&2
        exit 1
    fi
    
    # Make sure all files were created.
    no_output=0
    success=0
    for f in "${usefiles[@]}" ; do
        fout=$( basename "$f" )
        if [[ -s "$fout" ]] ; then
            success=$(( success + 1 ))
        else
            no_output=$(( no_output + 1 ))
            echo "Missing output file: $fout"
        fi
    done
    
    if [ "$success" -eq 0 ] ; then
        echo "Error: no files were output in this analysis cycle. Perhaps there are no obs at this time?" 1>&2
            exit 1
    fi
    if [ "$success" -ne "${#usefiles[@]}" ]; then
        echo "In analysis cycle ${CDATE}, only $success of ${#usefiles[@]} files were output."
        echo "Usually this means some files had no valid obs. See prior messages for details."
    else
        echo "In analysis cycle ${CDATE}, all $success of ${#usefiles[@]} files were output."
    fi
    
    # Merge the files.
    echo Merging files now...
    if ( ! ncrcat -O JRR-AOD_v3r2_${sat}_*.nc "${FINALFILEv1_tmp}" ) ; then
        echo "Error: ncrcat returned non-zero exit status" 1>&2
        exit 1
    fi
    
    # Make sure they really were merged.
    if [ ! -s "$FINALFILEv1_tmp" ]; then
        echo "Error: ncrcat did not create $FINALFILEv1_tmp ." 1>&2
        exit 1
    fi
     
    ncks --fix_rec_dmn all ${FINALFILEv1_tmp} ${FINALFILEv1}

    echo "IODA V1 to V2 for ${FINALFILEv1}"
    ${IODA1_IODA2_EXEC} ${FINALFILEv1} ${FINALFILEv2}
    err=$?
    if [ ${err} -ne 0 ]; then
        echo "IODA V1 to V2 failed for ${sat} at ${CDATE} and exit."
	exit 100
    fi

    echo "IODA V2 to V3 for ${FINALFILEv2}"
    ${IODA2_IODA3_EXEC} "${FINALFILEv2}" "${FINALFILEv3}" "${IODA2_IODA3_OBSYAML}"
    err=$?
    if [ $err -eq 0 ]; then
        /bin/mv ${FINALFILEv1}  ${AODOUTDIR}/
        /bin/mv ${FINALFILEv2}  ${AODOUTDIR}/
        /bin/mv ${FINALFILEv3}  ${AODOUTDIR}/
	echo ${AODSAT}
	echo ${sat}
	echo ${FINALFILEv2}
	if [ "${AODSAT}" == "npp" ] && [ ${sat} == "npp" ]; then
	    echo "equal to npp"
	    echo ${CDATE} >> ${MISS_NOAA_J01}
	    /bin/cp ${AODOUTDIR}/${AODTYPE}_AOD_npp.${CDATE}.iodav3.nc  ${AODOUTDIR}/${AODTYPE}_AOD_j01.${CDATE}.iodav3.nc
	fi

	if [ "${AODSAT}" == "j01" ] && [ ${sat} == "j01" ]; then
	    echo "equal to j01"
	    echo ${CDATE} >> ${MISS_NOAA_NPP}
	    /bin/cp ${AODOUTDIR}/${AODTYPE}_AOD_j01.${CDATE}.iodav3.nc  ${AODOUTDIR}/${AODTYPE}_AOD_npp.${CDATE}.iodav3.nc
	fi
        /bin/rm -rf JRR-AOD_*_${sat}_*.nc
        err=$?
    else
        echo "IODA_UPGRADER failed for ${FINALFILEv1} and exit."
	exit 1
    fi

done
    
if [[ $err -eq 0 ]]; then
    /bin/rm -rf $DATA
fi
echo ${CDATE} > ${TASKRC}
echo $(date) EXITING $0 with return code $err >&2
exit $err
