#!/bin/bash
if [ $# -ne 3 ]; then
   echo Usage: $0 platform Queue nosofs_ver
   echo For Example: $0 ptmp dev v3.5.0 for saving outputs on ptmp in queue dev
   exit 1
fi
##  platform = H1
##  que= dev or devhigh or devmax
set -x
platform=$1
que=$2
nosofs_ver=$3
echo $platform $que
export $platform
#nosofs_ver=v3.5.0

mkdir /lfs/h1/nos/$platform/$LOGNAME/rpt/$nosofs_ver

ftvar=aws  ##use either ftp or aws for transferring model output 
#if  [[ $platform == "Hl"* ]]; then
   VER=/lfs/h1/nos/nosofs/noscrub/$LOGNAME/packages/nosofs.${nosofs_ver}/versions/run.ver
   . $VER
   HOMEnos=/lfs/h1/nos/nosofs/noscrub/$LOGNAME/packages/nosofs.${nosofs_ver}
   version_file="\/lfs\/h1\/nos\/nosofs\/noscrub\/\$LOGNAME\/packages\/nosofs\.${nosofs_ver}\/versions\/run\.ver"

   phase=d

   job_script_prep='\/lfs\/h1\/nos\/nosofs\/noscrub\/$LOGNAME\/packages\/nosofs\.\${nosofs_ver}\/jobs\/JNOS_OFS_PREP'
   job_script_nf='\/lfs\/h1\/nos\/nosofs\/noscrub\/$LOGNAME\/packages\/nosofs\.\${nosofs_ver}\/jobs\/JNOS_OFS_NOWCST_FCST'
   job_script_ftp_get='\/lfs\/h1\/nos\/nosofs\/noscrub\/$LOGNAME\/packages\/nosofs\.\${nosofs_ver}\/jobs\/JNOS_OFS_FTP_GET'
   job_script_obs='\/lfs\/h1\/nos\/nosofs\/noscrub\/\$LOGNAME\/packages\/nosofs\.\${nosofs_ver}\/jobs\/JNOS_OFS_OBS'


   if [ $ftvar = 'aws' ]
   then
     job_script_ft='\/lfs\/h1\/nos\/nosofs\/noscrub\/\$LOGNAME\/packages\/nosofs\.\${nosofs_ver}\/jobs\/JNOS_OFS_AWS'
   else
     job_script_ft='\/lfs\/h1\/nos\/nosofs\/noscrub\/\$LOGNAME\/packages\/nosofs\.\${nosofs_ver}\/jobs\/JNOS_OFS_FTP'     
   fi
   ptile=128
   queue=$que
#else
#   echo platform needs to be either "ptmp" "stmp" 
#   exit
#fi
echo nosofs version is set to $nosofs_ver
echo HOMEnos is set to $HOMEnos

## WRITE jnos_${model}_prep_${cyc}.pbs
for model in cbofs dbofs tbofs ciofs gomofs leofs lmhofs loofs lsofs ngofs2 sfbofs creofs
do
  CYC1=00;CYC2=06; CYC3=12; CYC4=18
  if [ $model == "sfbofs" -o $model == "creofs"  -o $model == "ngofs2" -o -z "${model##wcofs*}" ]; then
     CYC1=03;CYC2=09; CYC3=15; CYC4=21
  fi
  total_tasks=$ptile
  for cyc in $CYC1 $CYC2 $CYC3 $CYC4
  do
   sed -e "s/MODEL/$model/g" \
       -e "s/QUEUE/$queue/g" \
       -e "s/TOTAL_TASKS/$total_tasks/g" \
       -e "s/CYC/$cyc/g" \
       -e "s/PLATFORM/$platform/g"  \
       -e "s/LOGNAME1/$LOGNAME/g"  \
       -e "s/NOSOFSVER/$nosofs_ver/g" \
       -e "s/PHASE/$phase/g" \
       -e "s/VERSION_FILE/$version_file/g" \
       -e "s/JOB_SCRIPT_PREP/$job_script_prep/g" \
       -e "s/PTILE/$ptile/g" nos_prep.ecf.dev > ./new/jnos_${model}_prep_${cyc}.pbs
  done 
done
for model in wcofs wcofs_free wcofs_da
do
  workdir='\/lfs\/h1\/nos\/ptmp\/\$LOGNAME\/\${model}\/work'
  CYC1=00;CYC2=06; CYC3=12; CYC4=18
  if [ $model == "sfbofs" -o $model == "creofs"  -o $model == "ngofs2" -o -z "${model##wcofs*}" ]; then
     CYC1=03;CYC2=09; CYC3=15; CYC4=21
  fi
  total_tasks=$ptile
#  total_tasks=`expr $ptile \* 2 `
  for cyc in $CYC1 
  do
   sed -e "s/MODEL/$model/g" \
       -e "s/QUEUE/$queue/g" \
       -e "s/TOTAL_TASKS/$total_tasks/g" \
       -e "s/CYC/$cyc/g" \
       -e "s/PLATFORM/$platform/g" \
       -e "s/LOGNAME1/$LOGNAME/g" \
       -e "s/NOSOFSVER/$nosofs_ver/g" \
       -e "s/PHASE/$phase/g" \
       -e "s/VERSION_FILE/$version_file/g" \
       -e "s/JOB_SCRIPT_PREP/$job_script_prep/g" \
       -e "s/PTILE/$ptile/g" nos_prep.ecf.dev > ./new/jnos_${model}_prep_${cyc}.pbs
  done 
done
## WRITE jnos_${model}_nowcst_fcst_${cyc}.pbs
for model in cbofs dbofs tbofs ciofs gomofs leofs lmhofs loofs lsofs ngofs2 sfbofs creofs
do
   run_time=02:30:00
   ptile=128
   total_nodes=1
   total_tasks=128
#   total_tasks=`expr $ptile \* 2 `
   CYC1=00;CYC2=06; CYC3=12; CYC4=18
   if [ $model == "sfbofs" -o $model == "creofs"  -o $model == "ngofs2" -o -z "${model##wcofs*}" ]; then
     CYC1=03;CYC2=09; CYC3=15; CYC4=21
   fi

   if [ $model == "ngofs2" ]; then
#       total_tasks=128
#       total_tasks=`expr $ptile \* 2 `
       total_nodes=6
       ptile=126
   elif [ $model == "creofs" ]; then
    #       total_tasks=256
#       total_tasks=`expr $ptile \* 2 `
       total_nodes=1
       ptile=128

   elif [ $model == "sfbofs" ]; then
#       total_tasks=256
#      total_tasks=`expr $ptile \* 2 `       
       total_nodes=2
       ptile=70
   elif [ $model == "cbofs" ]; then
#       total_tasks=256
#       total_tasks=`expr $ptile \* 2 `
       total_nodes=1
       ptile=128
   elif [ $model == "tbofs" ]; then
#       total_tasks=256
#       total_tasks=`expr $ptile \* 2 `
       total_nodes=1
       ptile=98
   elif [ $model == "dbofs" ]; then
#       total_tasks=256
#       total_tasks=`expr $ptile \* 2 `
       total_nodes=1
       ptile=128
   elif [ $model == "ciofs" ]; then
#        total_tasks=256
#        total_tasks=`expr $ptile \* 2 `
	total_nodes=7
	ptile=117
   elif [ $model == "lmhofs" ]; then
#        total_tasks=256
#	total_tasks=`expr $ptile \* 2 `
	total_nodes=5
	ptile=126
   elif [ $model == "leofs" ]; then
#        total_tasks=256
#	total_tasks=`expr $ptile \* 2 `
	total_nodes=1
	ptile=48
   elif [ $model == "lsofs" ]; then
#        total_tasks=256
#	total_tasks=`expr $ptile \* 2 `
	total_nodes=7
	ptile=100
   elif [ $model == "loofs" ]; then
#        total_tasks=256
#	total_tasks=`expr $ptile \* 2 `
	total_nodes=4
	ptile=128
   elif [ $model == "gomofs" ]; then
#        total_tasks=256
#	total_tasks=`expr $ptile \* 2 `
	total_nodes=9
	ptile=125
   fi


  for cyc in $CYC1 $CYC2 $CYC3 $CYC4
  do
   sed -e "s/MODEL/$model/g" \
       -e "s/QUEUE/$queue/g" \
       -e "s/TOTAL_TASKS/$total_tasks/g" \
       -e "s/TOTAL_NODES/$total_nodes/g" \
       -e "s/RUN_TIME/$run_time/g" \
       -e "s/CYC/$cyc/g" \
       -e "s/PLATFORM/$platform/g" \
       -e "s/LOGNAME1/$LOGNAME/g"  \
       -e "s/NOSOFSVER/$nosofs_ver/g" \
       -e "s/PHASE/$phase/g" \
       -e "s/VERSION_FILE/$version_file/g" \
       -e "s/JOB_SCRIPT_NF/$job_script_nf/g" \
       -e "s/PTILE/$ptile/g" nos_nowcst_fcst.ecf.dev > ./new/jnos_${model}_nowcst_fcst_${cyc}.pbs
#   if [ $model == 'wcofs_da' ]; then
#    sed -i "s/.*-w.*/#BSUB -w 'done (${model}_obs_${phase}_$cyc)'/" ./new/jnos_${model}_nowcst_fcst_${cyc}.pbs
#   fi
  done 
done
for model in wcofs wcofs_free
do
   run_time=02:30:00
   total_nodes=6
   total_tasks=128
   CYC1=03;CYC2=09; CYC3=15; CYC4=21
   total_nodes=4
   ptile=120

   for cyc in $CYC1 
   do
     sed -e "s/MODEL/$model/g" \
       -e "s/QUEUE/$queue/g" \
       -e "s/TOTAL_TASKS/$total_tasks/g" \
       -e "s/TOTAL_NODES/$total_nodes/g" \
       -e "s/RUN_TIME/$run_time/g" \
       -e "s/CYC/$cyc/g" \
       -e "s/PLATFORM/$platform/g" \
       -e "s/LOGNAME1/$LOGNAME/g"  \
       -e "s/NOSOFSVER/$nosofs_ver/g" \
       -e "s/PHASE/$phase/g" \
       -e "s/VERSION_FILE/$version_file/g" \
       -e "s/JOB_SCRIPT_NF/$job_script_nf/g" \
       -e "s/PTILE/$ptile/g" nos_nowcst_fcst.ecf.dev > ./new/jnos_${model}_nowcst_fcst_${cyc}.pbs
   done 

done

for model in wcofs_da
do
   run_time=02:30:00
   total_nodes=6
   total_tasks=128
   CYC1=03;CYC2=09; CYC3=15; CYC4=21
   total_nodes=4
   ptile=64
   run_time=12:00:00

   for cyc in $CYC1
   do
    sed -e "s/MODEL/$model/g" \
        -e "s/QUEUE/$queue/g" \
        -e "s/TOTAL_TASKS/$total_tasks/g" \
        -e "s/TOTAL_NODES/$total_nodes/g" \
        -e "s/RUN_TIME/$run_time/g" \
        -e "s/CYC/$cyc/g" \
        -e "s/PLATFORM/$platform/g" \
        -e "s/LOGNAME1/$LOGNAME/g"  \
        -e "s/NOSOFSVER/$nosofs_ver/g" \
        -e "s/PHASE/$phase/g" \
        -e "s/VERSION_FILE/$version_file/g" \
        -e "s/JOB_SCRIPT_NF/$job_script_nf/g" \
        -e "s/PTILE/$ptile/g" wcofs_da_nowcst_fcst.ecf.dev > ./new/jnos_${model}_nowcst_fcst_${cyc}.pbs
   done

done
												      


## WRITE jnos_${model}_ftp_${cyc}.pbs
if [ $ftvar = aws ]
then
  ftque=$queue
else
  ftque=dev_transfer
fi
for model in cbofs dbofs tbofs ciofs gomofs leofs lmhofs loofs lsofs ngofs2 sfbofs creofs
do
   CYC1=00;CYC2=06; CYC3=12; CYC4=18
   if [ $model == "sfbofs" -o $model == "creofs" -o $model == "ngofs2"  -o -z "${model##wcofs*}" ]; then
     CYC1=03;CYC2=09; CYC3=15; CYC4=21
   fi
  for cyc in $CYC1 $CYC2 $CYC3 $CYC4
  do
   sed -e "s/MODEL/$model/g" \
       -e "s/QUEUE/dev_transfer/g" \
       -e "s/CYC/$cyc/g" \
       -e "s/PLATFORM/$platform/g" \
       -e "s/LOGNAME1/$LOGNAME/g"  \
       -e "s/NOSOFSVER/$nosofs_ver/g" \
       -e "s/PHASE/$phase/g" \
       -e "s/VERSION_FILE/$version_file/g" \
       -e "s/JOB_SCRIPT_FILETRANSFER/$job_script_ft/g" \
       -e "s/PTILE/$ptile/g" nos_${ftvar}.ecf.dev > ./new/jnos_${model}_${ftvar}_${cyc}.pbs
  done 
  for cyc in $CYC1 $CYC2 $CYC3 $CYC4
  do
   sed -e "s/MODEL/$model/g" \
       -e "s/QUEUE/dev_transfer/g" \
       -e "s/CYC/$cyc/g" \
       -e "s/PLATFORM/$platform/g" \
       -e "s/LOGNAME1/$LOGNAME/g"  \
       -e "s/NOSOFSVER/$nosofs_ver/g" \
       -e "s/PHASE/$phase/g" \
       -e "s/VERSION_FILE/$version_file/g" \
       -e "s/JOB_SCRIPT_FILETRANSFER/$job_script_ft/g" \
       -e "s/PTILE/$ptile/g" nos_${ftvar}.ecf.prod > ./new/jnos_${model}_${ftvar}_${cyc}.prod
  done 
done

for model in wcofs wcofs_free wcofs_da
do
   CYC1=00;CYC2=06; CYC3=12; CYC4=18
   if [ $model == "sfbofs" -o $model == "creofs" -o $model == "ngofs2"  -o -z "${model##wcofs*}" ]; then
     CYC1=03;CYC2=09; CYC3=15; CYC4=21
   fi
  for cyc in $CYC1 
  do
   sed -e "s/MODEL/$model/g" \
       -e "s/QUEUE/$ftque/g" \
       -e "s/CYC/$cyc/g" \
       -e "s/PLATFORM/$platform/g" \
       -e "s/LOGNAME1/$LOGNAME/g"  \
       -e "s/NOSOFSVER/$nosofs_ver/g" \
       -e "s/PHASE/$phase/g" \
       -e "s/VERSION_FILE/$version_file/g" \
       -e "s/JOB_SCRIPT_FILETRANSFER/$job_script_ft/g" \
       -e "s/PTILE/$ptile/g" nos_${ftvar}.ecf.dev > ./new/jnos_${model}_${ftvar}_${cyc}.pbs
  done 
  for cyc in $CYC1 
  do
   sed -e "s/MODEL/$model/g" \
       -e "s/QUEUE/dev_transfer/g" \
       -e "s/CYC/$cyc/g" \
       -e "s/PLATFORM/$platform/g" \
       -e "s/LOGNAME1/$LOGNAME/g"  \
       -e "s/NOSOFSVER/$nosofs_ver/g" \
       -e "s/PHASE/$phase/g" \
       -e "s/VERSION_FILE/$version_file/g" \
       -e "s/JOB_SCRIPT_FILETRANSFER/$job_script_ft/g" \
       -e "s/PTILE/$ptile/g" nos_${ftvar}.ecf.prod > ./new/jnos_${model}_${ftvar}_${cyc}.prod
  done 
done


## WRITE subjobs_${model}_${cyc}.sh
for model in cbofs dbofs tbofs ciofs gomofs leofs lmhofs loofs lsofs ngofs2 sfbofs creofs
do
   CYC1=00;CYC2=06; CYC3=12; CYC4=18
   if [ $model == "sfbofs" -o $model == "creofs"  -o $model == "ngofs2" -o -z "${model##wcofs*}" ]; then
     CYC1=03;CYC2=09; CYC3=15; CYC4=21
   fi
  
  for cyc in $CYC1 $CYC2 $CYC3 $CYC4
  do
    echo "#!/bin/bash -l" > ./new/subjobs_${model}_${cyc}.sh
    echo -e ". $VER" >> ./new/subjobs_${model}_${cyc}.sh
    echo "module load envvar/\${envvars_ver:?}" >> ./new/subjobs_${model}_${cyc}.sh
    echo "module load PrgEnv-intel/\${PrgEnv_intel_ver}" >> ./new/subjobs_${model}_${cyc}.sh
    echo "module load craype/\${craype_ver}" >> ./new/subjobs_${model}_${cyc}.sh
    echo "module load intel/\${intel_ver}" >> ./new/subjobs_${model}_${cyc}.sh
    
    export RPTDIR=/lfs/h1/nos/$platform/$LOGNAME/rpt/$nosofs_ver
    echo "rm -f $RPTDIR/${model}_*_${cyc}.out"  >> ./new/subjobs_${model}_${cyc}.sh
    echo "rm -f $RPTDIR/${model}_*_${cyc}.err"  >> ./new/subjobs_${model}_${cyc}.sh
    echo "export LSFDIR=$HOMEnos/pbs " >> ./new/subjobs_${model}_${cyc}.sh
    echo "PREP=\$(qsub  \$LSFDIR/jnos_${model}_prep_${cyc}.pbs) " >> ./new/subjobs_${model}_${cyc}.sh
#    echo "qsub -W depend=afterok:\$PREP \$LSFDIR/jnos_${model}_nowcst_fcst_${cyc}.pbs" >> ./new/subjobs_${model}_${cyc}.sh

    echo "NFRUN=\$(qsub -W depend=afterok:\$PREP \$LSFDIR/jnos_${model}_nowcst_fcst_${cyc}.pbs)" >> ./new/subjobs_${model}_${cyc}.sh
    echo "qsub -W depend=afterok:\$NFRUN \$LSFDIR/jnos_${model}_${ftvar}_${cyc}.pbs" >> ./new/subjobs_${model}_${cyc}.sh
  done 
## create subjobs script for operational file transfer
  for cyc in $CYC1 $CYC2 $CYC3 $CYC4
  do
    echo "#!/bin/bash -l" > ./new/subjobs_${model}_aws_${cyc}.sh
    echo -e ". $VER" >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module purge " >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module load envvar/\${envvars_ver:?}" >>  ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module load PrgEnv-intel/\${PrgEnv_intel_ver}" >>  ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module load craype/\${craype_ver}" >>  ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module load intel/\${intel_ver}" >>  ./new/subjobs_${model}_aws_${cyc}.sh 
    export RPTDIR=/lfs/h1/nos/$platform/$LOGNAME/rpt/$nosofs_ver
    echo "rm -f $RPTDIR/${model}_aws_${cyc}_prod.out"  >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "rm -f $RPTDIR/${model}_aws_${cyc}_prod.err"  >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "export LSFDIR=$HOMEnos/pbs " >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "qsub \$LSFDIR/jnos_${model}_${ftvar}_${cyc}.prod" >> ./new/subjobs_${model}_aws_${cyc}.sh
  done 

done
for model in wcofs wcofs_free wcofs_da
do
   CYC1=00;CYC2=06; CYC3=12; CYC4=18
   if [ $model == "sfbofs" -o $model == "creofs"  -o $model == "ngofs2" -o -z "${model##wcofs*}" ]; then
     CYC1=03;CYC2=09; CYC3=15; CYC4=21
   fi
  
  for cyc in $CYC1 
  do
    echo "#!/bin/bash -l" > ./new/subjobs_${model}_${cyc}.sh
    echo -e ". $VER" >> ./new/subjobs_${model}_${cyc}.sh
    echo "module purge"  >>  ./new/subjobs_${model}_${cyc}.sh
    echo "module load envvar/\${envvars_ver:?}" >>  ./new/subjobs_${model}_${cyc}.sh
    echo "module load PrgEnv-intel/\${PrgEnv_intel_ver}" >> ./new/subjobs_${model}_${cyc}.sh
    echo "module load craype/\${craype_ver}" >>  ./new/subjobs_${model}_${cyc}.sh
    echo "module load intel/\${intel_ver}" >>      ./new/subjobs_${model}_${cyc}.sh
    echo "export LSFDIR=$HOMEnos/pbs " >> ./new/subjobs_${model}_${cyc}.sh
    export RPTDIR=/lfs/h1/nos/$platform/$LOGNAME/rpt/$nosofs_ver
    echo "rm -f $RPTDIR/${model}_*_${cyc}.out"  >> ./new/subjobs_${model}_${cyc}.sh
    echo "rm -f $RPTDIR/${model}_*_${cyc}.err"  >> ./new/subjobs_${model}_${cyc}.sh

    if [ $model == "wcofs_da" ]; then 
      echo "PREP=\$(qsub \$LSFDIR/jnos_${model}_prep_${cyc}.pbs) " >> ./new/subjobs_${model}_${cyc}.sh
      echo "OBS=\$(qsub -W depend=afterok:\$PREP \$LSFDIR/jnos_${model}_obs_${cyc}.pbs) "  >> ./new/subjobs_${model}_${cyc}.sh
#      echo "qsub -W depend=afterok:\$OBS \$LSFDIR/jnos_${model}_nowcst_fcst_${cyc}.pbs " >> ./new/subjobs_${model}_${cyc}.sh
      echo "NFRUN=\$(qsub -W depend=afterok:\$OBS \$LSFDIR/jnos_${model}_nowcst_fcst_${cyc}.pbs) " >> ./new/subjobs_${model}_${cyc}.sh
      echo "qsub -W depend=afterok:\$NFRUN \$LSFDIR/jnos_${model}_${ftvar}_${cyc}.pbs" >> ./new/subjobs_${model}_${cyc}.sh
    else
      echo "PREP=\$(qsub \$LSFDIR/jnos_${model}_prep_${cyc}.pbs) " >> ./new/subjobs_${model}_${cyc}.sh
#      echo "qsub -W depend=afterok:\$PREP \$LSFDIR/jnos_${model}_nowcst_fcst_${cyc}.pbs " >> ./new/subjobs_${model}_${cyc}.sh
       echo "NFRUN=\$(qsub -W depend=afterok:\$PREP \$LSFDIR/jnos_${model}_nowcst_fcst_${cyc}.pbs) " >> ./new/subjobs_${model}_${cyc}.sh
       echo "qsub -W depend=afterok:\$NFRUN \$LSFDIR/jnos_${model}_${ftvar}_${cyc}.pbs" >> ./new/subjobs_${model}_${cyc}.sh

   fi
  done 
## create subjobs script for operational file transfer
  for cyc in $CYC1 
  do
    echo "#!/bin/bash -l" > ./new/subjobs_${model}_aws_${cyc}.sh
    echo -e ". $VER" >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module purge"  >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module load envvar/\${envvars_ver:?}" >> ./new/subjobs_${model}_aws_${cyc}.sh 
    echo "module load PrgEnv-intel/\${PrgEnv_intel_ver}" >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module load craype/\${craype_ver}" >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module load intel/\${intel_ver}" >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "export LSFDIR=$HOMEnos/pbs " >> ./new/subjobs_${model}_aws_${cyc}.sh
    RPTDIR=/lfs/h1/nos/$platform/$LOGNAME/rpt/$nosofs_ver
    echo "rm -f $RPTDIR/${model}_aws_${cyc}*.out"  >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "rm -f $RPTDIR/${model}_aws_${cyc}*.err"  >> ./new/subjobs_${model}_aws_${cyc}.sh

    echo "qsub \$LSFDIR/jnos_${model}_${ftvar}_${cyc}.prod" >> ./new/subjobs_${model}_aws_${cyc}.sh
  done 

done
## WCOFS DA prepares for obs.nc
model='wcofs_da'
CYC1=03;CYC2=09; CYC3=15; CYC4=21
for cyc in $CYC1 
do
   sed -e "s/MODEL/$model/g" \
       -e "s/QUEUE/$queue/g" \
       -e "s/CYC/$cyc/g" \
       -e "s/PLATFORM/$platform/g" \
       -e "s/LOGNAME1/$LOGNAME/g"  \
       -e "s/NOSOFSVER/$nosofs_ver/g" \
       -e "s/PHASE/$phase/g" \
       -e "s/VERSION_FILE/$version_file/g" \
       -e "s/JOB_SCRIPT_OBS/$job_script_obs/g" \
           nos_obs.ecf.dev > ./new/jnos_${model}_obs_${cyc}.pbs
done
## GLOFS AWS for Legacy LSOFS and LOOFS
model=glofs
cyc='00'
job_script_ft='\/lfs\/h1\/nos\/nosofs\/noscrub\/\$LOGNAME\/packages\/nosofs\.\${nosofs_ver}\/jobs\/JGLOFS_AWS'

CYC1=00;CYC2=06; CYC3=12; CYC4=18
for cyc in $CYC1 $CYC2 $CYC3 $CYC4
do
   sed -e "s/MODEL/$model/g" \
       -e "s/QUEUE/dev_transfer/g" \
       -e "s/CYC/$cyc/g" \
       -e "s/PLATFORM/$platform/g" \
       -e "s/LOGNAME1/$LOGNAME/g"  \
       -e "s/NOSOFSVER/$nosofs_ver/g" \
       -e "s/PHASE/$phase/g" \
       -e "s/VERSION_FILE/$version_file/g" \
       -e "s/JOB_SCRIPT_FILETRANSFER/$job_script_ft/g" \
       -e "s/PTILE/$ptile/g" nos_${ftvar}.ecf.prod > ./new/jnos_${model}_${ftvar}_${cyc}.prod

    echo "#!/bin/bash -l" > ./new/subjobs_${model}_aws_${cyc}.sh
    echo -e ". $VER" >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module purge " >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module load envvar/\${envvars_ver:?}" >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module load PrgEnv-intel/\${PrgEnv_intel_ver}" >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module load craype/\${craype_ver}" >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "module load intel/\${intel_ver}" >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "export LSFDIR=$HOMEnos/pbs " >> ./new/subjobs_${model}_aws_${cyc}.sh
    RPTDIR=/lfs/h1/nos/$platform/$LOGNAME/rpt/$nosofs_ver
    echo "rm -f $RPTDIR/${model}_aws_${cyc}_prod.out"  >> ./new/subjobs_${model}_aws_${cyc}.sh
    echo "rm -f $RPTDIR/${model}_aws_${cyc}_prod.err"  >> ./new/subjobs_${model}_aws_${cyc}.sh    
    echo "qsub \$LSFDIR/jnos_${model}_${ftvar}_${cyc}.prod" >> ./new/subjobs_${model}_aws_${cyc}.sh
done 
chmod 755 ./new/*
