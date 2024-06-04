#!/bin/bash
#PBS -N rsync2dogwood
#PBS -o /u/shelley.melchior/rsync_dogwood.out
#PBS -e /u/shelley.melchior/rsync_dogwood.out
#PBS -l select=1:ncpus=1:mem=1000MB
#PBS -q dev_transfer
#PBS -A VERF-DEV
#PBS -l walltime=02:00:00

module load rsync/3.2.2

#dtgarr="20240313"
dtgarr="20240320 20240321 20240322 20240323"
for dtg in ${dtgarr[@]}
do
  echo $dtg
  echo "rsync -ahr -P /lfs/h2/emc/vpppg/noscrub/shannon.shields/EVS_Data/evs/v1.0/prep/subseasonal/atmos.$dtg ddxfer.wcoss2.ncep.noaa.gov:/lfs/h2/emc/vpppg/noscrub/shelley.melchior/forSS/."
  rsync -ahr -P /lfs/h2/emc/vpppg/noscrub/shannon.shields/EVS_Data/evs/v1.0/prep/subseasonal/atmos.$dtg ddxfer.wcoss2.ncep.noaa.gov:/lfs/h2/emc/vpppg/noscrub/shelley.melchior/forSS/.
done

#rsync -ahr -P /lfs/h2/emc/vpppg/noscrub/shannon.shields/EVS_Data/evs/v1.0/prep/subseasonal/atmos.20240312 ddxfer.wcoss2.ncep.noaa.gov:/lfs/h2/emc/vpppg/noscrub/shelley.melchior/forSS/.

