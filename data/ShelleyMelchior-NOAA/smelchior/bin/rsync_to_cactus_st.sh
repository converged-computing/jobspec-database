#!/bin/bash
#PBS -N rsync2cactus
#PBS -o /u/shelley.melchior/rsync_cactus.out
#PBS -e /u/shelley.melchior/rsync_cactus.out
#PBS -l select=1:ncpus=1:mem=1000MB
#PBS -q dev_transfer
#PBS -A VERF-DEV
#PBS -l walltime=03:30:00

module load rsync/3.2.2

dtgarr="20240325"
#dtgarr="20240309 20240310"
for dtg in ${dtgarr[@]}
do
  echo $dtg
  echo "rsync -ahr -P /lfs/h2/emc/vpppg/noscrub/steven.simon/evs/v1.0/prep/global_ens/atmos.$dtg cdxfer.wcoss2.ncep.noaa.gov:/lfs/h2/emc/vpppg/noscrub/shelley.melchior/forSSi/16dayprep/."
  rsync -ahr -P /lfs/h2/emc/vpppg/noscrub/steven.simon/evs/v1.0/prep/global_ens/atmos.$dtg cdxfer.wcoss2.ncep.noaa.gov:/lfs/h2/emc/vpppg/noscrub/shelley.melchior/forSSi/16dayprep/.
done

#rsync -ahr -P /lfs/h2/emc/vpppg/noscrub/steven.simon/metplusworkspace/EVS cdxfer.wcoss2.ncep.noaa.gov:/lfs/h2/emc/vpppg/noscrub/shelley.melchior/forSSi/EVS/.

