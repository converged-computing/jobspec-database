#!/bin/sh
#PBS -l walltime=71:58:02
#PBS -l select=1:ncpus=32:mem=64GB:gpfs=false
# gpfs=false recommend by RCS 2022-04-24 to bias scheduler to running on Intel CX1
# Otherwise you sit in the medium queue, and run on the AMD EPYC (which is much
# lower performance for Gaussian).
# Subject to change as they 'optimise' the queues

module load "gaussian/g16-c01-avx2" 

cp /rds/general/user/rk820/home/dibs-g16//1_C5H5CN_B3LYP.gjf ./
cp /rds/general/user/rk820/home/dibs-g16//1_C5H5CN_B3LYP.chk ./

#convert old checkpoints to latest (i.e. for g03 checkpoints generated before ~Dec 2009)
#c8609 "1_C5H5CN_B3LYP.chk"   

# no more pbsexec... running directly. 
# g03 is sym-linked to whatever version you've loaded :^)
g03 1_C5H5CN_B3LYP.gjf

cp *.log  /rds/general/user/rk820/home/dibs-g16// 
cp *.chk  /rds/general/user/rk820/home/dibs-g16//

echo "For us, there is only the trying. The rest is not our business. ~T.S.Eliot"
