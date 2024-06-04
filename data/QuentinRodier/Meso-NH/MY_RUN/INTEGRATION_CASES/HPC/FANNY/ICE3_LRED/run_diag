#!/bin/sh
#SBATCH -J diag_fanny
#SBATCH -N 2            # nodes number (=NBP)
#SBATCH -n 256            # CPUs number (on all nodes) (=NBP*TPN)
#SBATCH -o diag_fanny.eo%j   #
#SBATCH -e diag_fanny.eo%j   #
#SBATCH -t 01:00:00    # time limit
# Echo des commandes
ulimit -c 0
ulimit -s unlimited
# Arrete du job des la premiere erreur
set -e
# Nom de la machine
hostname 


. ~rodierq/DEV_57/MNH-PHYEX070-b95d84d7/conf/profile_mesonh-LXifort-R8I4-MNH-V5-6-2-ECRAD140-MPIAUTO-O2

export MPIRUN="Mpirun -np 256"

set -x
set -e


ls -lrt

for ECH in '001' '002' '003' '004' '005' '006' '007' '008' '009' '010' '011' '012' '013' '014' '015' '016' '017' '018' '019' '020' '021' '022' '023' '024' '025'
do

cat > DIAG1.nam << EOF
&NAM_DIAG 
  CISO='PREVTK',
  LVAR_RS=T, 
  NCONV_KF=-1, 
  LVAR_MRW=T,
  LVAR_MRSV=F,
  LTRAJ=T,
  LTPZH=T,
  LMOIST_V=T, LMOIST_E=T,
  LMSLP=T,
  LCLD_COV=F, 
  LVORT=T,
  LVAR_PR=F, LTOTAL_PR=T, 
  NCAPE=1,
  LRADAR=T,
  LTHW=T,
/
&NAM_CONFIO LCDF4=T LLFIREAD=F LLFIOUT=F /
&NAM_DIAG_SURFn  N2M=1
/
&NAM_DIAG_FILE  YSUFFIX = "d" 
YINIFILE(1) = "FANNY.1.WENO5.${ECH}", 
YINIFILEPGD(1)='PGD_2.5km_AR'/
EOF
rm -f FANNY.1.WENO5.${ECH}d.???
time ${MPIRUN} DIAG${XYZ}
mv OUTPUT_LISTING0  OUTPUT_LISTING0_diag.${ECH}
mv OUTPUT_LISTING1  OUTPUT_LISTING1_diag.${ECH}
ls -lrt

done


rm -f file_for_xtransfer pipe_name


ls -lrt
rm -Rf OUTPUT
mkdir OUTPUT
mv OUTPUT_L* OUTPUT
mkdir LFI
mv *.lfi LFI/.
mv *.des LFI/.
ja
