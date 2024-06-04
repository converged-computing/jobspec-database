#!/bin/bash -l

#SBATCH -A rrg-wperciva
#SBATCH --nodes 2
#SBATCH --tasks-per-node=4
#SBATCH -t 72:00:00
#SBATCH -J cosmomc
#SBATCH --exclusive

cd $SLURM_SUBMIT_DIR

ir=0
####remember need to always go into this file and directly reset to 0 before new run: common_batch1_jbd.ini

echo $ir > countjbd.txt
echo $ir > testjbd.txt

#looping over the bootstrap realizations for fixed number of samples
for ((bl=0; bl<=999; bl++)); do

#module load null python intel-compilers/18 mpi/4.0.0-intel

cp data/lensingrsdfiles/bootstrapnz/nz_z1_kids_boot${bl}.dat data/lensingrsdfiles/nz_z1_kids_binned_bootstrap.dat
cp data/lensingrsdfiles/bootstrapnz/nz_z2_kids_boot${bl}.dat data/lensingrsdfiles/nz_z2_kids_binned_bootstrap.dat
cp data/lensingrsdfiles/bootstrapnz/nz_z3_kids_boot${bl}.dat data/lensingrsdfiles/nz_z3_kids_binned_bootstrap.dat
cp data/lensingrsdfiles/bootstrapnz/nz_z4_kids_boot${bl}.dat data/lensingrsdfiles/nz_z4_kids_binned_bootstrap.dat

ir=`echo "200*$bl"|bc`
fr=`echo "200*($bl+1)"|bc`

echo $bl >> countjbd.txt
echo $ir >> countjbd.txt
echo $fr >> countjbd.txt

#Need to always change this directory name  when creating a new directory
sed -i "/samples/ s/${ir}/${fr}/" /home/sjoudaki/projects/rrg-wperciva/sjoudaki/CosmoJBD/batch1/common_batch1_jbd.ini

echo $SLURM_JOB_ID >> testjbd.txt

mpirun --map-by node --bind-to none -np 8 -x OMP_NUM_THREADS=8 ./cosmomc testjbd.ini >> testjbd.txt

done
echo finished
