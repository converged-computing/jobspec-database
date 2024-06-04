#!/bin/bash
# This is a sample PBS script 
# temps CPU a ajuster au calcul
   #PBS -l cput=800:00:00
   #PBS -l nodes=1:ppn=4
# memoire a ajuster au calcul
   #PBS -l vmem=30gb
# a changer
# Pour savoir sur quel noeud on est
#echo $HOSTNAME
# Startdir = ou sont les fichiers d'input, par defaut HOMEDIR
#
StartDir=$PBS_O_WORKDIR
echo $StartDir
#
# SCRATCHDIR = espace temporaire (local au noeud et a vider apres le calcul)
# NE PAS MODIFIER
ulimit -s unlimited
export SCRATCHDIR=/scratch/$USER/$PBS_JOBID
#
cd $StartDir


############################################################################
module load Core/Intel/2015 
source /home/pradines/program/qp2TEST/qp2/quantum_package.rc 
export OMP_NUM_THREADS=4
############################################################################

atoms_input=("H" "He" "Li" "Be" "B" "C" "N" "O" "F" "Ne" )
spin_multicip=( "2" "1" "2" "1" "2" "3" "4" "3" "2" "1")

basis_tab=( "cc-pvdz" "cc-pvtz" "cc-pvqz")

for dist in 0 1 2
do
 basis=${basis_tab[$dist]}
 mkdir $basis
 cd $basis
 ######boucle sur les atomes
 for atom in 1 9 #3 4 5 6 7 8 9
 do
  input_xyz=${atoms_input[$atom]}.xyz
  echo -e "1\nXYZ file: coordinates in Angstrom\n${atoms_input[$atom]}          0.0000000000          0.0000000000          0.0000000000"  > ${atoms_input[$atom]}.xyz  
  ############## Boucle Ions
  for charge in 0 #1 -1 
  do
   if [ "$charge" -eq "-1" ]; then
    ezfio=${atoms_input[$atom]}-
    qp create_ezfio -b "${basis}" $input_xyz -o $ezfio -x -c -1 -m ${spin_multicip[$atom+1]}
   fi
   
   if [ "$charge" -eq "0" ]; then
    ezfio=${atoms_input[$atom]} 
    qp create_ezfio -b "${basis}" $input_xyz -o $ezfio -x -c 0 -m ${spin_multicip[$atom]}
   fi
   
   if [ "$charge" -eq "1" ]; then
    ezfio=${atoms_input[$atom]}+
    qp create_ezfio -b "${basis}" $input_xyz -o $ezfio -x -c +1 -m ${spin_multicip[$atom-1]}
   fi

   qp set_file $ezfio
   qp set_frozen_core 

   # First SCF calculation
   qp run scf | tee ${ezfio}.SCF.out
   qp run faire_plaisir_a_PF | tee ${ezfio}.truc_PF.out  

   chiffre_PF=`grep "< HOMO | dE_c/dn | HOMO >    =" ${ezfio}.truc_PF.out | cut -d "=" -f 2 | tail -1`

   cd ..
   echo $basis $chiffre_PF >> chiffre_PF_${ezfio} 
   cd $basis

  done
 done
cd ..
done

exit
