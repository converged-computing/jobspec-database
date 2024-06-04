#!/bin/bash --login
#PBS -N 174.4685.4730
# Select 5 nodes (each node has 24 cores. If we write select=2 here, we should write -n 48 in the aprun line below. maximum of 3072 cores)
#PBS -l select=192
#PBS -l walltime=24:00:00
#PBS -A e89-ic_m

module add vasp4
module load lammps

# Move to directory that script was submitted from
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)
cd $PBS_O_WORKDIR

#stable volume directories
list1="4.685a 4.730a"

#Job dir
dir="/work/e89/e89/tamellan/POINT_defect/FRENKEL_structures/E-V-curve/174_FREN_u3"

#Define a global counter
d=0
for i in $list1; do
#Select volume 
  cd "qha_"$i
#Define an in-folder counter
  c=0
  for j in `echo disp-*`; do
    echo $c $d
#Advance counters
    let c=c+1
    let d=d+1
#Select displacement
    cd $j
#In dir, copt start-stuff in
    cp $dir/qhaSUBstuff/* .  

#Call VASP
    aprun -n 48 /work/e89/e89/tamellan/vasp5_TI_lammps/vasp_5.3.5/vasp 2>&1 &

#Debug messages  
    echo "pwd is: " `pwd` 
    echo "the date is: " `date`
    echo "counnter is: " $c
    echo "directory is: " $j " in " $i
    echo "command is: aprun -n 48 /work/e89/e89/tamellan/vasp5_TI_lammps/vasp_5.3.5/vasp &"

#Short wait between node-job instances to get everything in order
    sleep 10

#Allow 96 instances of code, each running 48 cores
    if (( $c % 96 == 0 )); then wait; fi 
     
    cd ../
  done

  cd ../
  echo "Job Done: " $i $j
  echo "Job Done: " `pwd` 
  echo "Job Done: " `date` 

done
wait
echo "Total Job Done at " `date`

#  sed -i 's/EDIFF  =   1E-7/EDIFF  =   1E-10/g' INCAR  
