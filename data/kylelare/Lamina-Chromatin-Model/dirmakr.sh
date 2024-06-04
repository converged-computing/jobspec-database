#!/bin/bash -l
tx=1.1
path=/scratch/kl76/data/$tx
fname="init5.slurm"
partn="common"
cons='"opath"'
home=/scratch/kl76/data/
GMX='$GMX'
MDR='$MDR'
#mkdir /scratch/kl76/potential/wall/test/output_test/
#mkdir $path
mkdir $path
cd $path
for cx in 0.2; do
  mkdir $path/$cx
  for dSR in run1 run2 run3 run4 run5; do
    mkdir $path/$cx/$dSR
    cd
    cd $home
    #cp seq_C21.txt $path/$cx#/$dSR/$dLR
    #cp gamingedit.py $path/$cx#/$dSR/$dLR
    #cp gamingedit2.py $path/$cx/$dSR
    #cp wall.py $path/$cx/$dSR  
    #cp seqwall.txt $path/$cx/$dSR
    #cp walls-restraint.gro $path/$cx/$dSR
    #cp wall-init-snap.gro $path/$cx/$dSR
    #cp MDP-sim-no-anneal.mdp $path/$cx/$dSR
    cp HI-C.py $path/$cx/$dSR
    cd $path/$cx/$dSR
    name="C21-GM12878"
    content="#!/bin/bash -l
#SBATCH --job-name=chromatin
#SBATCH --partition= #partition name here
#SBATCH --account= #account name here
#SBATCH --ntasks=8
#SBATCH --constraint="opath" 
#SBATCH --export=ALL
#SBATCH --time=23:55:00
#SBATCH --mem-per-cpu=2GB

export OMP_NUM_THREADS=1

module load icc/2019.3.199-GCC-8.3.0  impi/2019.4.243 GROMACS/2019.5 Anaconda2

GMX=/opt/apps/software/MPI/intel/2019.3.199-GCC-8.3.0/impi/2019.4.243/GROMACS/2019.5/bin/gmx

MDR=/opt/apps/software/MPI/intel/2019.3.199-GCC-8.3.0/impi/2019.4.243/GROMACS/2019.5/bin/mdrun_mpi

name="C21-GM12878"

#srun -n 1 python gamingedit.py -seq seq_C21.txt  -tables yes -n $name -Csize 935

#srun -n 1 $GMX make_ndx -f $name-init.gro -o $name-index <<EOF
#0
#q
#EOF

#srun -n 1 $GMX grompp -f MDP-init.mdp -n $name-index.ndx -p $name-init.top -c $name-init.gro -r $name-restraint.gro -o $name-init.tpr -maxwarn -1

#srun $MDR -v -table table.xvg -tableb table_b0.xvg table_b1.xvg -s $name-init.tpr -c $name-init-snap.gro

#srun -n 1 python wall.py

#srun -n 1 python gamingedit2.py -seq seqwall.txt -tables yes -Csize 935 -B3lamina -$cx

#srun -n 1 $GMX make_ndx -f wall-init-snap.gro -o wall-index <<EOF
#0
#q
#EOF

#srun -n 1 $GMX grompp -f MDP-sim-no-anneal.mdp -n wall-index.ndx -p wall-sim.top -c wall-init-snap.gro -r walls-restraint.gro -o wall-sim.tpr -maxwarn -1

#srun $MDR -v -table table.xvg -tableb table_b0.xvg table_b1.xvg table_b2.xvg -s wall-sim.tpr -noddcheck -deffnm $name

#srun -n 1 $GMX trjconv -f $name.trr -s wall-sim.tpr -o all-$name-$cx-$dSR.gro -b 50 <<EOF
#0
#EOF

srun -n 1 python HI-C.py -f all-$name-$cx-$dSR.gro -n $name-$cx-$dSR

"
    echo "$content">$fname
    sbatch $fname
  done
done
exit 0
