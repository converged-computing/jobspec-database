#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
##SBATCH -w compute-0-4
##SBATCH --time=0-12:00:00
#SBATCH --partition=yethiraj
#SBATCH --job-name=nucl_hrex
#SBATCH --output=JOB.%J.out
#SBATCH --error=JOB.%J.err

export spack_root=/home/ajay/software/spack
. $spack_root/share/spack/setup-env.sh

export GMX_MAXBACKUP=-1
export PLUMED_MAXBACKUP=-1

nsteps=20000
replex=1000
n_mst=100
nrep=12

set -e
export OMP_NUM_THREADS=2
   spack load gromacs /ann

   echo "$(cat << EOF
        Ion: GROUP NDX_FILE=index.ndx NDX_GROUP=Ion
        WHOLEMOLECULES ENTITY0=Ion
        mat: CONTACT_MATRIX ATOMS=Ion SWITCH={RATIONAL R_0=0.35 NN=10000} NOPBC
        dfs: DFSCLUSTERING MATRIX=mat
        nat: CLUSTER_NATOMS CLUSTERS=dfs CLUSTER=1
        PRINT ARG=nat FILE=NAT
        DUMPGRAPH MATRIX=mat FILE=contact.dot
EOF
)" | plumed driver --igro min2.gro --plumed /dev/stdin &> /dev/null;

   if (( $(python contact.py contact.dot) ))
   	   then
           # Create plumed.dat to implement MST
           echo "Ion: GROUP NDX_FILE=index.ndx NDX_GROUP=Ion" > plumed.dat
           echo -e "WHOLEMOLECULES ENTITY0=Ion\n" >> plumed.dat

           n=1
           awk '{print $1+1","$2+1}' mst > edges
           while read p
           do
           echo "#D($p)" >> plumed.dat
           echo "DISTANCE ATOMS=$p LABEL=d$n NOPBC" >> plumed.dat
           echo "UPPER_WALLS ARG=d$n AT=0.35 KAPPA=1000.0 EXP=2 EPS=1 OFFSET=0 LABEL=uwall$n" >> plumed.dat
           echo " " >> plumed.dat
           n=$((n+1))
           done < edges
   fi
   mpirun -n 1 gmx_mpi grompp -c solv.gro -o npt.tpr -f mdp/npt.mdp -p system.top
   mpirun -n 1 gmx_mpi mdrun -deffnm npt -plumed plumed.dat

export OMP_NUM_THREADS=1

        tmin=300
        tmax=600

## build geometric progression of temperatures
        list=$(
        awk -v n=$nrep \
        -v tmin=$tmin \
        -v tmax=$tmax \
        'BEGIN{for(i=0;i<n;i++){
        t=tmin*exp(i*log(tmax/tmin)/(n-1));
        printf(t); if(i<n-1)printf(",");
        }
        }'
        )
        echo "intermediate temperatures are $list"

## clean directory (pre-existing folders)
        rm -fr topol*
#Create the replica folders
for((i=0;i<nrep;i++))
do
   mkdir -p topol$i
   cp index.ndx topol$i/
   lambda=$(echo $list | awk 'BEGIN{FS=",";}{print $1/$'$((i+1))';}')

   #process topology (create the "hamiltonian-scaled" forcefields)
   mpirun -n 1 plumed partial_tempering $lambda < processed.top > topol"$i"/topol.top

   #mpirun -n 1 gmx_mpi grompp -c em.gro -o topol"$i"/em.tpr -f em.mdp -p topol"$i"/topol.top

   # prepare tpr files
   mpirun -n 1 gmx_mpi grompp -c npt.gro -o topol"$i"/topol.tpr -f mdp/md.mdp -p topol"$i"/topol.top
done


a=$(eval echo topol{0..$((nrep-1))})
mpirun --use-hwthread-cpus -n $nrep gmx_mpi mdrun -deffnm topol -v -plumed ../plumed.dat -multidir $a -maxh 0.1 -pin on -dlb no -nsteps 1000 -gpu_id 0 -replex 100 -hrex



for i in `seq 1 $n_mst`
do
	echo "$(cat << EOF 
	Ion: GROUP NDX_FILE=index.ndx NDX_GROUP=Ion
	WHOLEMOLECULES ENTITY0=Ion
	mat: CONTACT_MATRIX ATOMS=Ion SWITCH={RATIONAL R_0=0.35 NN=10000} NOPBC
	dfs: DFSCLUSTERING MATRIX=mat
	nat: CLUSTER_NATOMS CLUSTERS=dfs CLUSTER=1
	PRINT ARG=nat FILE=NAT
	DUMPGRAPH MATRIX=mat FILE=contact.dot
EOF
)" | plumed driver --igro topol11/topol.gro --plumed /dev/stdin &> /dev/null; 
	
	if (( $(python contact.py contact.dot) ))
	then 
		# Create plumed.dat to implement MST
		echo "Ion: GROUP NDX_FILE=../index.ndx NDX_GROUP=Ion" > plumed.dat
		echo -e "WHOLEMOLECULES ENTITY0=Ion\n" >> plumed.dat
		
		n=1
		awk '{print $1+1","$2+1}' mst > edges
		while read p
		do
		echo "#D($p)" >> plumed.dat
		echo "DISTANCE ATOMS=$p LABEL=d$n NOPBC" >> plumed.dat
		echo "UPPER_WALLS ARG=d$n AT=0.35 KAPPA=100.0 EXP=2 EPS=1 OFFSET=0 LABEL=uwall$n" >> plumed.dat
		echo " " >> plumed.dat
		n=$((n+1))
		done < edges
	fi
        mpirun --use-hwthread-cpus -n $nrep gmx_mpi mdrun -deffnm topol -cpi topol.cpt -v -plumed ../plumed.dat -multidir $a -pin on -dlb no -nsteps $nsteps -gpu_id 0 -replex $replex -hrex
	rm -rf bck.* graph.* \#* 
#	cp plumed.dat plumed"$i".dat
#	cp mst mst"$i"
done
