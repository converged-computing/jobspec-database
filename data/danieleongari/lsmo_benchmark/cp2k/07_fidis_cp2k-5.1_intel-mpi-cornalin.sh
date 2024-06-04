#!/bin/bash 

coresxcpu=28     #to adjust according to the machine
rm -rf "${0%.*}" #remove pre-existing dir
mkdir  "${0%.*}" #make a dir with the same name as the executable
cd "${0%.*}"

for ncores in 8 16 32 64 96 128 192 256
do

nnodes=$((ncores / coresxcpu))
remainder=$((ncores % coresxcpu))
if [ $remainder != 0 ]; then nnodes=$((nnodes+1)); fi #nnodes= number of nodes needed
 
mkdir ncores_$(printf %03d $ncores)                   #make dir like "ncores_008"
cd ncores_$(printf %03d $ncores)

cp ../../00_inputs/* .

cat > run.slurm << EOF
#!/bin/bash -l
#SBATCH --nodes=$nnodes
#SBATCH --ntasks=$ncores
#SBATCH --time=00:30:00
#SBATCH --partition=parallel
#SBATCH --constraint=E5v4


source /ssoft/spack/bin/slmodules.sh -r deprecated   
module load intel
module load intel-mpi intel-mkl
module load cp2k/5.1-mpi

date_start=\$(date +%s)
srun /ssoft/spack/cornalin/v2/opt/spack/linux-rhel7-x86_E5v4_Mellanox/intel-17.0.2/cp2k-5.1-gke4ujsx5qhx6zt7ncfo22hajqhp6n6b/bin/cp2k.popt -i *.inp -o output.out                 #running command
date_end=\$(date +%s)
time_run=\$((date_end-date_start))
echo "$(printf %03d $ncores)_cpus \$time_run seconds"

rm -f GTH_BASIS_SETS POTENTIAL *xyz *restart          #remove useless big files (please!) 
EOF

sbatch run.slurm

cd ..

done
