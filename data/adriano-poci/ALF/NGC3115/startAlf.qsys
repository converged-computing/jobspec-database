#!/bin/bash -l
#SBATCH -A oz059
#SBATCH --job-name="alf_NGC3115_SN100_aperture"
#SBATCH -D "/fred/oz059/poci/alf/NGC3115"
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=TIME_LIMIT_90,TIME_LIMIT,FAIL
#SBATCH --mail-user=adriano.poci@durham.ac.uk
#SBATCH -o "/fred/oz059/poci/alf/NGC3115/out.log" # Standard out to galaxy
#SBATCH -e "/fred/oz059/poci/alf/NGC3115/out.log" # Standard err to galaxy
#SBATCH --open-mode=append

source ${HOME}/.bashrc

module load gcc/9.2.0
module load openmpi/4.0.2
module load python/3.10.4
module load anaconda3/2021.05
export ALF_HOME=/fred/oz059/poci/alf/

### Compile clean version of `alf`
cd ${ALF_HOME}src
cp alf.perm.f90 alf.f90
# Remove prior placeholders on velz
sed -i "/prlo%velz = -999./d" alf.f90
sed -i "/prhi%velz = 999./d" alf.f90
make all && make clean
cd ${ALF_HOME}
# Run aperture fit
mpirun --oversubscribe -np ${SLURM_CPUS_PER_TASK} ./bin/alf.exe "NGC3115_SN100_aperture" 2>&1 | tee -a "NGC3115/out_aperture.log"

# Read in the aperture fit
Ipy='ipython --pylab --pprint --autoindent'
galax='NGC3115'
SN=100
pythonOutput=$($Ipy alf_aperRead.py -- -g "$galax" -sn "$SN")
echo "$pythonOutput" 2>&1 | tee -a "NGC3115/out_aperture.log"
# Temporary variable for the last line of the Python output
readarray -t tmp <<< $(echo "$pythonOutput" | tail -n1)
# Transform into bash array
IFS=',' read -ra aperKin <<< "$tmp"
echo "${aperKin[*]}" 2>&1 | tee -a "NGC3115/out_aperture.log"

### Compile modified velocity priors
cd src
cp alf.perm.f90 alf.f90
# `bc` arithmetic to define the lower and upper velocity bounds
newVLo=$(bc -l <<< "(${aperKin[0]} - ${aperKin[1]}) - 5.0 * (${aperKin[2]} + ${aperKin[3]})")
newVHi=$(bc -l <<< "(${aperKin[0]} + ${aperKin[1]}) + 5.0 * (${aperKin[2]} + ${aperKin[3]})")
sed -i "s/prlo%velz = -999./prlo%velz = ${newVLo}/g" alf.f90
sed -i "s/prhi%velz = 999./prhi%velz = ${newVHi}/g" alf.f90
# Replace the placeholder value in `sed` script
sed -i "s/velz = 999/velz = ${aperKin[0]}/g" ${ALF_HOME}NGC3115/alf_replace.sed
# Run `sed` using the multi-line script
# Pipe to temporary file
sed -n -f ${ALF_HOME}NGC3115/alf_replace.sed alf.f90 >> alf_tmp.f90
mv alf_tmp.f90 alf.f90

make all && make clean

# Move executables to local directory
cd ${ALF_HOME}
mkdir NGC3115/bin
cp bin/* NGC3115/bin/
find "NGC3115" -name "alf*.qsys" -type f -exec sbatch {} \;
