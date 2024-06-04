#!/bin/bash
#SBATCH --partition=hpc          # partition (queue)
#SBATCH --nodes=1                # number of nodes
#SBATCH --mem=160G               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=72:00:00          # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR
#SBATCH --export=ALL
#SBATCH --exclusive

# Necessary to pseudo-revert to old memory allocation behaviour
export MALLOC_ARENA_MAX=4

module load java/default
module load cuda/default
module load matlab/R2019b
module load openmpi/gnu
source ~/OpenFOAM-plus/etc/bashrc

destFolderName="/scratch/jkrusk2s/sailCFD/"
baseFolderName="/home/jkrusk2s/Code/sail/domains/escooter/pe/v1906/"
nCases=10;
startCase=200;

for (( i=$startCase; i<$startCase+$nCases; i++ ))
do
	caseName=$destFolderName"case$i"
	echo $caseName
    rm -rf $caseName
	cp -TR $baseFolderName $caseName
	sbatch -D "$caseName" $caseName/submit.sh
done 

# Run experiment
matlab -batch "escooter_runSail('nCases',$nCases,'caseStart',$startCase)"
for (( i=$startCase; i<$startCase+$nCases; i++ ))
do
    caseName=$destFolderName"case$i"
	touch "$caseName/stop.signal"
done 