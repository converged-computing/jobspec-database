mkdir somedirtorun3ddna

git clone https://github.com/aidenlab/3d-dna.git
cd 3d-dna

#create python env
module load StdEnv/2020 python/3.11.2

virtualenv 3ddna

source 3ddna/bin/activate

  pip install scipy numpy matplotlib #libraries required for 3d-dna 

deactivate

ln -s /pathtoyourfasta/yourfasta.fasta
ln -s /pathtoyourmergednodups/merged_nodups.txt

#!/bin/bash
#SBATCH --account=def-rieseber
#SBATCH --time=7-0
#SBATCH --cpus-per-task=15
#SBATCH --mem=100G
module load StdEnv/2020 python/3.10.2 java/17.0.2 lastz/1.04.03

export PATH="/somedirwhereis3ddna/3d-dna:$PATH"

source /home/egonza02/scratch/SOFTWARE/3D_DNA/3d-dna/3DDNA/bin/activate

run-asm-pipeline.sh -r 0 Harg2202r1.0-20210824.genome.new_names.fasta merged_nodups.txt

deactivate
