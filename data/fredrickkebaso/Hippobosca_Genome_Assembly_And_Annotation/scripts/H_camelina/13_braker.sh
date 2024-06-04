
#PBS -l select=1:ncpus=48:mpiprocs=48:mem=800gb
#PBS -q bigmem
#PBS -W group_list=bigmemq
#PBS -l walltime=48:00:00
#PBS -o /mnt/lustre/users/fkebaso/hippo/hcamelina_male/results_old/braker/spades/braker.out
#PBS -e /mnt/lustre/users/fkebaso/hippo/hcamelina_male/results_old/braker/spades/braker.err
#PBS -m abe
#PBS -M fredrickkebaso@gmail.com
#PBS -N braker_spades_hc_m

set -eu 

# ---------------- braker Gene Prediction ----------------

# Ab inition gene prediction

# ----------------Modules------------------

# Load Singularity module

module load chpc/singularity/3.5.3

# Create output variables
echo "Creating output variables..."
basedir="/mnt/lustre/users/fkebaso/hippo/hcamelina_male"
results="${basedir}/results_old/braker/spades"
genome="${basedir}/results_old/repeatmasker/spades/hcamelina_m_spades_genome.fa"
protein_file="${basedir}/results_old/braker/protein_db/Dme_glossina_proteins.fa"
threads=48
mincontig=100
email="fredrickkebaso@gmail.com"

# Set path to BRAKER Singularity container
export BRAKER_SIF="/apps/chpc/bio/BRAKER-3.0.3/braker3.sif"

# Check whether Singularity is installed
echo "Checking for Singularity..."
if ! command -v singularity &> /dev/null
then
    echo "Singularity could not be found."
    exit 1
fi

# Create new output directory
echo "Creating new output directory..."

mkdir -p "${results}"

# Create empty output files for braker.out and braker.err
echo "Creating empty output files..."
touch "${results}/braker.err" "${results}/braker.out"

# ---------------- Run -----------------------

# Run BRAKER using Singularity
echo "Running BRAKER..."
singularity exec -B ${PWD}:${PWD} ${BRAKER_SIF} braker.pl \
--genome="${genome}" \
--prot_seq="${protein_file}" \
--threads "${threads}" \
--workingdir="${results}" \
--min_contig="${mincontig}" \
--augustus_args="--species=fly" \
--softmasking \
--gff3 \
--makehub \
--email "${email}" 

echo "BRAKER completed successfully !!!"

# -B ${PWD}:${PWD}: Bind-mount the current working directory inside the container
# ${BRAKER_SIF}: Path to the Singularity container to be used
# braker.pl: Name of the Braker script to be executed inside the container
# --genome="${genome}": Path to the genome FASTA file
# --prot_seq=$protein_file: Path to the protein sequence file
# --threads 48: Number of threads to be used
# --workingdir="${results}": Directory to store output files
# --min_contig=100: Minimum length of contigs to be considered
# --augustus_args="--species=fly": Additional arguments to be passed to the Augustus gene predictor
# --makehub: Create a UCSC genome browser hub for visualization
# --gff3: Generate output in GFF3 format
# --email "${email}": Email address to receive job notifications
