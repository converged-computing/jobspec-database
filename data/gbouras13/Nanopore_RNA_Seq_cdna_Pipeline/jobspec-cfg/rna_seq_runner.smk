"""
The snakefile that runs the pipeline.

# local tests
snakemake -s rna_seq_runner.smk -c 16 --use-conda --config Reads=fastqs Output='/Users/a1667917/Documents/Ghais/Rat_RNA_Seq/test' DBDIR='/Users/a1667917/Documents/Ghais/Rat_RNA_Seq/Rat_Transcriptome'  --conda-frontend conda

# HPC
# on login node from pipeline dir
snakemake -s rna_seq_runner.smk -c 1 --use-conda --config Reads=fastqs Output=test --conda-create-envs-only --conda-frontend conda
# to run
snakemake -s rna_seq_runner.smk --use-conda --config Reads=Fastqs/ Output=test DBDIR='/hpcfs/users/a1667917/STAR_Ref_Genomes' --profile wgs_tcga
"""


### DEFAULT CONFIG FILE
configfile: os.path.join(workflow.basedir,  'config', 'config.yaml')

BigJobMem = config["BigJobMem"]
BigJobCpu = config["BigJobCpu"]
# STAR can only use 16 threads on hpc for some reason
# https://github.com/alexdobin/STAR/issues/1074
MediumJobCpu = config["MediumJobCpu"]

# need to specify the reads directory
READS = config['Reads']

### DIRECTORIES
include: "rules/directories.smk"

# Parse the samples and read files
include: "rules/samples.smk"
sampleReads = parseSamples(READS)
SAMPLES = sampleReads.keys()
# write tsv
writeSamplesTsv(sampleReads, os.path.join(OUTPUT, 'rna_seq_samples.tsv'))


# Import rules and functions
include: "rules/targets.smk"
include: "rules/qc.smk"
include: "rules/align.smk"
include: "rules/quantify.smk"

rule all:
    input:
        QCFiles,
        AlignFiles,
        BambuFiles
        