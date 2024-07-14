"""
The snakefile that runs the pipeline.
Manual launch example:

"""

import os

### DEFAULT CONFIG FILE

configfile: os.path.join(  'config', 'config.yaml')


### DIRECTORIES

include: "rules/directories.smk"

# get if needed
CSV = config['csv']
OUTPUT = config['Output']
GuppyDir = config['GuppyDir']
RefFastaDir = config['RefFastaDir']
NanoDiscoSingularityDir = config['NanoDiscoSingularityDir']
C308Fast5Dir = config['C308Fast5Dir']

BigJobMem = config["BigJobMem"]
BigJobCpu = config["BigJobCpu"]
MedJobCpu = config["MedJobCpu"]
SmallJobMem = config["SmallJobMem"]


# Parse the samples and read files
include: "rules/samples.smk"
dictReads = parseSamples(CSV)
SAMPLES = list(dictReads.keys())
#print(SAMPLES)


# Import rules and functions
include: "rules/targets.smk"
include: "rules/compress_fast5.smk"
include: "rules/index_reference.smk"
include: "rules/nanodisco_preprocess.smk"
# dont even need chunk
#include: "rules/nanodisco_chunk.smk"
include: "rules/nanodisco_difference.smk"
include: "rules/nanodisco_merge.smk"
include: "rules/nanodisco_motif.smk"


rule all:
    input:
        NanoDiscoFiles
