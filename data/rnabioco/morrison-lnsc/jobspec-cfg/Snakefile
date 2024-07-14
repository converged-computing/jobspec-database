# ===== Pipeline for running cellranger ========================================


# Configure shell for all rules 
shell.executable("/bin/bash")
shell.prefix("set -o nounset -o pipefail -o errexit -x; ")
import subprocess
import shutil
import glob
import os 
import re


# Parameters from config.yaml
RAW_DATA     = config["RAW_DATA"]
RESULTS      = config["RESULTS"]
RNA_SAMPLES  = config["RNA_SAMPLES"]
GENOME       = config["GENOME"]
ADT_SAMPLES  = config["ADT_SAMPLES"]
ADT_REF      = config["ADT_REF"]
ANTIBODIES   = config["ANTIBODIES"]
VDJ_SAMPLES  = config["VDJ_SAMPLES"]
VDJ_REF      = config["VDJ_REF"]
MAX_JOBS     = config["MAX_JOBS"]
LSF_TEMPLATE = config["LSF_TEMPLATE"]

CHEM = "auto"

if "CHEM" in config:
    if config["CHEM"]:
        CHEM = config["CHEM"]


# Function to check paths for input files/directories
def _check_path(path):
    if os.path.exists(path):
        return os.path.abspath(path)
    else:
        sys.exit("ERROR: " + path + " does not exist.")


# Set sample/group names
FASTQ_REGEX = "_S[0-9]+_L[0-9]+_R[12]_[0-9]+\.fastq\.gz"

SAMPLES = RNA_SAMPLES

if RNA_SAMPLES:
    RNA_SAMPLES = [x.strip() for x in RNA_SAMPLES]
    GENOME      = _check_path(GENOME)

if ADT_SAMPLES:
    ADT_SAMPLES = [re.sub(" ", "", x) for x in ADT_SAMPLES]
    SAMPLES     = [re.sub(",", "_", x) for x in ADT_SAMPLES]
    ADT_REF     = _check_path(ADT_REF)

    if RNA_SAMPLES:
        SAMPLES = [x + "-" + y for x, y in zip(RNA_SAMPLES, SAMPLES)]

if VDJ_SAMPLES:
    VDJ_SAMPLES = [x.strip() for x in VDJ_SAMPLES]
    VDJ_REF     = _check_path(VDJ_REF)


# Check directory/file paths
RAW_DATA = [_check_path(x) for x in RAW_DATA]
RESULTS  = _check_path(RESULTS)

if LSF_TEMPLATE:
    LSF_TEMPLATE = _check_path(LSF_TEMPLATE)
else:
    LSF_TEMPLATE = "lsf"

FASTQ_DIR = RESULTS + "/fastqs"

if not os.path.exists(FASTQ_DIR):
    os.makedirs(FASTQ_DIR)


# Final output files
rule all:
    input:
        RESULTS + "/logs/.merge_fastqs_done",

        RESULTS + "/logs/.antibody_csv_done",

        expand(
            RESULTS + "/logs/.{sample}_count_done", 
            sample = SAMPLES
        ),

        expand(
            RESULTS + "/logs/.{vdj_sample}_vdj_done",
            vdj_sample = VDJ_SAMPLES
        ),

        RESULTS + "/count_metrics.csv"

include: "src/rules/cellranger.snake"



