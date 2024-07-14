#CONFIG FILE ###########################################################################################################################################################

# Define config file in the snakemake command within the SLURM submission script

#VARIABLE SETUP ########################################################################################################################################################

#Import packages for file name management
import glob #use wildcards to take in file names

import os
#needed for pointing to location of dbs in rules kr2pair and kr2unpair

envvars:
  "SLURM_TMPDIR"
# Will be used in rule cp_db
# Load databases into tempdir for efficiency

#Detect wildcards from input files names
filename = config['pathtoinput'] + "Mock_{comm}_R1.{ext}.gz"
COMM,EXT = glob_wildcards(filename)

#Wildcards from config dictionaries
#Dynamically picks up list of databases and kr2 confidence levels from config
DB = list(config['dbs'].keys())
CONF=list(config['confidence'].keys())


#CATCH-ALL RULE ###################################################################################################################################################

rule all:
  #List all the final output files expected here.
  input:
    expand(config['analysis_round'] + "/{db}/SSRchecks/{comm}_{db}_c{conf}_blSSR_falsepos.txt", db=DB, comm=COMM, conf=CONF),
    expand(config['analysis_round'] + "/{db}/Salmonella/{comm}_{db}_c{conf}_salmreads.txt", db=DB, comm=COMM, conf=CONF)
  

#INITIAL STEPS ################################################################################################################################################

rule trim:
  input:
    fwd = config['pathtoinput'] + "Mock_{comm}_R1.fq.gz",
    rev = config['pathtoinput'] + "Mock_{comm}_R2.fq.gz"
  output:
    fwdp = temp(config['analysis_round'] + "/Mock_{comm}_R1.P.fq"),
    fwdu = temp(config['analysis_round'] + "/Mock_{comm}_R1.U.fq"),
    revp = temp(config['analysis_round'] + "/Mock_{comm}_R2.P.fq"),
    revu = temp(config['analysis_round'] + "/Mock_{comm}_R2.U.fq"),
    log = config['analysis_round'] + "/Trimlogs/{comm}_trimlog.txt"
  group: "trimrules"
  shell:
    "java -jar $EBROOTTRIMMOMATIC/trimmomatic-0.39.jar PE -summary {output.log} \
    {input.fwd} {input.rev} {output.fwdp} {output.fwdu} {output.revp} {output.revu} \
    SLIDINGWINDOW:4:20 MINLEN:36"
        
#Concatenate unpaired sequences together
rule cat_unpaired:
  input:
    unpaired_R1 = config['analysis_round'] + "/Mock_{comm}_R1.U.fq",
    unpaired_R2 = config['analysis_round'] + "/Mock_{comm}_R2.U.fq"
  group: "trimrules"
  output:
    temp(config['analysis_round'] + "/{comm}_unpaired.fq.gz")
  shell:
    "cat {input.unpaired_R1} {input.unpaired_R2} | gzip > {output}"


#DATABASE ANNOTATION######################################################################################################################################

#Copy database files to SLURM temporary directory

rule cp_db_kr2bac:
  input:
    config['dbs']['kr2bac']
  output: directory(os.path.join(os.environ["SLURM_TMPDIR"], "kr2bac"))
  shell:
    "cp -R {input} {output}"

rule cp_db_kr2std:
  input:
    config['dbs']['kr2std']
  output: directory(os.path.join(os.environ["SLURM_TMPDIR"], "kr2std"))
  shell:
    "cp -R {input} {output}"

rule cp_db_kr2plrename:
  input:
    config['dbs']['kr2plrename']
  output: directory(os.path.join(os.environ["SLURM_TMPDIR"], "kr2plrename"))
  shell:
    "cp -R {input} {output}"

#Classify against bacteria database
rule kr2pair_kr2bac:
  input:
    paired_R1 = config['analysis_round'] + "/Mock_{comm}_R1.P.fq",
    paired_R2 = config['analysis_round'] + "/Mock_{comm}_R2.P.fq",
    db = rules.cp_db_kr2bac.output
  output:
    text = temp(config['analysis_round'] + "/kr2bac/Kr2reports/{comm}_paired.kr2bac_c{conf}_kr.txt"),
    report = config['analysis_round'] + "/kr2bac/Kr2reports/{comm}_paired.kr2bac_c{conf}.rep"
  params: 
    confidence = lambda wildcards: config['confidence'][wildcards.conf]
  shell: "kraken2 -db {input.db} --paired {input.paired_R1} {input.paired_R2} \
  --use-names \
  --output {output.text} \
  --confidence {params.confidence} \
  --report {output.report} --report-zero-counts "

rule kr2unpair_kr2bac:
  input:
    unpaired = config['analysis_round'] + "/{comm}_unpaired.fq.gz",
    db = rules.cp_db_kr2bac.output
  output:
    text = temp(config['analysis_round'] + "/kr2bac/Kr2reports/{comm}_unpaired.kr2bac_c{conf}_kr.txt"),
    report = config['analysis_round'] + "/kr2bac/Kr2reports/{comm}_unpaired.kr2bac_c{conf}.rep"      
  params: 
    confidence = lambda wildcards: config['confidence'][wildcards.conf]
  shell: "kraken2 -db {input.db} {input.unpaired} \
  --use-names \
  --output {output.text} \
  --confidence {params.confidence} \
  --report {output.report} --report-zero-counts "


#Classify against standard database
rule kr2pair_kr2std:
  input:
    paired_R1 = config['analysis_round'] + "/Mock_{comm}_R1.P.fq",
    paired_R2 = config['analysis_round'] + "/Mock_{comm}_R2.P.fq",
    db = rules.cp_db_kr2std.output
  output:
    text = temp(config['analysis_round'] + "/kr2std/Kr2reports/{comm}_paired.kr2std_c{conf}_kr.txt"),
    report = config['analysis_round'] + "/kr2std/Kr2reports/{comm}_paired.kr2std_c{conf}.rep"
  params: 
    confidence = lambda wildcards: config['confidence'][wildcards.conf]
  shell: "kraken2 -db {input.db} --paired {input.paired_R1} {input.paired_R2} \
  --use-names \
  --output {output.text} \
  --confidence {params.confidence} \
  --report {output.report} --report-zero-counts "

rule kr2unpair_kr2std:
  input:
    unpaired = config['analysis_round'] + "/{comm}_unpaired.fq.gz",
    db = rules.cp_db_kr2std.output
  output:
    text = temp(config['analysis_round'] + "/kr2std/Kr2reports/{comm}_unpaired.kr2std_c{conf}_kr.txt"),
    report = config['analysis_round'] + "/kr2std/Kr2reports/{comm}_unpaired.kr2std_c{conf}.rep"      
  params: 
    confidence = lambda wildcards: config['confidence'][wildcards.conf]
  shell: "kraken2 -db {input.db} {input.unpaired} \
  --use-names \
  --output {output.text} \
  --confidence {params.confidence} \
  --report {output.report} --report-zero-counts "


#Classify against plasmid-renamed database
rule kr2pair_kr2plrename:
  input:
    paired_R1 = config['analysis_round'] + "/Mock_{comm}_R1.P.fq",
    paired_R2 = config['analysis_round'] + "/Mock_{comm}_R2.P.fq",
    db = rules.cp_db_kr2plrename.output
  output:
    text = temp(config['analysis_round'] + "/kr2plrename/Kr2reports/{comm}_paired.kr2plrename_c{conf}_kr.txt"),
    report = config['analysis_round'] + "/kr2plrename/Kr2reports/{comm}_paired.kr2plrename_c{conf}.rep"
  params: 
    confidence = lambda wildcards: config['confidence'][wildcards.conf]
  shell: "kraken2 -db {input.db} --paired {input.paired_R1} {input.paired_R2} \
  --use-names \
  --output {output.text} \
  --confidence {params.confidence} \
  --report {output.report} --report-zero-counts "

rule kr2unpair_kr2plrename:
  input:
    unpaired = config['analysis_round'] + "/{comm}_unpaired.fq.gz",
    db = rules.cp_db_kr2plrename.output
  output:
    text = temp(config['analysis_round'] + "/kr2plrename/Kr2reports/{comm}_unpaired.kr2plrename_c{conf}_kr.txt"),
    report = config['analysis_round'] + "/kr2plrename/Kr2reports/{comm}_unpaired.kr2plrename_c{conf}.rep"      
  params: 
    confidence = lambda wildcards: config['confidence'][wildcards.conf]
  shell: "kraken2 -db {input.db} {input.unpaired} \
  --use-names \
  --output {output.text} \
  --confidence {params.confidence} \
  --report {output.report} --report-zero-counts "


#Concatenate paired and unpaired for each library+confidence combo

rule concat_pair_unpair:
  input:
    paired = config['analysis_round'] + "/{db}/Kr2reports/{comm}_paired.{db}_c{conf}_kr.txt",
    unpaired = config['analysis_round'] + "/{db}/Kr2reports/{comm}_unpaired.{db}_c{conf}_kr.txt"
  output:
    config['analysis_round'] + "/{db}/{comm}_{db}_c{conf}_kr.txt.gz"
  group: "salmrules"
  shell:
    "cat {input.paired} {input.unpaired} | gzip > {output}"


#SALMONELLA READS AND HITS###################################################################################################################

#Get data for every read that was identified as being in the Salmonella genus
rule catch_salmhits:
  input:
    config['analysis_round'] + "/{db}/{comm}_{db}_c{conf}_kr.txt.gz"
  output:
    config['analysis_round'] + "/{db}/Salmonella/{comm}_{db}_c{conf}_salmhits.txt"
  group: "salmrules"
  shell:
    """
    zcat {input} | awk -F '\t' '($3 ~ "Salmonella") && ($3 !~ "plasmid") && ($3 !~ "virus")' - > {output}
    """


#Get read rows of Salmonella simulated reads from Kraken2 output files
rule catch_salmreads:
  input:
    config['analysis_round'] + "/{db}/{comm}_{db}_c{conf}_kr.txt.gz"
  output:
    config['analysis_round'] + "/{db}/Salmonella/{comm}_{db}_c{conf}_salmreads.txt"
  group: "salmrules"
  shell:
    """
    zcat {input} |awk -F '\t' '($2 ~ "Salmonella")' - > {output}
    """


#Extract seqs IDed as Salmonella from original metagenome files
#Output in fasta format
rule ext_salmhitIDs:
  input:
    salmhits = rules.catch_salmhits.output
  output:
    IDS = temp(config['analysis_round'] + "/{db}/Salmonella/{comm}_{db}_c{conf}_salmhitIDs.txt")
  group: "salmrules"
  shell:
    "cat {input.salmhits} | cut -f 2 > {output.IDS}"


rule ext_salmhits:
  input:
    salmhitIDs = rules.ext_salmhitIDs.output.IDS,
    seqfile_1 = config['pathtoinput'] + "Mock_{comm}_R1.fq.gz",
    seqfile_2 = config['pathtoinput'] + "Mock_{comm}_R2.fq.gz"
  output:
    fasta = config['analysis_round'] + "/{db}/Salmonella/{comm}_{db}_c{conf}_salmhits.fa",
  group: "salmrules"
  shell:
    "filterbyname.sh in1={input.seqfile_1} in2={input.seqfile_2} names={input.salmhitIDs} include=t out={output.fasta}"


#CONFIRM SALMONELLA HITS##############################################################################################################

#Compare reads IDed as Salmonella against a database of 403 species-specific regions of Salmonella enterica (from Liang et al 2017)
rule blastn_salmSSR:
  input:
    fasta = config['analysis_round'] + "/{db}/Salmonella/{comm}_{db}_c{conf}_salmhits.fa",
  output:
    config['analysis_round'] + "/{db}/SSRchecks/{comm}_{db}_c{conf}_blSSRs_salmhits.txt",
  params: 
    pathtodb = config['samSSR_db'],
    max_target_seqs = 1,
    max_hsps = 1
  group: "salmrules"
  shell:
    "blastn -db {params.pathtodb} -query {input.fasta} -out {output} -outfmt 6 -max_target_seqs {params.max_target_seqs} -max_hsps {params.max_hsps}"

#Get the names of reads that slip through as false positives
rule ext_blSSRfalsepos:
  input: config['analysis_round'] + "/{db}/SSRchecks/{comm}_{db}_c{conf}_blSSRs_salmhits.txt"
  output: config['analysis_round'] + "/{db}/SSRchecks/{comm}_{db}_c{conf}_blSSR_falsepos.txt"
  group: "salmrules"
  shell:
    """
    awk -F '\t' '($1 !~ "Salmonella")' {input} > {output}
    """