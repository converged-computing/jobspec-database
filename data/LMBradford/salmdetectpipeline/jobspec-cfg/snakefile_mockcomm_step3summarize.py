#This snakefile should be used after steps 1 (setup) and 2 (analyze)
#It reads step2 output files:
##"Round[#]"/{db}/SSRchecks/{comm}_{db}_c{conf}_blSSR_falsepos.txt"
##"Round[#]/{db}/Salmonella/{comm}_{db}_c{conf}_salmreads.txt"
#And produces summary tables.

#VARIABLE SETUP ########################################################################################################################################################

#Import packages for file name management
import glob
#These imports let me use wildcards to take in file names and maintain the name info throughout the workflow
configfile: "/scratch/bradford/MockcommEnterobac_project2/configs/mockcomm_R1-2_step3_summarize.yaml"

#Detect wildcards from input files names
#Trimlogs directory has 1 file per input set, so it's a good place to catch filenames
filename = config['analysis_round'] + "/Trimlogs/{comm}_trimlog.txt"
COMM, = glob_wildcards(filename)

print(filename)
print(COMM)

#Wildcards from config dictionaries
##Major updated from snakefile v1! 
#Dynamically picks up list of databases and kr2 confidence levels from config
DB = list(config['dbs'].keys())
CONF=list(config['confidence'].keys())

#CATCH-ALL RULE ###################################################################################################################################################

rule all:
  #List all the final output files expected here.
  input:
    config['analysis_round'] + "/Results/" + config['analysis_round'] +"_results_table.txt"

#SUMMARY FILES ########################################################################################333

rule summarize_hits:
  input:
    salmhits = expand(config['analysis_round'] + "/{db}/Salmonella/{comm}_{db}_c{conf}_salmhits.txt", db = DB, comm = COMM, conf = CONF)
  output: config['analysis_round'] + "/Results/" + config['analysis_round'] +"_hits_summary.txt"
  script: "scripts/summarize_hits.py"

rule summarize_reads:
  input:
    salmhits = expand(config['analysis_round'] + "/{db}/Salmonella/{comm}_{db}_c{conf}_salmreads.txt", db = DB, comm = COMM, conf = CONF)
  output: config['analysis_round'] + "/Results/" + config['analysis_round'] +"_reads_summary.txt"
  script: "scripts/summarize_reads.py"


rule sort_blSSR:
  input: config['analysis_round'] + "/{db}/SSRchecks/{comm}_{db}_c{conf}_blSSRs_salmhits.txt"
  output: temp(config['analysis_round'] + "/{db}/SSRchecks/{comm}_{db}_c{conf}_SSRsortlist.txt")
  shell:
    "cat {input} | cut -f 1 | sort -u > {output}"

rule summarize_blSSRs:
  input: expand(rules.sort_blSSR.output, comm= COMM, db = DB, conf = CONF)
  output: config['analysis_round'] + "/Results/" + config['analysis_round'] +"_blSSRs_summary.txt"
  script: "scripts/count_checkhits.py"
  
rule summary_master:
  input:
    config['analysis_round'] + "/Results/" + config['analysis_round'] +"_reads_summary.txt",
    config['analysis_round'] + "/Results/" + config['analysis_round'] +"_hits_summary.txt",
    config['analysis_round'] + "/Results/" + config['analysis_round'] +"_blSSRs_summary.txt",
  output:
    config['analysis_round'] + "/Results/" + config['analysis_round'] +"_results_table.txt"
  script:
    "scripts/summary_table.py"
