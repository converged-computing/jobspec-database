#!/usr/bin/env nextflow

// Import subworkflows to be run in the workflow
include { checkInputs                                } from './modules/checkCohort'
include { createIntervalLists                        } from './modules/createIntervalLists'
include { mutect2                                    } from './modules/mutect2'
include { GatherVcfs                                 } from './modules/GatherVcfs'
include { MergeMutectStats                           } from './modules/MergeMutectStats'
include { LearnReadOrientationModel                  } from './modules/LearnReadOrientationModel'
include { GetPileupSummaries_T; GetPileupSummaries_N } from './modules/GetPileupSummaries'
include{  CalculateContamination                     } from './modules/CalculateContamination'
include { FilterMutectCalls                          } from './modules/FilterMutectCalls'
include { getFilteredVariants                        } from './modules/getFilteredVariants'
   


/// Print a header for your pipeline 

log.info """\

===================================================================
===================================================================
SOMATIC SHORT V - NF 
===================================================================
===================================================================

Created by the Sydney Informatics Hub, University of Sydney

Documentation	@ https://github.com/Sydney-Informatics-Hub/Somatic-shortV-nf

Cite					@ https://doi.org/10.48546/workflowhub.workflow.691.1

Log issues    @ https://github.com/Sydney-Informatics-Hub/Somatic-shortV-nf/issues

All the default parameters are set in `nextflow.config`

=======================================================================================
Workflow run parameters 
=======================================================================================
version                    : ${params.version}
input                      : ${params.input}
reference                  : ${params.ref}
dict                       : ${params.dict} 
common_biallelic_variants  : ${params.common_biallelic_variants}
ponvcf                     : ${params.ponvcf}
outDir                     : ${params.outDir}
workDir                    : ${workflow.workDir}

=======================================================================================

 """

/// Help function 
// This is an example of how to set out the help function that 
// will be run if run command is incorrect (if set in workflow) 
// or missing/  

def helpMessage() {
    log.info"""
  Usage:   nextflow run main.nf --input samples.csv \
                     --ref /path/to/ref.fasta --dict /path/to/ref.dict \
                     --ponvcf /path/to/pon \
                     --common_biallelic_variants /path/to/common_biallelic_variants

  Required Arguments:
    --input		                      Full path and name of sample input file (csv format)
	  --ref			                      Full path and name of reference genome (fasta format)
    --dict                          Full path and name of reference genome dictionary file (dict format)
    --ponvcf                        Full path and name of the Panel of Normals file (vcf format)
    --common_biallelic_variants     Full path and name of the common biallelic variant resources file (vcf format)
	
  Optional Arguments:
    --outDir                        Specify name of results directory. 
    --number_of_intervals           Define a specific number genomic-intervals for parallelisation


  HPC accounting arguments:
    --whoami                    HPC user name (Setonix or Gadi HPC)
    --gadi_account              Project accounting code for NCI Gadi (e.g. aa00)
  """.stripIndent()
}

/// Main workflow structure. 

workflow {

// Show help message if --help is run or if any required params are not 
// provided at runtime

  if ( params.help == true || params.ref == false || params.input == false || params.ponvcf == false || params.common_biallelic_variants == false || params.dict == false)
	{   

          // Invoke the help function above and exit
          helpMessage()
          exit 1
        
	} 


	else 
	{
	
  // Check inputs file exists
	checkInputs(Channel.fromPath(params.input, checkIfExists: true))
	
  // Split cohort file to collect info for each sample
	bam_pair_ch = checkInputs.out
		.splitCsv(header: true, sep:",")
		.map { row -> tuple(row.sampleID, file(row.bam_N), file(row.bam_T))}
	

//Run the processes 

// Create an intervalList based on user specified number_of_intervals
createIntervalLists(params.number_of_intervals)
intervalList=createIntervalLists.out[2].tokenize(" ") 

// Run mutect2 on a Tumor/Normal sample-pair
mutect2(bam_pair_ch,intervalList,params.ponvcf,params.ponvcf+'.tbi',createIntervalLists.out[0])


// Gather multiple VCF files from a scatter operation into a single VCF file
GatherVcfs(mutect2.out[0].collect(),bam_pair_ch)

// Combine the stats files across the scattered Mutect2 run
MergeMutectStats(mutect2.out[1].collect(),bam_pair_ch)

// Run the gatk LearnReadOrientationModel 
LearnReadOrientationModel(mutect2.out[2].collect(),bam_pair_ch)
  
// Tabulate pileup metrics for inferring contamination - Tumor samples
GetPileupSummaries_T(params.common_biallelic_variants,params.common_biallelic_variants+'.tbi', bam_pair_ch)

// Tabulate pileup metrics for inferring contamination - Normal samples
GetPileupSummaries_N(params.common_biallelic_variants,params.common_biallelic_variants+'.tbi', bam_pair_ch)
  
// Calculate the fraction of reads coming from cross-sample contamination
CalculateContamination(bam_pair_ch,GetPileupSummaries_T.out.collect(),GetPileupSummaries_N.out.collect())

// Filter somatic SNVs and indels called by Mutect2
FilterMutectCalls(bam_pair_ch,MergeMutectStats.out[1].collect(),CalculateContamination.out[0].collect(),CalculateContamination.out[1].collect(),GatherVcfs.out[0].collect(),GatherVcfs.out[1].collect(),LearnReadOrientationModel.out.collect(),params.ref)	

// Select the subset of filtered variants from the VCF file 
getFilteredVariants(bam_pair_ch,FilterMutectCalls.out.collect(),params.ref)

}}

workflow.onComplete {
  summary = """
=======================================================================================
Workflow execution summary
=======================================================================================

Duration    : ${workflow.duration}
Success     : ${workflow.success}
workDir     : ${workflow.workDir}
Exit status : ${workflow.exitStatus}
outDir      : ${params.outDir}

=======================================================================================
  """
  println summary

}
