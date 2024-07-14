#!/usr/bin/env nextflow

/*
Note:
Before running the script, please set the parameters in the config file params.yaml
*/

//Step1:input data files
nextflow.enable.dsl=2

def R1Lst = []
def sampleNames = []

myDir = file("$params.input")
myDir.eachFileMatch ~/.*_1.fastq.gz/, {R1Lst << it.name}

R1Lst.sort()
R1Lst.each{
   def x = it.minus("_1.fastq.gz")
     println x
   sampleNames.add(x)
}

//Step2: process the inputed data

T = Channel.fromList(sampleNames)

include { fastqc } from './modules/fastqc.nf'
include { humanscrubber } from './modules/humanscrubber.nf'
include { trimmomatic } from './modules/trimmomatic.nf'
include { bbduk } from './modules/bbduk.nf'
include { fastqc_clean } from './modules/fastqc_clean.nf'
include { multiqc } from './modules/multiqc.nf'
include { frag_bwa } from './modules/frag_bwa.nf'
include { frag_samtools } from './modules/frag_samtools.nf'
include { primer_trim_ivar } from './modules/primer_trim_ivar.nf'
include { primer_trim_samtools } from './modules/primer_trim_samtools.nf'
include { assembly } from './modules/assembly.nf'
//include { stats } from './modules/stats.nf'
//include { vadr } from './modules/vadr.nf'
include { pystats } from './modules/pystats.nf'
workflow {
    //quality(T) | frag | primer | assembly | pystats | view
    fastqc(T) | humanscrubber | trimmomatic | bbduk | fastqc_clean | multiqc | frag_bwa | frag_samtools | primer_trim_ivar | primer_trim_samtools | assembly | pystats | view
}
