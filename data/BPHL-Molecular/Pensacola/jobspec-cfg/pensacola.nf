#!/usr/bin/env nextflow

/*
Note:
Before running the script, please set the parameters in the config file params.yaml
*/

//Step1:input data files
nextflow.enable.dsl=2
def L001R1Lst = []
def sampleNames = []
myDir = file("$params.input")

//myDir.eachFileMatch ~/.*_1.fastq.gz/, {L001R1Lst << it.name}
myDir.eachFileMatch ~/.*.bam/, {L001R1Lst << it.name}
L001R1Lst.sort()
L001R1Lst.each{
   def x = it.minus(".bam")
     //println x
   sampleNames.add(x)
}
//println L001R1Lst
//println sampleNames


//Step2: process the inputed data
A = Channel.fromList(sampleNames)
//A.view()

include { bam2fastq } from './modules/bam2fastq.nf'
include { kraken } from './modules/kraken.nf'
include { bracken } from './modules/bracken.nf'
include { pbsmrtlink } from './modules/pbsmrtlink.nf'
include { statistics } from './modules/statistics.nf'
include { variant } from './modules/variant.nf'
include { snpeff } from './modules/snpeff.nf'


workflow {
   //bam2fastq(A) | pbsmrtlink | statistics | variant | view
   bam2fastq(A) | kraken | bracken | pbsmrtlink | statistics | variant | snpeff | view
}
