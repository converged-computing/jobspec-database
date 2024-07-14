#! /usr/bin/env
args=commandArgs(trailingOnly = T)

if (length(args)!=3) {
  stop("Please supply the following three arguments in order:
       If you want raw coverage counts:
       bamFileName bigwigFileName raw
       If you want normalised (reads per million) coverage:
       bamFileName bigwigFileName rpm\n")
}


#bamFileName<-"./bamSTAR/493_HS1.sorted.bam"
bamFileName<-args[1]
bigwigFileName<-args[2]
normalise<-ifelse(args[3]=="rpm",T,F)

#' Calculate read coverage from bam file
#'
#' Calculate raw or normalised coverage from bam file
#' @param bamFileName Name of the bam file to use to calculate coverage
#' @param normalise Should the coverage be normalised as reads per million
#' @return genomic ranges of read coverage
getReadCoverage<-function(bamFileName,normalise=F){
  bamFile<-GenomicAlignments::readGAlignments(bamFileName)
  sampleCoverage<-GenomicAlignments::coverage(bamFile)[1:7]
  if(normalise){
    rpm_norm <- sampleCoverage/length(bamFile)*1e6
    return(rpm_norm)
  } else {
    return(sampleCoverage)
  }
}


bwDir<-dirname(bigwigFileName)
if(!dir.exists(bwDir)){
  dir.create(bwDir,recursive=T)
}

if(normalise==T){
  myCov<-getReadCoverage(bamFileName,normalise=T)
} else {
  myCov<-getReadCoverage(bamFileName,normalise=F)
}

rtracklayer::export.bw(myCov,bigwigFileName)

