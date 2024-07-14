# File: 08_shearwaterML.R
# Auth: umar.niazi@kcl.ac.uk
# DESC: somatic mutations analysis
# Date: 27/09/2021

source('header.R')
library(GenomicRanges)
library(org.Hs.eg.db)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)

# select some genes of interest
cvSymbols = c('NOTCH1', 'TP53', 'TP63')
dfGenes = AnnotationDbi::select(org.Hs.eg.db, keys = cvSymbols, 
                                columns = c('ENTREZID'),
                                keytype = 'SYMBOL')

dfGenes

# get the genes into GRanges object
oGRgenes = genes(TxDb.Hsapiens.UCSC.hg38.knownGene)
oGRgenes = oGRgenes[as.character(dfGenes$ENTREZID)]

library(deepSNV)
# see here for usage
# https://www.bioconductor.org/packages/devel/bioc/vignettes/deepSNV/inst/doc/shearwaterML.html
csFiles = list.files('dataExternal/subsample/', pattern = '*.bam$',full.names = T)

arCounts = loadAllData(csFiles, oGRgenes, q=30)
pvals = betabinLRT(arCounts)$pvals
qvals = p.adjust(pvals, method = 'BH')
dim(qvals) = dim(pvals)
oVcf = qvals2Vcf(qvals, arCounts, oGRgenes, samples = csFiles, mvcf = TRUE)
dir.create('results')
save(oVcf, file='results/oVcf.rds')
