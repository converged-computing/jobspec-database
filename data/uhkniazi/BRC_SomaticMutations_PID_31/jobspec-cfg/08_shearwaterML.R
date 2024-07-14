# File: 08_shearwaterML.R
# Auth: umar.niazi@kcl.ac.uk
# DESC: somatic mutations analysis
# Date: 21/10/2022

source('header.R')
library(GenomicRanges)
library(org.Hs.eg.db)
library(TxDb.Hsapiens.UCSC.hg19.knownGene)

## get the tumor file index from commandline
args = as.numeric(commandArgs(trailingOnly = TRUE))
print(args)
#stop()
# select some genes of interest
dfGenes = read.csv('dataExternal/pilotProjectGeneList.csv', stringsAsFactors = F,
                   header=F)
# select some genes of interest
cvSymbols = dfGenes$V1
# remove duplicated symbols and whitespace
cvSymbols = gsub(' ', '', cvSymbols)
length(cvSymbols)
cvSymbols = unique(cvSymbols); length(cvSymbols)
dfGenes = AnnotationDbi::select(org.Hs.eg.db, keys = cvSymbols, 
                                columns = c('ENTREZID'),
                                keytype = 'SYMBOL')

dim(dfGenes)
dfGenes = na.omit(dfGenes); dim(dfGenes)
rownames(dfGenes) = dfGenes$ENTREZID
# get the genes into GRanges object
oGRgenes = genes(TxDb.Hsapiens.UCSC.hg19.knownGene,
                 filter=list('gene_id'=dfGenes$ENTREZID))
table(rownames(dfGenes) %in% names(oGRgenes))
## synchronize both objects w.r.t. gene ids
dfGenes = dfGenes[names(oGRgenes),]
dim(dfGenes)
identical(rownames(dfGenes), names(oGRgenes))
oGRgenes$SYMBOL = dfGenes$SYMBOL
### this chunk may vary w.r.t. file paths if run on 
### hpc instead of local machine
# load meta data
dfMeta = read.csv('dataExternal/filelist.csv', header=T, stringsAsFactors = F)
str(dfMeta)

#### set working directory to appropriate location with bam files
setwd('results/bams/')
csFiles = list.files('.', pattern = '*sortc.bam$')
## match the file names to sample ids in metadata table
f = gsub('_q30_.+', '.bam', csFiles)
table(dfMeta$fname %in% f)
i = match(dfMeta$fname, f)
identical(dfMeta$fname, f[i])
dfMeta$fname_sorted = csFiles[i]

## load the list of files with duplicates removed
csFiles = list.files('.', pattern = '*sortc_rd.bam$')
f = gsub('_q30_.+', '.bam', csFiles)
table(dfMeta$fname %in% f)
i = match(dfMeta$fname, f)
identical(dfMeta$fname, f[i])
dfMeta$fname_sorted_rd = csFiles[i]

dim(dfMeta)

###### adding shearwater code here
library(deepSNV)
library(BSgenome.Hsapiens.UCSC.hg19)
oBSgenome = BSgenome.Hsapiens.UCSC.hg19
### the seqlevels style of the bam files and gene names don't match
### change to match
seqlevelsStyle(oGRgenes) = 'NCBI'
seqlevelsStyle(oBSgenome) = 'NCBI'
# regions <- oGRgenes
#counts <- loadAllData(files, regions, q=10)

### error checks if the files have problems or not
# lTry = lapply(dfMeta$fname_sorted_rd,  function(x){
#   tryCatch(loadAllData(x, oGRgenes[1], q=25), error=function(e) return(NULL))  
# })
# 
# table(sapply(lTry, is.null))
# ## if all ok continue

tumourbams = dfMeta$fname_sorted_rd[args[1]]
normalbams = dfMeta$fname_sorted_rd[-c(args[1])]
cvTumSampleName = as.character(dfMeta$title[args[1]])
min_totest = 2 # Minimum number of mutant reads in the tumour samples to test a site

# Choose prior, either null:
prior        <- NULL

## Subfunction: Estimating rho (I use grid ML search but we could use method of moments estimator as mg14 does)

estimateRho_gridml = function(x, mu, prior=NULL) { # By Tim Coorens	
  rhovec = 10^seq(-6,-0.5,by=0.05) # rho will be bounded within 1e-6 and 0.32
  mm = c(x[,3:4])
  cov = c(x[,1:2])
  if(is.null(prior)){
    ll = sapply(rhovec, function(rhoj) sum(dbetabinom(x=mm, size=cov, rho=rhoj, prob=mu, log=T)))
  }
  else{
    ll = sapply(rhovec, function(rhoj) sum(dbetabinom(x=mm, size=cov, rho=rhoj, prob=mu, log=T))) + log(prior)
  }
  rhovec[ll==max(ll)][1]
}


logbb = deepSNV:::logbb
dbetabinom = VGAM::dbetabinom
cvNucleotides = c("A", "T", "C", "G", "-", "a", "t", "c", "g", 
                  "_")
myShearWaterML = function(iGeneIndex){
  regions = oGRgenes[iGeneIndex]
  
  tumcounts_obj_all = loadAllData(tumourbams, regions, q=25)
  ## loading the normal counts
  normcounts_obj = loadAllData(normalbams, regions, q=25)
  
  norm_total = apply(normcounts_obj[,,1:5]+normcounts_obj[,,6:10], c(2,3), sum) # Global counts from the reference panel of samples
  norm_total = norm_total/rowSums(norm_total) # Relative global counts (we will exclude from testing any site with >=40% mismatches in the normal panel, removing the reference bases)
  if(length(tumourbams) > 1) {
    tum_total = apply(tumcounts_obj_all[,,1:5]+tumcounts_obj_all[,,6:10], c(2,3), sum) # Global counts from the tumour samples
  } else {
    tum_total = tumcounts_obj_all[1,,1:5]+tumcounts_obj_all[1,,6:10]
  }
  tum_total = tum_total/rowSums(tum_total)
  sample_list = cvTumSampleName#'tum_test' #sapply(strsplit(tumourbams,"/"), function(x) substr(x[length(x)],1,nchar(x[length(x)])-4)) 
  
  ## Likelihood-Ratio Test
  
  mutations_allsamples = NULL
  for (tum in 1:length(tumourbams)) { # Iteratively running the code for the 1-sample wrapper_shearwaterML.R code
    
    # Recreating the 1-sample tumcounts_obj object
    tumcounts_obj = tumcounts_obj_all;
    tumcounts_obj = array(tumcounts_obj[tum,,], dim=c(1,dim(tumcounts_obj[tum,,])))
    
    #### norm_total < 0.4 - relative global count >=40%, remove from the analysis - see a few lines before in the code
    mutsites = which((tumcounts_obj[1,,1:5] + tumcounts_obj[1,,6:10])>=min_totest & norm_total<0.4, arr.ind=T) # Selecting promising sites
    
    if (nrow(mutsites)>0) {
      
      # Modified by fa8 to include rho in the output
      #mutations = data.frame(sampleID=sample_list[tum], chr=tumcounts_obj$coordinates$chr[mutsites[,1]], pos=tumcounts_obj$coordinates$pos[mutsites[,1]], ref=NA, mut=tumcounts_obj$nucleotides[mutsites[,2]], xfw=NA, xbw=NA, nfw=NA, nbw=NA, mu=NA, pval=NA)
      mutations = data.frame(sampleID=sample_list[tum], 
                             chr=as.character(seqnames(regions)),
                             pos=start(regions)+mutsites[,1]-1,
                             ref=NA,
                             mut=cvNucleotides[mutsites[,2]],
                             xfw=NA, xbw=NA, nfw=NA, nbw=NA, mu=NA, pval=NA,rho=NA)
      # relative frequency/proportion
      mutations$tum_globalvaf = apply(mutsites, 1, function(x) tum_total[x[1],x[2]]) # relative counts
      l = length(cvNucleotides)/2
      # index for fw and rev strand necleotide position
      indsm = cbind(mutsites[,2], mutsites[,2]+l)
      sites_gr = GRanges(mutations$chr, IRanges(mutations$pos,mutations$pos))
      ### sort this after using human genome BSGenome object
      # seqs = as.character(scanFa(genome_file, sites_gr))
      seqs = getSeq(oBSgenome, sites_gr)
      mutations$ref = as.character(seqs)
      # for each mutant sample list, loop through each identified site
      for (j in 1:nrow(mutations)) {
        
        inds = indsm[j,]
        # row position (region coordinate) in 1st column of mutsites array/matrix
        norcounts = normcounts_obj[,mutsites[j],] # using all normal panel
        tumcounts = tumcounts_obj[,mutsites[j],] # using single mutant file
        
        # Shearwater test
        pseudo = .Machine$double.eps
        # counts in tumour sample on fw, rv, and total coverage 
        x.fw = tumcounts[inds[1]]
        x.bw = tumcounts[inds[2]]
        n.fw = sum(tumcounts[1:l])
        n.bw = sum(tumcounts[(l+1):(2*l)])
        
        # same but for all normal sample panel
        X.fw = sum(norcounts[,inds[1]])
        X.bw = sum(norcounts[,inds[2]])
        N.fw = sum(norcounts[,1:l])
        N.bw = sum(norcounts[,(l+1):(2*l)])
        # using normal samples to estimate rho
        mu = max(X.fw+X.bw,pseudo)/max(N.fw+N.bw,pseudo)
        counts = cbind(rowSums(norcounts[,1:l]),rowSums(norcounts[,(l+1):(2*l)]),norcounts[,inds])
        rho = estimateRho_gridml(counts,mu,prior)
        
        ###############################
        #### Here we can save rho  ####
        ###############################
        
        
        rdisp = (1 - rho)/rho
        
        prob0.fw = (X.fw + x.fw)/(N.fw + n.fw); prob0.fw[prob0.fw==0] = pseudo
        prob1s.fw = x.fw/(n.fw+pseudo); prob1s.fw[prob1s.fw==0] = pseudo
        prob1c.fw = X.fw/(N.fw+pseudo); prob1c.fw[prob1c.fw==0] = pseudo
        prob1s.fw = pmax(prob1s.fw,prob1c.fw) # Min error rate is that of the population (one-sided test)
        # Modified by fa8 to avoid p=1, which results in beta=0
        nu0.fw = prob0.fw * rdisp; nu1s.fw = min(1-pseudo,prob1s.fw) * rdisp; nu1c.fw = min(1-pseudo,prob1c.fw) * rdisp; 
        #nu0.fw = prob0.fw * rdisp; nu1s.fw = prob1s.fw * rdisp; nu1c.fw = prob1c.fw * rdisp; 
        
        prob0.bw = (X.bw + x.bw)/(N.bw + n.bw); prob0.bw[prob0.bw==0] = pseudo
        prob1s.bw = x.bw/(n.bw+pseudo); prob1s.bw[prob1s.bw==0] = pseudo
        prob1c.bw = X.bw/(N.bw+pseudo); prob1c.bw[prob1c.bw==0] = pseudo
        prob1s.bw = pmax(prob1s.bw,prob1c.bw) # Min error rate is that of the population (one-sided test)
        # Modified by fa8 to avoid p=1, which results in beta=0
        nu0.bw = prob0.bw * rdisp; nu1s.bw = min(1-pseudo,prob1s.bw) * rdisp; nu1c.bw = min(1-pseudo,prob1c.bw) * rdisp; 
        #nu0.bw = prob0.bw * rdisp; nu1s.bw = prob1s.bw * rdisp; nu1c.bw = prob1c.bw * rdisp; 
        
        # Likelihood-Ratio Tests
        LL.fw = logbb(x.fw, n.fw, nu0.fw, rdisp) + logbb(X.fw, N.fw, nu0.fw, rdisp) - logbb(x.fw, n.fw, nu1s.fw, rdisp) - logbb(X.fw, N.fw, nu1c.fw, rdisp)
        pvals_fw = pchisq(-2*LL.fw, df=1, lower.tail=F)/2 # We divide by 2 as we are performing a 1-sided test
        LL.bw = logbb(x.bw, n.bw, nu0.bw, rdisp) + logbb(X.bw, N.bw, nu0.bw, rdisp) - logbb(x.bw, n.bw, nu1s.bw, rdisp) - logbb(X.bw, N.bw, nu1c.bw, rdisp)
        pvals_bw = pchisq(-2*LL.bw, df=1, lower.tail=F)/2 # We divide by 2 as we are performing a 1-sided test
        pvals_both = pchisq(-2*(log(pvals_fw)+log(pvals_bw)),4,low=F) # Fisher's combined p-value
        
        # Saving the result
        # Modified by fa8 to save rho
        #mutations[j,6:11] = c(x.fw, x.bw, n.fw, n.bw, mu, pvals_both)
        mutations[j,6:12] = c(x.fw, x.bw, n.fw, n.bw, mu, pvals_both,rho)
      }
      mutations_allsamples = rbind(mutations_allsamples, mutations)
    }
  }
  
  
  # Fa8: not testing T>T or A>A (related to alt homozygotes):
  if(nrow(mutations_allsamples) > 0) { # "If" suggested by Andrew 
    mutations_allsamples <- mutations_allsamples[which(mutations_allsamples$ref!=mutations_allsamples$mut),] 
  }
  # end of modification
  ## calculated adjusted p-value
  mutations_allsamples$p.adj = p.adjust(mutations_allsamples$pval, method = 'BH')
  mutations_allsamples$ENTREZID = regions$gene_id
  mutations_allsamples$SYMBOL = regions$SYMBOL
  gc(full = T)
  return(mutations_allsamples)
}

ldfResults = lapply(seq_along(oGRgenes), function(iRegion){
  tryCatch(myShearWaterML(iRegion), error=function(e) return(NA))
})

length(ldfResults)
names(ldfResults) = oGRgenes$SYMBOL
# remove any NA genes i.e. where trycatch sent an error
# this can happen when no mutations were detected on a gene
ldfResults[is.na(ldfResults)] = NULL
length(ldfResults)
dfResults = do.call(rbind, ldfResults)
dim(dfResults)
table(complete.cases(dfResults))
dfResults = dfResults[complete.cases(dfResults), ]
# table(dfResults$p.adj < 0.01)
# dfResults.adj = dfResults[dfResults$p.adj < 0.01, ]
n = (paste0('../../results/', cvTumSampleName, '.csv'))
#setwd(gcswd)
write.csv(dfResults, file=n)
######



###################### old code to clean up
## start with the 1st test sample and others as controls
# csFiles = c(test=dfMeta$files[8], control=dfMeta$files[-8])
# library(deepSNV)
# loVCF = lapply(seq_along(oGRgenes), function(x){
#   arCounts = loadAllData(csFiles, oGRgenes[x], q=30)
# })
# setwd(gcswd)
# names(loSnv) = dfGenes[names(oGRgenes),'SYMBOL']
# csSave = paste0('results/', csFiles['test'], '_loSnv.rds')
# ## saved after running on rosalind
# save(loSnv, file=csSave)
# 
# ## load from here in case this was run on a different machine
# ## or continue analysis from here
# load('results/WTCHG_894903_73075379_q30_fixm_sortc_rd.bam_loSnv.rds')
# iMetaIndex = 7
# 
# lResults = lapply(loSnv, myDeepSNVSummary, value='data.frame')
# ## create a data frame
# dfResults = do.call(rbind, lResults)
# i = match(rownames(dfResults), dfGenes$SYMBOL)
# identical(rownames(dfResults), dfGenes$SYMBOL[i])
# ## if this is false, it is probably due to multiple hits on single gene
# ## otherwise skip this matching section
# s = rownames(dfResults)
# s = gsub('\\.\\d*', '', s)
# i = match(s, dfGenes$SYMBOL)
# identical(s, dfGenes$SYMBOL[i])
# 
# dfResults$ENTREZID = dfGenes$ENTREZID[i]
# cvName = paste0('results/varCall', dfMeta$SampleID[iMetaIndex], '.csv')
# write.csv(dfResults, file=cvName)
# 
# ## convert to VCF
# loVcf = lapply(loSnv, myDeepSNVSummary, value='VCF')
# # drop the null values
# loVcf[sapply(loVcf, is.null)] = NULL
# oVcf = do.call(rbind, loVcf)
# cvName = paste0('results/varCall', dfMeta$SampleID[iMetaIndex], '.vcf')
# writeVcf(oVcf, file=cvName)
# 
# 
# ######### old test code
# ##################################################################
# source('header.R')
# library(GenomicRanges)
# library(org.Hs.eg.db)
# library(TxDb.Hsapiens.UCSC.hg38.knownGene)
# 
# # select some genes of interest
# cvSymbols = c('NOTCH1', 'TP53', 'TP63')
# dfGenes = AnnotationDbi::select(org.Hs.eg.db, keys = cvSymbols,
#                                 columns = c('ENTREZID'),
#                                 keytype = 'SYMBOL')
# 
# dfGenes
# 
# # get the genes into GRanges object
# oGRgenes = genes(TxDb.Hsapiens.UCSC.hg38.knownGene)
# oGRgenes = oGRgenes[as.character(dfGenes$ENTREZID)]
# 
# library(deepSNV)
# # see here for usage
# # https://www.bioconductor.org/packages/devel/bioc/vignettes/deepSNV/inst/doc/shearwaterML.html
# csFiles = list.files('dataExternal/subsample/', pattern = '*.bam$',full.names = T)
# 
# arCounts = loadAllData(csFiles, oGRgenes, q=10)
# pvals = betabinLRT(arCounts)$pvals
# qvals = p.adjust(pvals, method = 'BH')
# dim(qvals) = dim(pvals)
# oVcf = qvals2Vcf(qvals, arCounts, oGRgenes, samples = csFiles, mvcf = TRUE)
# dir.create('results')
# save(oVcf, file='results/oVcf.rds')
# #################################################################