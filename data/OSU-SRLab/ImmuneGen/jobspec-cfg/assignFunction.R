##############################Create RNAseq dataframe#######################
args = commandArgs(trailingOnly = T)
name = args[1]

#go to the gene directory
setwd(paste0('/fs/ess/PAS0854/Active_projects/', name))

#open the master and control files
master = read.csv('Master.csv')
control = read.csv('Controls.csv')

#put the master and control together
rna = rbind(master, control)

library(data.table)
library(DESeq2)

rm(master)
rm(control)

#get a list of the TCGA RNA files
rna1 = rna[grepl('TCGA', rna$ID), ]
rna1 = paste0(rna1$RNAseq, "_batch_corrected.csv")

#get a list of all the TCGA RNA files in the TCGA RNA dir
lso = list.files(path = '/fs/ess/PAS0854/Active_projects/TCGA_BatchResult/', 
                 pattern = "_batch_corrected.csv")
#read in the gene names
nam = readLines('/fs/ess/PAS0854/Active_projects/Gene_Names_TCGABatch.txt')
#get the names of the files that you need to read in
lso = lso[unlist(lapply(rna1, function(x) return(which(lso == x))))]
#read them in and put them in to a dataframe
lt = lapply(lso, function(x) 
  fread(paste0('/fs/ess/PAS0854/Active_projects/TCGA_BatchResult/', x),
        select ='x'))
lt1 = do.call(cbind, lt)
lt1 = cbind(nam, lt1)
colnames(lt1) = c('gene_symbol', lso)

#repeat the above steps for the ORIEN samples
rna1 = rna[grepl('SL', rna$RNAseq), ]
rna1 = paste0(rna1$RNAseq, "_batch_corrected.csv")

lso = list.files(path = '/fs/ess/PAS0854/Raven//MSI/RNASeq_Batch_adjusted/', 
                 pattern = "_batch_corrected.csv")
nam = readLines('/fs/ess/PAS0854/Raven/MSI/RNASeq_Batch_adjusted/Gene_Names.txt')
#get the names of the files that you need to read in
lso = lso[unlist(lapply(rna1, function(x) return(which(lso == x))))]
#read them in and put them in to a dataframe
lt = lapply(lso, function(x) 
  fread(paste0('/fs/ess/PAS0854/Raven//MSI/RNASeq_Batch_adjusted/', x),
        select ='x'))
lo1 = do.call(cbind, lt)
lo1 = cbind(nam, lo1)
colnames(lo1) = c('gene_symbol', lso)

#clean up your environment
rm(lso)
rm(lt)
rm(nam)
rm(rna1)

#create a master list of all the genes
genes = c(lo1$gene_symbol, lt1$gene_symbol)
#select only the genes that appear twice
genes = genes[genes %in% lo1$gene_symbol & genes %in% lt1$gene_symbol]
#remove duplicate entries
genes = unique(genes)

#Go to the dataframes and subset so they only include genes from master list
ori = lo1[lo1$gene_symbol %in% genes, ]
tcg = lt1[lt1$gene_symbol %in% genes, ]

#make sure the genes match the other of the master list
ori = ori[match(genes, ori$gene_symbol), ]
tcg = tcg[match(genes, tcg$gene_symbol), ]

#combine the TCGA dataframe and the ORIEN dataframe
all = cbind(ori[, 2:ncol(ori)], tcg[, 2:ncol(tcg)])
all = as.data.frame(all)
#set their gene names to the master gene list
rownames(all) = genes

#clean up environment
rm(lo1)
rm(lt1)
rm(ori)
rm(tcg)

#remove the tail of the colnames
colnames(all) = gsub('_batch_corrected.csv', "", colnames(all))

#now you need the type information from the RNA file
#get just the files with entries in the RNA column
set = rna$RNAseq[!(is.na(rna$RNAseq))]
set[set == 'none'] = NA
set = set[!(is.na(set))]
#get just the files that are in you all dataframe
set = set[set %in% colnames(all)]

rna = rna[rna$RNAseq %in% set, ]

#make sure the columns of all match the order of set
all = all[, match(set, colnames(all))]

#create coldata using the subsetting RNA dataframe
coldata  = cbind(rna$RNAseq, rna$Type)
forGraph = coldata[, 2]
coldata[, 2][!(coldata[, 2] == 'control')] = 'mutant'

write.csv(coldata, 'mut.stat.key.csv')

#create batches for batch correction
batches = grepl('TCGA', colnames(all))
batches[batches == TRUE] = 'TCGA'
batches[batches ==FALSE] = 'Orien'

library(sva)

#ensure all is a numeric matrix rounded to counts
all = apply(all, 2, function(x) as.numeric(as.character(x)))
all = round(all, 0)

#batch correct
all = ComBat_seq(all, batches)
#gene names are the same as master list
rownames(all) =genes
all = as.data.frame(all)

#write out results
write.csv(all, 'RNAseq.csv')

write.table(all, 'RNAseqForCib.txt', sep="\t")

############################determine function of variants###################
#extract SV stat column from coldata
coldata = as.data.frame(coldata[,2])
#rename column 
colnames(coldata) = c('condition')
#make it a factor with control "Wildtype"
coldata$condition = factor(coldata$condition, levels = unique(coldata$condition))
coldata$condition <- relevel(coldata$condition, ref = "control")
#make DESeq object
dds = DESeqDataSetFromMatrix(
  countData = all,
  colData = coldata,
  design= ~ condition)
#run DESeq
dds = DESeq(dds)

ddsr = results(dds, tidy=T)

counts = counts(dds, normalized = T)

library(KEGGREST)

#get a list of all the pathways associated with GOI
pathwaylist = keggFind("genes", name)

#make dataframe with GOI and mut.stat
cd = counts[which(rownames(counts) == name), ]
cd = as.data.frame(cd)
cd = cbind(forGraph, cd)
colnames(cd) = c('Mut.Stat', 'Exp')

#make control dataframe
cdc = subset(cd, Mut.Stat == 'control')
cdc = as.data.frame(cdc)

#make sure expression is numeric
cdc$Exp = as.numeric(as.character(cdc$Exp))

#put the low samples (<Q2) in cdl
low = summary(cdc$Exp)[2][[1]]
cdl = rownames(cdc[cdc$Exp < low, ])
#put the low samples (>Q3) in cdh
high = summary(cdc$Exp)[4][[1]]
cdh = rownames(cdc[cdc$Exp > high, ])
group = c(cdl, cdh)

cdc = cdc[!(rownames(cdc) %in% cdl) & !(rownames(cdc) %in% cdh), ]

library("AnnotationDbi")
library("org.Hs.eg.db")
library('pathview')
library('gage')
library('gageData')
library('dplyr')
library('vegan')

#extract SV stat column from testdata
testdata = cbind(group, c(rep('low', length(cdl)), rep('high', length(cdh))))
allt = counts[, colnames(counts) %in% testdata[, 1]]
allc = counts[, colnames(counts) %in% rownames(cdc)]
allt = allt[, match(testdata[, 1], colnames(allt))]
testdata = as.data.frame(testdata)
#rename column 
colnames(testdata) = c('names', 'condition')
#make it a factor with control "Wildtype"
testdata$condition = factor(testdata$condition, levels = unique(testdata$condition))
#make DESeq object
allt = round(allt, 0)
ddst = DESeqDataSetFromMatrix(
  countData = allt,
  colData = testdata,
  design= ~ condition)
#run DESeq
ddst = DESeq(ddst)

ddstr = results(ddst, tidy=T)

#extract counts from ddst
countst = counts(ddst)

#rename genes so they match with names from gage
ddstr$ensembl = mapIds(org.Hs.eg.db,
                       keys=ddstr$row, 
                       column="ENSEMBL",
                       keytype="SYMBOL",
                       multiVals="first")
ddstr$entrez = mapIds(org.Hs.eg.db,
                      keys=ddstr$row, 
                      column="ENTREZID",
                      keytype="SYMBOL",
                      multiVals="first")
ddstr$name =   mapIds(org.Hs.eg.db,
                      keys=ddstr$row, 
                      column="GENENAME",
                      keytype="SYMBOL",
                      multiVals="first")

#store fold changes in 'foldchanges'
foldchanges = ddstr$log2FoldChange
#preserve names
names(foldchanges) = ddstr$entrez

#extract the HSA results from pathwaylist
newnames = names(pathwaylist)[grepl('hsa', names(pathwaylist))]

#get the pathway from the HSA results, only need 1st one
query <- keggGet(newnames[1])
newnames = paste(names(query[[1]]$PATHWAY), query[[1]]$PATHWAY)

#store all the kegg pathways in kg.hsa
kg.hsa=kegg.gsets("hsa")
#extract all the pathways (in correct format) based on newnames
take = unlist(lapply(newnames, function(x) which(grepl(x, names(kg.hsa$kg.sets)) == T)))
take = kg.hsa$kg.sets[take]

#perform gage on those pathways
keggres = gage(foldchanges, gsets=take, same.dir=TRUE)
stats = keggres$stats

#extract HSA pathways that have a log fold change >2
test = keggGet(gsub("\\ .*", "", rownames(stats)))
#get genes associated with HSA pathways from previous step
test = unlist(lapply(1:length(test[[1]]$GENE[grepl(";", test[[1]]$GENE)]), 
                     function(x) 
                       unlist(strsplit(test[[1]]$GENE[grepl(";", test[[1]]$GENE)], ";")[[x]][1])))
test = unique(test)
#check significance of each gene in test
test2 = c()
for (x in test){
  padj = ddstr[ddsr$row == x, 7]
  if (padj < .001){
    test2 = c(test2, x)
  } else{
    test2 = c(test2, NA)
  }
}
test2 = test2[!(is.na(test2))]
test = test2
rm(test2)


#find most different gene in the list not your gene
idea = ddstr[ddstr$row %in% test, c('row', 'log2FoldChange')]
idea = idea[order(-abs(idea$log2FoldChange)),]
idea = idea[idea[, 1] != name, ]
idea = idea[1:10, ]
idea = idea[!(is.na(idea$row)), ]

mutdat = cbind(colnames(all), coldata)
mutdat = mutdat[mutdat[, 2] != 'control', ]
#create allm with the counts object and the anems from mutdat
allm = counts[, colnames(counts) %in% mutdat[, 1]]

key = matrix(colnames(allm), ncol(allm), 1)
for (y in 1:nrow(idea)){
  GOI = idea[y, 1]
  #take only the rownames from previous step
  allm1 = allm[rownames(allm) == GOI, ]
  #extract the log fold change
  stat = ddstr[ddstr$row == GOI, 'log2FoldChange']
  func = c()
  if (stat > 0){
    #if the gene is increased in the high group
    sum = summary(as.numeric(allm1))
    h = sum[5]
    l = sum[2]
    #higher than 3rd quart = GOF
    #lower than 1st quart = LOF
    for (x in 1:length(allm1)){
      if (allm1[x] >= h){
        func = c(func, 'GOF')
      } else if (allm1[x] <= l){
        func = c(func, 'LOF')
      } else {
        func = c(func, 'unknown')
      }
    }
  } else {
    #opposite as above if the log fold change in lower in high
    sum = summary(as.numeric(allm1))
    h = sum[5]
    l = sum[2]
    for (x in 1:length(allm1)){
      if (allm1[x] >= h){
        func = c(func, 'LOF')
      } else if (allm1[x] <= l){
        func = c(func, 'GOF')
      } else {
        func = c(func, 'unknown')
      }
    }
  }
  key = cbind(key, func)
  colnames(key)[ncol(key)] = idea[y,1]
}

key2 = matrix(key[, 1], nrow(key), 1)
gof = c()
lof = c()
unk = c()
for (x in 1:nrow(key)){
  row = key[x, 2:ncol(key)]
  gof = c(gof, length(row[row == 'GOF']))
  lof = c(lof, length(row[row == 'LOF']))
  unk = c(unk, length(row[row == 'unknown']))
}
key2 = cbind(key2, gof, lof, unk)
key2 = as.data.frame(key2)
key2$gof = as.numeric(as.character(key2$gof))
key2$lof = as.numeric(as.character(key2$lof))
key2$unk = as.numeric(as.character(key2$unk))

func2 = unlist(lapply(1:nrow(key2), function(x){
  if (sum(key2[x, 2:3]) == 0){
    return('unknown')
  } else if (key2[x, 2] > 0 && key2[x, 3] == 0){
    return('GOF')
  } else if (key2[x, 2] == 0 && key2[x, 3] > 0) {
    return('LOF')
  } else {
    return('check')
  }
}))
key2 = cbind(key2, func2)
ma2 = key2[key2$func2 == 'check',]
ma3 = key2[key2$func2 != 'check',]

pval = c()
est = c()
for (x in 1:nrow(ma2)){
  bi = binom.test(ma2[x, 2], ma2[x, 2]+ ma2[x, 3], .5)
  pval = c(pval, bi$p.value)
  est = c(est, bi$estimate[[1]])
}
ma2 = cbind(ma2, pval, est)
ma2$func2 = unlist(lapply(1:nrow(ma2), function(x){
  if (ma2$pval[x] < .05){
    if (ma2$est[x] > .5){
      return('GOF')
    } else {
      return('LOF')
    }
  } else {
    return('unknown')
  }
}))

key = rbind(ma3, ma2[, 1:5])

#combine names and function

master = read.csv('Master.csv')

func = c()
for (x in 1:nrow(master)){
  if (master$RNAseq[x] %in% key[, 1]){
    func = c(func, key[, 5][which(key[, 1] == master$RNAseq[x])])
  } else {
    func = c(func, 'unknown')
  }
}

master = cbind(master, func)
colnames(master)[ncol(master)] = 'Function'
write.csv(master, 'Master.csv')

#write it out
write.csv(key, 'LOF.GOF.key.csv')
