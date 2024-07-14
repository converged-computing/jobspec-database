###################################DESEQ first#############################
args = commandArgs(trailingOnly = T)
name = args[1]

#go to the gene directory
setwd(paste0('/fs/ess/PAS0854/Active_projects/', name))
all = read.csv('RNAseq.csv')
rownames(all) = all[, 1]
all = all[, -1]
mutdata = read.csv('mut.stat.key.csv')
funcdata = read.csv('LOF.GOF.key.csv')
funcdata = funcdata[, c(2,6)]
colnames(funcdata)[1:2] = c('V1', 'V2')
funcdata1 = rbind(funcdata, mutdata[mutdata[, 3] == 'control',c(2, 3)])
funcdata2 = funcdata1[funcdata1$V2 != 'unknown', ]
all1 = all[, gsub("\\.", "-", colnames(all)) %in% funcdata1$V1]

#extract SV stat column from funcdata2
coldata = as.data.frame(funcdata2[,2])
#rename column 
colnames(coldata) = c('condition')
#make it a factor with control "Wildtype"
coldata$condition = factor(coldata$condition, levels = unique(coldata$condition))
coldata$condition <- relevel(coldata$condition, ref = "control")
#make DESeq object
dds = DESeqDataSetFromMatrix(
  countData = all1,
  colData = coldata,
  design= ~ condition)
#run DESeq
dds = DESeq(dds)

ddsrg = results(dds, tidy=T, name = "condition_GOF_vs_control")
ddsrl = results(dds, tidy=T, name = "condition_LOF_vs_control")

counts = counts(dds, normalized = T)

############################ make graphs ##################################

#expression boxplot

library(ggplot2)
#plot results

pdf(file = paste0('./', name, " Expression.pdf"))

cd = all[which(rownames(all) == name), ]
cd = t(cd)
cd = as.data.frame(cd)
cd = cbind(funcdata1$V2, cd)
colnames(cd) = c('Mut.Stat', 'Exp')

k = kruskal.test(Exp ~ Mut.Stat, data = cd)

expstat = ggplot(cd, aes(x = Mut.Stat, y = log(round(Exp, 0), 2), fill = Mut.Stat)) +
  geom_boxplot() +
  xlab(paste0(name, ' Function')) +
  ylab(paste0("Log-base-2 ", name, " Count")) +
  annotate("text", x = 1, y = -.5, label = paste0('Kruskal-Wallis: ', round(k$p.value, 2)))

print(expstat)

dev.off()

rm(cd)

#pathway map

library("AnnotationDbi")
library("org.Hs.eg.db")
library('pathview')
library('gage')
library('gageData')
library('KEGGREST')

kg.hsa=kegg.gsets("hsa")
################################GOF first##########################

dir.create("./GOF")
setwd('./GOF')

ddsrg$ensembl = mapIds(org.Hs.eg.db,
                       keys=ddsrg$row, 
                       column="ENSEMBL",
                       keytype="SYMBOL",
                       multiVals="first")
ddsrg$entrez = mapIds(org.Hs.eg.db,
                      keys=ddsrg$row, 
                      column="ENTREZID",
                      keytype="SYMBOL",
                      multiVals="first")
ddsrg$name =   mapIds(org.Hs.eg.db,
                      keys=ddsrg$row, 
                      column="GENENAME",
                      keytype="SYMBOL",
                      multiVals="first")

foldchanges = ddsrg$log2FoldChange
names(foldchanges) = ddsrg$entrez

keggres = gage(foldchanges, gsets=kg.hsa$kg.sets, same.dir=TRUE)

write.csv(keggres$stats, 'GOFvsControl_PathwayStats.csv')

#greater first

keggrespathways = data.frame(id=rownames(keggres$greater), keggres$greater) %>% 
  tbl_df() %>% 
  filter(row_number()<=5) %>% 
  .$id %>% 
  as.character()

keggresids = substr(keggrespathways, start=1, stop=8)

dir.create('./PathwaysGreater')

setwd('./PathwaysGreater')

tmp = sapply(keggresids, function(pid) pathview(gene.data=foldchanges, pathway.id=pid, species="hsa"))

setwd('..')

#less second

keggrespathways = data.frame(id=rownames(keggres$less), keggres$less) %>% 
  tbl_df() %>% 
  filter(row_number()<=5) %>% 
  .$id %>% 
  as.character()

keggresids = substr(keggrespathways, start=1, stop=8)

dir.create('./PathwaysLess')

setwd('./PathwaysLess')

tmp = sapply(keggresids, function(pid) pathview(gene.data=foldchanges, pathway.id=pid, species="hsa"))

setwd('..')

#Immune Pathways

dir.create('./PathwaysImmune')

setwd('./PathwaysImmune')

immunels = c('04640', '04610', '04611', '04613', '04620', '04624', '04621', 
             '04622', '04623', '04625', '04650', '04612', '04660', '04658', '04659', '04657', 
             '04662', '04664', '04666', '04670', '04672', '04062')

take = unlist(lapply(immunels, function(x) which(grepl(x, names(kg.hsa$kg.sets)) == T)))
keggresids = kg.hsa$kg.sets[take]
keggresids = gsub("\\ .*", "", names(keggresids))

tmp = sapply(keggresids, function(pid) pathview(gene.data=foldchanges, pathway.id=pid, species="hsa"))

setwd('../../')


#################################LOF results now##########################

dir.create("./LOF")
setwd("./LOF")

ddsrl$ensembl = mapIds(org.Hs.eg.db,
                       keys=ddsrl$row, 
                       column="ENSEMBL",
                       keytype="SYMBOL",
                       multiVals="first")
ddsrl$entrez = mapIds(org.Hs.eg.db,
                      keys=ddsrl$row, 
                      column="ENTREZID",
                      keytype="SYMBOL",
                      multiVals="first")
ddsrl$name =   mapIds(org.Hs.eg.db,
                      keys=ddsrl$row, 
                      column="GENENAME",
                      keytype="SYMBOL",
                      multiVals="first")

foldchanges = ddsrl$log2FoldChange
names(foldchanges) = ddsrl$entrez

keggres = gage(foldchanges, gsets=kg.hsa$kg.sets, same.dir=TRUE)

write.csv(keggres$stats, 'LOFvsControl_PathwayStats.csv')

#greater first

keggrespathways = data.frame(id=rownames(keggres$greater), keggres$greater) %>% 
  tbl_df() %>% 
  filter(row_number()<=5) %>% 
  .$id %>% 
  as.character()

keggresids = substr(keggrespathways, start=1, stop=8)

dir.create('./PathwaysGreater')

setwd('./PathwaysGreater')

tmp = sapply(keggresids, function(pid) pathview(gene.data=foldchanges, pathway.id=pid, species="hsa"))

setwd('..')

#less second

keggrespathways = data.frame(id=rownames(keggres$less), keggres$less) %>% 
  tbl_df() %>% 
  filter(row_number()<=5) %>% 
  .$id %>% 
  as.character()

keggresids = substr(keggrespathways, start=1, stop=8)

dir.create('./PathwaysLess')

setwd('./PathwaysLess')

tmp = sapply(keggresids, function(pid) pathview(gene.data=foldchanges, pathway.id=pid, species="hsa"))

setwd('..')

#Immune Pathways

dir.create('./PathwaysImmune')

setwd('./PathwaysImmune')

immunels = c('04640', '04610', '04611', '04613', '04620', '04624', '04621', 
             '04622', '04623', '04625', '04650', '04612', '04660', '04658', '04659', '04657', 
             '04662', '04664', '04666', '04670', '04672', '04062')

take = unlist(lapply(immunels, function(x) which(grepl(x, names(kg.hsa$kg.sets)) == T)))
keggresids = kg.hsa$kg.sets[take]
keggresids = gsub("\\ .*", "", names(keggresids))

tmp = sapply(keggresids, function(pid) pathview(gene.data=foldchanges, pathway.id=pid, species="hsa"))
#Need to run cibersort separately.



