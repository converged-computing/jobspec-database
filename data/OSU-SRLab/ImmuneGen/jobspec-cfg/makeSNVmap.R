args = commandArgs(trailingOnly = T)
name = args[1]
reg = args[2]
reg1 = strsplit(strsplit(reg, "-")[[1]][1], ":")[[1]][2]
reg2 = strsplit(reg, "-")[[1]][2]

#load the filtered data frame

ts = read.table('TCGA.SNV.maf', sep="\t", row.names = NULL, header=T, fill=NA)

os = read.table('Orien.SNV.maf', sep="\t", row.names = NULL, header=T, fill=NA)
os = os[os$Gene.refGene == name, ]

###########################################Get list of SNVs and functions##################

master=read.csv('Master.csv')
msnv = master[grepl('SNV', master$Type), ]
msnvt = msnv[grepl('TCGA', msnv$ID), ]
msnvo = msnv[!grepl('TCGA', msnv$ID), ]
altso = c()
startso = c()
for (x in 1:nrow(msnvo)){
  samp = msnvo$WXS.WGS[x]
  alt = os$GeneDetail.refGene[grepl(samp, os$Sample)]
  alt = unlist(lapply(strsplit(alt, ";"), function(x) x[1]))
  alt = paste0(unique(alt), collapse = ";")
  altso = c(altso, alt)
  st = paste0(unique(os$Start[grepl(samp, os$Sample)]), collapse=";")
  startso = c(startso, st)
}
altso[grepl('dist', altso)] = "."
alignment = strsplit(altso[altso != "."][1], ":")[[1]][1]
altst = c()
startst = c()
for (x in 1:nrow(msnvt)){
  samp = msnvt$ID[x]
  alt = ts$all_effects[grepl(samp, ts$Tumor_Sample_Barcode)]
  alt = strsplit(alt, ",")
  alt = unlist(lapply(alt, function(x){
    id = which(grepl(alignment, x));
    paste0(c(x[id], x[id + 1]), collapse=":")
  }))
  alt = paste0(alt, collapse = ";")
  altst = c(altst, alt)
  startst = c(startst, paste0(unique(ts$Start_Position[grepl(samp, ts$Tumor_Sample_Barcode)]), collapse = ";"))
}

test = cbind(rbind(msnvt[, c('ID', "Function")], msnvo[, c('ID', "Function")]), c(altst, altso), c(startso, startst))
colnames(test) = c('ID', 'Function', 'Alts', 'Starts')
write.table(test, 'SNV.bySample.txt', sep="\t")

alts = paste0(test$Alts, collapse = ";")
alts = unlist(strsplit(alts, ";"))
alts = unlist(lapply(strsplit(alts, ":"), function(x) return(x[2])))
alts = unique(alts)
ma = matrix(NA, nrow =1, ncol = 4)
colnames(ma) = c('Alt', 'GOF', "LOF", "Unknown")
for (x in alts){
  row = c(x)
  ids = which(grepl(x, test$Alts) == T)
  ft = table(test$Function[ids])
  if ('GOF' %in% names(ft)){
    row = c(row, ft[which(names(ft) == 'GOF')][[1]])
  } else {
    row = c(row, 0)
  }
  if ('LOF' %in% names(ft)){
    row = c(row, ft[which(names(ft) == 'LOF')][[1]])
  } else {
    row = c(row, 0)
  }
  if ('unknown' %in% names(ft)){
    row = c(row, ft[which(names(ft) == 'unknown')][[1]])
  } else {
    row = c(row, 0)
  }
  ma = rbind(ma, row)
}
ma = ma[-1, ]
ma = as.data.frame(ma)
ma[, 2] = as.numeric(as.character(ma[, 2]))
ma[, 3] = as.numeric(as.character(ma[, 3]))
ma[, 4] = as.numeric(as.character(ma[, 4]))

func2 = unlist(lapply(1:nrow(ma), function(x){
  if (sum(ma[x, 2:3]) == 0){
    return('unknown')
  } else if (ma[x, 2] > 0 && ma[x, 3] == 0){
    return('GOF')
  } else if (ma[x, 2] == 0 && ma[x, 3] > 0) {
    return('LOF')
  } else {
    return('check')
  }
}))
ma = cbind(ma, func2)
ma2 = ma[ma$func2 == 'check',]
ma3 = ma[ma$func2 != 'check',]

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
ma2 = ma2[, -c(6, 7)]
ma = rbind(ma2, ma3)
start = c()
for (x in 1:nrow(ma)){
  alt = ma[x, 1]
  alt = gsub("\\+", "\\\\+", alt)
  alt = gsub("\\.", "\\\\.", alt)
  alt = gsub("\\*", "\\\\*", alt)
  alt = gsub("\\>", "\\\\>", alt)
  if (T %in% grepl(alt, ts$all_effects)){
    id = which(grepl(alt, ts$all_effects) == T)[1]
    if (is.na(ts$Start_Position[id])){
      start = c(start, ts$End_Position[id])
    } else {
      start = c(start, ts$Start_Position[id])
    }
  } else {
    id = which(grepl(alt, os$GeneDetail.refGene) == T)[1]
    if (is.na(os$Start[id])){
      start = c(start, os$End[id])
    } else {
      start = c(start, os$Start[id])
    }
  }
}
ma = cbind(ma, start)
ma = ma[!(is.na(ma$Alt)), ]

write.table(ma, 'SNV.Freq.txt', sep="\t")

###########################Make graphs#################################

library(reshape2)

#make a key for the gene length
xlims = matrix(c(
  name,as.numeric(reg1),as.numeric(reg2)), nrow = 1, ncol = 3)
xlimsm = as.data.frame(xlims)
colnames(xlimsm) = c('gene', 'from', 'to')
xlimsm$from = as.numeric(as.character(xlimsm$from))
xlimsm$to = as.numeric(as.character(xlimsm$to))

forg = cbind(ma$Alt, 
             unlist(lapply(1:nrow(ma), function(x) sum(ma$GOF[x], 
                                                       ma$LOF[x], 
                                                       ma$Unknown[x]))), 
             ma$func2, ma$start)
colnames(forg) = c('Alteration', "Frequency", "Function", 'Start')
forg = as.data.frame(forg)
forg$Start = as.numeric(as.character(forg$Start))
forg$Frequency = as.numeric(as.character(forg$Frequency))
forg = forg[forg$Frequency > 0, ]


library(ggplot2)
library(gggenes)
library(cowplot)
library(gridExtra)

#load the gene information from exon list
subgenes = read.table('/fs/project/PAS0854/Raven/MSI/MSI_se_scratch/MSI.H.tables/Exonic_MSI/exon_list.csv', sep=",", header=T)

#store x axis start and end from the key
xsa = xlimsm[1, 2]
xso = xlimsm[1, 3]
subg = subset(subgenes, Name == name)
gp = ggplot(xlimsm, aes(xmin = from, xmax = to, y=gene)) +
  geom_gene_arrow(fill = 'white') + 
  geom_subgene_arrow(data = subg, 
                     aes(xmin = xsa, xmax = xso, 
                         y = Name, fill = Name,
                         xsubmin = Start, xsubmax = End), color='black', alpha=.7) +
  theme_genes() + 
  theme(legend.position = 'none')
#make the density plot
hiso = ggplot(forg, aes(x=Start, y=Alteration)) + 
  geom_point(aes(color = Function, size = Frequency), alpha = .5) + 
  xlim(as.numeric(xsa),as.numeric(xso)) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), 
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))  + 
  ylab('') +
  ggtitle('SNV Map')
hiso2 = ggplot(forg, aes(x=Start)) + 
  geom_density(aes(color = Function), alpha = 0) + 
  xlim(as.numeric(xsa),as.numeric(xso)) +
  theme(legend.position = 'none', axis.text.y = element_blank(), axis.ticks.y = element_blank(), 
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))

pdf(paste0('./SNV.pdf'))

print(plot_grid(hiso, hiso2, gp, align = "v", axis = "rl", nrow = 3, rel_heights = c(3/4, 1/8, 1/8)))

dev.off()
