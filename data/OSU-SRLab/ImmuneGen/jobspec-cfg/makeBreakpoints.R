args = commandArgs(trailingOnly = T)
name = args[1]
reg = args[2]
reg1 = strsplit(strsplit(reg, "-")[[1]][1], ":")[[1]][2]
reg2 = strsplit(reg, "-")[[1]][2]

setwd(paste0('/fs/ess/PAS0854/Active_projects/', name))

#load the filtered data frame
alltog = read.table('G.D.M.res.csv', row.names = NULL, header=T, sep=',')

#extract the starts and ends of the SVs
end = unlist(lapply(1:nrow(alltog), function(x) return(as.numeric(as.character(strsplit(strsplit(alltog$INFO[x], ";")[[1]][4], "=")[[1]][2])))))
start = as.numeric(as.character(alltog$POS))
chr = alltog$CHROM
#extract the types of the SVs
type = gsub("<", "", alltog$ALT)
type = gsub(">", "", type)
samp = alltog$Sample
#put the starts, ends, and types into a dataframe
tog = cbind(samp, type, start, end)
tog = as.data.frame(tog)

#make sure the starts and ends are numeric
tog$start = as.numeric(as.character(tog$start))
tog$end = as.numeric(as.character(tog$end))

library(reshape2)

#make a key for the gene length
xlims = matrix(c(
  name,as.numeric(reg1),as.numeric(reg2)), nrow = 1, ncol = 3)
xlimsm = as.data.frame(xlims)
colnames(xlimsm) = c('gene', 'from', 'to')
xlimsm$from = as.numeric(as.character(xlimsm$from))
xlimsm$to = as.numeric(as.character(xlimsm$to))
#only keep SVs that are inside the gene region

tog = tog[!(is.na(tog$start)), ]

out = c()
for (x in 1:nrow(tog)){
  if (tog$start[x] < xlimsm$from){
    #record whether the start is before the gene start
    out = c(out, x)
  } else if (tog$end[x] > xlimsm$to){
    #record whether the end is after the gene end
    out = c(out, x)
  }
}
#take out the samples that are in out
if (!(is.null(out))){
  tog = tog[-out, ] 
}
#melt
master = read.csv('Master.csv')
func = c()
for (x in 1:nrow(tog)){
  id = which(grepl(tog$samp[x], master$ID) == T)
  func = c(func, master$Function[id])
}
tog = cbind(tog, func)

togm = melt(tog, id.vars = c('samp', 'type', 'func'))

library(ggplot2)
library(gggenes)
library(cowplot)
library(gridExtra)

#load the gene information from exon list
subgenes = read.table('/fs/ess/PAS0854/Raven/MSI/MSI_se_scratch/MSI.H.tables/Exonic_MSI/exon_list.csv', sep=",", header=T)

#open a new pdf
pdf(paste0('./Breakpoints.pdf'))
pdf.options(width = 11, height = 9)
#store x axis start and end from the key
xsa = xlimsm[1, 2]
xso = xlimsm[1, 3]
#first make the segments
seg = ggplot(togm, aes(x=value, y=samp)) + geom_point(aes(color = func)) + 
  geom_segment(aes(x=start, xend = end, y=samp, yend=samp, color = func), data = tog) + 
  facet_wrap(~ type) +
  ggtitle(paste0(name," Breakpoints")) + 
  xlab('Position') + 
  ylab('Sample ID') + 
  xlim(as.numeric(xsa),as.numeric(xso)) + 
  theme(axis.text.y = element_blank())
subg = subset(subgenes, Name == name)
#make the gene diagram
gp = ggplot(xlimsm, aes(xmin = from, xmax = to, y=gene)) +
  geom_gene_arrow(fill = 'white') + 
  geom_subgene_arrow(data = subg, 
                     aes(xmin = xsa, xmax = xso, 
                         y = Name, fill = Name,
                         xsubmin = Start, xsubmax = End), color='black', alpha=.7) +
  theme_genes() +
  theme(legend.position = 'none')
#make the density plot
his = ggplot(togm, aes(x=value)) + geom_density(aes(fill = "red"), alpha = .5) + 
  xlim(as.numeric(xsa),as.numeric(xso)) +
  theme(legend.position = 'none')
#put the plots onto one page
print(plot_grid(seg, his, gp, align = "v", axis = 'lr', nrow = 3, rel_heights = c(3/4, 1/8, 1/8)))
#stop printing to the pdf
dev.off()
