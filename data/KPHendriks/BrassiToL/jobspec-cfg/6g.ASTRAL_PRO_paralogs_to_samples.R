#A short script to collect paralog sample names from a .tre file created from IQ-TREE2 gene trees.

library(ape)
iqtrees<-read.tree(file = "6f.ASTRAL_PRO_input_gene_trees.tre")

#Collect all paralog sample names into a vector.
paralog_names<-NA
for (tree in 1:length(iqtrees)){
  paralog_names<-c(paralog_names, iqtrees[[tree]]$tip.label)
}

#Reduce vector to only unique values.
paralog_names<-sort(unique(paralog_names))

#Create a dataframe to match sample names sample names.
paralog_to_sample<-data.frame(paralog=paralog_names, sample=NA)
paralog_to_sample$sample<-sapply(strsplit(as.character(paralog_to_sample$paralog), "[.]"), '[', 1)

#Export the table as a txt file for use in ASTRAL-PRO.
write.table(paralog_to_sample, file = "6h.ASTRAL_PRO_paralogs_to_samples.txt", sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)

