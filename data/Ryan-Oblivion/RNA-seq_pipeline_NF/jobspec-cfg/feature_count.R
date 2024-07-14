# running R commands to get feature counts of the bam files

# featureCounts is part of the Rsubread suite of tools
library(Rsubread)
library(ggplot2)
# now I want to loop through the bam files and use each bam in the feature counts tool
# first i want to create a txt file of all bam names and then use readlines to look through
# the names on each line in the bam txt file


# featureCounts also takes an annotation file, so I will use the same gtf file from 
# ther cut&run pipeline

# this gtf file works only for gene_id
#annotation_gtf_file = '../Homo_sapiens.GRCh38.109.chr.gtf.gz'

# try this gtf file for the gene name
#annotation_gtf_file = '../gencode.v19.annotation.gtf.gz'

#Try this get file which was used in the genome generate step for star
annotation_gtf_file = '../test_this.gtf'

bam_files = readLines(paste("./store_bam_files/merged_bams/", "merged_bams.txt", sep = ""))

for (x in bam_files) {

file_name = paste0("./store_bam_files/merged_bams/", x)
output_name = paste0(x, "_counts.txt")

fc = featureCounts( files = file_name,
annot.ext = annotation_gtf_file,
isGTFAnnotationFile = TRUE,
GTF.featureType = "exon",
GTF.attrType = "gene_name",
isPairedEnd = TRUE,
countReadPairs = TRUE
) 


write.table(fc$counts, file = output_name, sep = "\t", quote = FALSE)

}


library(tidyverse)

# a list of all the count files in this directory
files = list.files(path = ".", pattern = "*counts.txt")

total_files = length(files)
half_files = total_files / 2

list_names = list()

for (x in c(1:total_files)) { file_name = files[x]
var_name = paste0("x", gsub("-", "_", file_name)) 
assign(var_name, read.csv(files[x], sep = "\t", header = FALSE, skip = 1))

list_names = append(var_name, list_names)

}


# the names will be stored as strings. we dont want that so use get

df_list = lapply(list_names, get)


first_column = df_list[[1]][,1]

second_columns = lapply(df_list, function(df) df[, 2])

combined_df = do.call(cbind, c(list(first_column), second_columns))

rownames(combined_df) = combined_df[,1]

combined_df = combined_df[,-1]

colnames(combined_df) = list_names


# now we make the matrix 
cts = as.matrix(combined_df)
mode(cts) = "numeric"

coldata = data.frame( condition = c(rep("control", half_files), rep("knockdown", half_files)),
row.names = list_names)
#sample = c('c1','c1','c2','c2','c3','c3','kd1','kd1','kd2','kd2','kd3', 'kd3'),
#run = c('c1a','c1a','c2b','c2b','c3a','c3a','kd1a','kd1a','kd2b','kd2b','kd3a', 'kd3a'),
#row.names = list_names)

# this lets deseq2 know which are biological replicates (sample) and which are technical replicates (run)

coldata$condition = as.factor(coldata$condition)


library("DESeq2")

dds = DESeqDataSetFromMatrix( countData = cts, colData = coldata, design = ~ condition)

dds = DESeq(dds)

# collapseing the technical replicates
#dds <- collapseReplicates(dds, groupby = dds$sample, run = dds$run)

res = results(dds)

# making box plots for the replicates to see if any are unfit

par(mar=c(8,5,2,2))
boxplot(log10(assays(dds)[["cooks"]]), range=0, las=2)



png(file='box_plot_replicates.png', width=1000, height=1000)
boxplot(log10(assays(dds)[["cooks"]]), range=0, las=2)
dev.off()

# below I use lfcShrink for log fold change shrinkage for visulization and ranking
# helps with ranking of genes
# remember to site that we used apeglm as the shrinkage type. find in deseq2 package manuel

resLFC = lfcShrink(dds, coef="condition_knockdown_vs_control", type= "apeglm")


# now plot-ma of res then plot-ma of resLFC

ma_plot_res = plotMA(res, ylim = c(-4,4))
ma_plot_reslfc = plotMA(resLFC, ylim = c(-4,4))

ma_plot_res
ma_plot_reslfc

# we are eliminating a lot of noise when shrinking the lfc of our dds data, 
# as you can see in the second plot, while retaining the important genes
# the genes removed are low count genes, and we do this without needing to use other filtering thresholds

# now I want to order the results by pvalue. most significant to least significant

#resOrdered = res[order(res$pvalue),]

# do the same for the lfc also
#resLFCordered = resLFC[order(resLFC$pvalue),]

# if i want the adjusted pvalue to be set to 0.05, then I need to change alpha from defualt at 0.1

#res05 = results(dds, alpha = 0.05)


# plotting plotma png 
png(file = "ma_plot_reslfc.png", height = 1000, width = 1000)
ma_plot_reslfc = plotMA(resLFC, ylim = c(-4,4))
dev.off()

# now plotting the res data without the lfc shrink
png(file = "ma_plot_res.png", height = 1000, width = 1000)
ma_plot_res = plotMA(res, ylim = c(-4,4))
dev.off()

# now we take a subset of the genes that pass this value, then save as a table

#resLFC_0.05 = subset(resLFCordered, padj <= 0.05)

#write.table(as.data.frame(resLFC_0.05), file = "condition_kd_vs_ctr.csv")



# we can look at the individual gene counts across replicates and conditions

png(file = 'gene_counts_qa.png', height = 1000, width = 1000)
counts_select_gene = plotCounts(dds, gene=which.min(res$padj), intgroup="condition")
dev.off()


# here is where I want to plot and show which points represent which replicate it originates from 

gene_name = rownames(res[which.min(res$padj),])[1]
gene_name

data_counts = plotCounts(dds, gene=which.min(res$padj), intgroup="condition", returnData = TRUE)

data_counts$replicates = rownames(data_counts)

show_replicates = ggplot(data_counts, aes(x = condition, y = count, color = replicates)) + 
geom_point()  +
ggtitle(gene_name)

png(file = 'counts_with_replicates.png', height = 1000, width = 1000)
show_replicates
dev.off()


# just doing the above but in a for loop to get alot of gene counts

# ordering the dataframe by ascending order 

ordered_resLFC = resLFC[order(resLFC$padj),]
#ordered_resLFC

# getting only the adjusted p-values that are 0.05 or less

keep = which(ordered_resLFC$padj <= 0.05)

res_LFC_FDR = ordered_resLFC[keep,]

# now from the FDR list which genes pass the log2FoldChange threshold

keep_FC = which( res_LFC_FDR$log2FoldChange <= -1 | res_LFC_FDR$log2FoldChange >= 1)

res_LFC_FDR_FC = res_LFC_FDR[keep_FC, ]

res_LFC_FDR_FC

# getting only the top 20 genes with the lowest adjusted p-value and fold change threshold passed

top_20_padj_fc = head(res_LFC_FDR_FC, 20)


# now i want to take all the gene names and put it in a list to loop through it and create a gene count
# table for each

#library(os)

top_20_genes = row.names(top_20_padj_fc)


length(top_20_genes)


dir.create('gene_counts_with_reps')

for (x in c(1:length(top_20_genes))){
    data_counts = plotCounts(dds, gene=top_20_genes[x], intgroup="condition", returnData = TRUE)
    data_counts$replicates = rownames(data_counts)
    file_name = paste0(top_20_genes[x], '_counts_with_reps.png')
    file_path = paste0('gene_counts_with_reps/', file_name)
    png(file = file_path, width = 1000, height = 1000)
    print(ggplot(data_counts, aes( x = condition, y = count, color = replicates)) + 
    geom_point() +
    ggtitle(top_20_genes[x]))
    dev.off()
}


# this is similar to above but now I am getting the genes that did not pass the padj threshold and plotting their counts

# ordering the dataframe by ascending order 

ordered_resLFC = resLFC[order(resLFC$padj),]
#ordered_resLFC

# getting only the adjusted p-values that are 0.05 or greater. meaning they did not pass the padj

keep = which(ordered_resLFC$padj >= 0.05)

res_LFC_FDR_fail = ordered_resLFC[keep,]


# getting only the top 20 genes that did not pass the adjusted p-value.

top_50_padj = head(res_LFC_FDR_fail, 50)


# now i want to take all the gene names and put it in a list to loop through it and create a gene count
# table for each

#library(os)

top_50_genes = row.names(top_50_padj)


length(top_50_genes)

#plotCounts(dds, gene=top_20_genes[1], intgroup="condition", returnData = TRUE)
#ggplot(data_counts, aes( x = condition, y = count, color = replicates)) + 
#geom_point() +
#ggtitle(top_20_genes[1])

dir.create('gene_counts_failed_fdr')

for (x in c(1:length(top_50_genes))){
    data_counts = plotCounts(dds, gene=top_50_genes[x], intgroup="condition", returnData = TRUE)
    data_counts$replicates = rownames(data_counts)
    file_name = paste0(top_50_genes[x], '_counts_with_reps_failed.png')
    file_path = paste0('gene_counts_failed_fdr/', file_name)
    png(file = file_path, width = 1000, height = 1000)
    print(ggplot(data_counts, aes( x = condition, y = count, color = replicates)) + 
    geom_point() +
    ggtitle(top_50_genes[x]))
    dev.off()
    }



# now to show a pca of the different conditions to find any batch effects

rld <- rlog(dds, blind=FALSE)
pca_plot = plotPCA(rld, intgroup=c("condition"))

png(file = 'pca_analysis.png', height = 1000, width = 1000)
pca_plot 
dev.off()



library(EnhancedVolcano)

# trying pCutoff of 0.05 instead of 10e-5
# also using y = pvalue because y = padj doesnt get any genes when a cutoff of 10e-5 is set. NOT ANYMORE

v_plot_padj = EnhancedVolcano(resLFC, lab = rownames(resLFC), x = 'log2FoldChange', y = 'padj', pCutoff 
=0.05, FCcutoff = 0.5, title = "Genes Down/Up Regulated", subtitle = "Adjusted P-value vs shrunken LFC",
xlab = bquote(~Log[2] ~ "fold change"),
ylab = bquote(~Log[10] ~ "Padj-value"),
legendLabels = c("Not Signigicant", expression(Log[2] ~ FC), "padj passed", expression(p - adj ~ and ~ log[2] ~ FC ~ passed)))
png(file = 'v_plot_padj.png', height = 1000, width = 1000)
v_plot_padj
dev.off()


# relaxing the padj log2fold change from 0.5 to 0.2

v_plot_padj_fc_relaxed = EnhancedVolcano(resLFC, lab = rownames(resLFC), x = 'log2FoldChange', y = 'padj', pCutoff 
=0.05, FCcutoff = 0.2, title = "Genes Down/Up Regulated", subtitle = "Adjusted P-value vs shrunken LFC",
xlab = bquote(~Log[2] ~ "fold change"),
ylab = bquote(~Log[10] ~ "Padj-value"),
legendLabels = c("Not Signigicant", expression(Log[2] ~ FC), "padj passed", expression(p - adj ~ and ~ log[2] ~ FC ~ passed)))
png(file = 'v_plot_padj_fc_relaxed.png', height = 1000, width = 1000)
v_plot_padj
dev.off()

# making a volcano plot with relaxed threshold of 0.1

v_plot_padj_relaxed = EnhancedVolcano(resLFC, lab = rownames(resLFC), x = 'log2FoldChange', y = 'padj', pCutoff 
=0.1, FCcutoff = 0.5, title = "Genes Down/Up Regulated", subtitle = "Adjusted P-value vs shrunken LFC",
xlab = bquote(~Log[2] ~ "fold change"),
ylab = bquote(~Log[10] ~ "Padj-value"),
legendLabels = c("Not Signigicant", expression(Log[2] ~ FC), "padj passed", expression(p - adj ~ and ~ log[2] ~ FC ~ passed)))
png(file = 'v_plot_relaxed_padj.png', height = 1000, width = 1000)
v_plot_padj_relaxed
dev.off()

# lets make a v-plot with pvalue also

v_plot_pvalue = EnhancedVolcano(resLFC, lab = rownames(resLFC), x = 'log2FoldChange', y = 'pvalue', pCutoff
=0.05, FCcutoff = 0.5, title = "Genes Down/Up Regulated", subtitle = "P-value vs shrunken LFC",
xlab = bquote(~Log[2] ~ "fold change"),
ylab = bquote(~Log[10] ~ "P-value"),
legendLabels = c("Not Signigicant", expression(Log[2] ~ FC), "P-value passed", expression(p - value ~ and ~ log[2] ~ FC ~ passed)))

png(file = 'v_plot_pvalue.png', height = 1000, width = 1000)
v_plot_pvalue
dev.off()

# now we want to get all of the genes displayed in the Volcano plot into a chart

# first I find which rows have a padj value of less than or equal to 10e-4 
keep = which(resLFC$padj <= 0.05)


# then I only keep those rows
res_padj_LFC = resLFC[keep,]
#res_padj_LFC


# now I want to find which rows of this new data set has a LFC of >= 0.5 or <= -0.5
keep2 = which(res_padj_LFC$log2FoldChange >= 0.5 | res_padj_LFC$log2FoldChange <= -0.5 )

res_padj_LFC_final = res_padj_LFC[keep2,]

# This variable here contains the target genes displayed in the volcano plot with padj as its y-axis 
res_padj_LFC_final

# as you can see we have a total of 13 genes differentially expressed under the thresholds mentioned
length(res_padj_LFC_final[,1])

# now I will create a tsv file containing the selected genes

names_col = c('gene', 'baseMean', 'log2FoldChange', 'IfcSE', 'pvalue', 'padj')

df_target_genes = data.frame(res_padj_LFC_final)

new_df = cbind( genes = rownames(df_target_genes), df_target_genes)
rownames(new_df) = NULL
new_df
write.table( new_df, file = 'de_genes_lfc_shrunk_padj.tsv', sep = '\t', quote = FALSE, row.names = FALSE)


# next i want to just save the genes that pass a lower threshold of 0.1
keep_relaxed = which(resLFC$padj <= 0.1)
res_padj_relaxed = resLFC[keep_relaxed,]
keep_relaxed_2 = which(res_padj_relaxed$log2FoldChange >= 0.5 | res_padj_relaxed$log2FoldChange <= -0.5 )
res_padj_relaxed_final = res_padj_relaxed[keep_relaxed_2,]

names_col = c('gene', 'baseMean', 'log2FoldChange', 'IfcSE', 'pvalue', 'padj')
df_target_genes_relaxed = data.frame(res_padj_relaxed_final)
new_df_relaxed = cbind( genes = rownames(df_target_genes_relaxed), df_target_genes_relaxed)
rownames(new_df_relaxed) = NULL
write.table( new_df_relaxed, file = 'de_genes_relaxed_lfc_shrunk_padj.tsv', sep = '\t', quote = FALSE, row.names = FALSE)



# next i want to just save the genes that pass a lower threshold fold change of 0.2
keep_norm = which(resLFC$padj <= 0.05)
res_padj_norm = resLFC[keep_norm,]
keep_fc_relaxed = which(res_padj_norm$log2FoldChange >= 0.2 | res_padj_norm$log2FoldChange <= -0.2 )
res_padj_fc_relaxed = res_padj_norm[keep_fc_relaxed,]

names_col = c('gene', 'baseMean', 'log2FoldChange', 'IfcSE', 'pvalue', 'padj')
df_target_genes_fc_relaxed = data.frame(res_padj_fc_relaxed)
new_df_fc_relaxed = cbind( genes = rownames(df_target_genes_fc_relaxed), df_target_genes_fc_relaxed)
rownames(new_df_fc_relaxed) = NULL
write.table( new_df_fc_relaxed, file = 'de_genes_fc_relaxed_padj.tsv', sep = '\t', quote = FALSE, row.names = FALSE)