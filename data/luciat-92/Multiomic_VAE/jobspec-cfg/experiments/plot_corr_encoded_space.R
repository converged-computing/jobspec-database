#library(umap)
Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 2)
library(ggplot2)
library(RColorBrewer)
library(dplyr)
library(tidyverse)
library(ggpubr)
library(argparse)
library(pheatmap)

# load input with args
parser <- ArgumentParser(description = "")
parser$add_argument("--folder_model", type = "character", help = "Input and Output fold")
parser$add_argument("--depmap_meta_file", type = "character", help = "file depmap info")
parser$add_argument("--private_enc", type = "logical", default = FALSE, help = "if the encoding is private")
parser$add_argument("--input_data_file", type = "character", nargs = "*", help = "file omics data")
args <- parser$parse_args()
print(args)

folder_model <- args$folder_model
depmap_meta_file <- args$depmap_meta_file
private_enc <- args$private_enc
input_data_file <- args$input_data_file

###########
#folder_model <- "/Volumes/iorio/lucia/Multiomic_VAE/experiments/experiment_3/mvae_gan/samples_tcgaonly_nfeat_var5000_norm_feat_flag_False_norm_type_zscore_only_shared_False_beta_0.0005/"
#depmap_meta_file <- "/Volumes/iorio/lucia/datasets/DEPMAP_PORTAL/version_23Q2/Model.csv"
#private_enc = FALSE
# input_data_file <- c("/Volumes/iorio/lucia/Multiomic_VAE/data/preprocessed/gene_expression_var1000_tcgaonly.csv.gz", "/Volumes/iorio/lucia/Multiomic_VAE/data/preprocessed/methylation_var1000_tcgaonly.csv.gz")
###########

print("Start")
# read a gz file
input_data <- lapply(input_data_file, function(x) readr::read_csv(gzfile(x)))
name_data <- c("GeneExpr", "Methylation")

depmap_sample_df <- readr::read_csv(depmap_meta_file)
if(private_enc){
  umap_df <- readr::read_csv(sprintf("%s/plots/private/umap.csv", folder_model))
  enc_df <- readr::read_csv(sprintf("%s/private_encoded_features.csv", folder_model))
  fold_plot_output <- sprintf("%s/plots/private/", folder_model)
} else {
  umap_df <- readr::read_csv(sprintf("%s/plots/umap.csv", folder_model))
  enc_df <- readr::read_csv(sprintf("%s/encoded_features.csv", folder_model))
  fold_plot_output <- sprintf("%s/plots/", folder_model)
}

input_mat <- list()
for(i in 1:length(input_data)){
  samples_id_data <- input_data[[i]]$sample_id
  input_mat[[i]] <- as.matrix(input_data[[i]][, -1])
  rownames(input_mat[[i]]) <- input_data[[i]]$sample_id
  input_mat[[i]] <- input_mat[[i]][enc_df$sample_id, ]
}

# refactor type, first xena then depmap
umap_df <- umap_df %>% 
  mutate(type = factor(type, levels = c("xena", "depmap"))) %>%
  arrange(type) %>%
  mutate(study = ifelse(study == "CL_depmap", "Cell Line - DepMap", paste0("Tissue - ", study))) %>%
  filter(!is.na(site)) %>%
  rename(sample_type = !!("_sample_type"))

# compute correlation with 
enc_mat <- as.matrix(enc_df[, -1])
rownames(enc_mat) <- enc_df$sample_id
colnames(enc_mat) <- paste0("Latent", 1:ncol(enc_mat))

cor_mat <- list()
for(i in 1:length(input_mat)){
  cor_mat[[i]] <- matrix(NA, ncol = ncol(enc_mat), nrow = ncol(input_mat[[i]]), 
                         dimnames = list(colnames(input_mat[[i]]), colnames(enc_mat)))
  for(j in 1:ncol(enc_mat)){
    print(j)
    cor_mat[[i]][, j] <- unname(apply(input_mat[[i]], 2, function(x) cor(x,enc_mat[, j], method = "spearman")))
  }  
}
# save correlation
for(i in 1:length(input_mat)){
  write.csv(cor_mat[[i]], sprintf("%s/correlation_%sVSlatent.csv", fold_plot_output, name_data[i]))
}

for(i in 1:length(input_mat)){
  pheatmap(t(cor_mat[[i]]), 
           cluster_rows = TRUE, 
           cluster_cols = TRUE, 
           show_rownames = TRUE, 
           show_colnames = FALSE, 
           filename = sprintf("%s/correlation_heatmap_%sVSlatent.png", fold_plot_output, name_data[i]))
  
}

# pheatmap of enc_mat 
df_ann <- umap_df %>% 
  select(sample_id, study, site, sample_type) %>%
  column_to_rownames(var = "sample_id")
df_ann <- df_ann[rownames(enc_mat), ]

n_colors <- length(unique(df_ann$site))
all_c <- colors()
set.seed(12344)
colors <- sample(all_c[!grepl("grey", all_c) & !grepl("gray", all_c)], n_colors)
names(colors) <- unique(df_ann$site)

a = t(scale(enc_mat))
a[a < -4] <- -4
a[a > 4] <- 4
pheatmap(a, 
         # add annotation on columns
         annotation_col = df_ann,
         # change color annotation
        annotation_colors = list(study = c("Cell Line - DepMap" = "#e8702a",  "Tissue - TCGA" = "#0c457d"),
                                 sample_type = c("Primary Tumor" = "red",
                                                 "Normal Tissue" = "blue",
                                                 "Metastatic" = "darkgreen",
                                                 "Recurrent Tumor" = "violet",
                                                 "Primary Blood Derived Cancer" = "orange"),
                                 site = colors),
         cluster_rows = TRUE, 
         cluster_cols = TRUE, 
         show_rownames = TRUE, 
         show_colnames = FALSE)



