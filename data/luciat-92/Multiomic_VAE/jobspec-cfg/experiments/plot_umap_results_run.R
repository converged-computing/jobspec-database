#library(umap)
library(ggplot2)
library(RColorBrewer)
library(dplyr)
library(tidyverse)
library(ggpubr)
library(argparse)

# load input with args
parser <- ArgumentParser(description = "")
parser$add_argument("--folder_model", type = "character", help = "Input and Output fold")
parser$add_argument("--depmap_meta_file", type = "character", help = "file depmap info")
parser$add_argument("--file_purity", type = "character", help = "file with tumour purity")
parser$add_argument("--private_enc", type = "logical", default = FALSE, help = "if the encoding is private")
args <- parser$parse_args()
print(args)

folder_model <- args$folder_model
depmap_meta_file <- args$depmap_meta_file
file_purity <- args$file_purity
private_enc <- args$private_enc

###########
#folder_model <- "/Volumes/iorio/lucia/Multiomic_VAE/experiments/experiment_2/vae_gan/samples__ngene_var5000_norm_feat_flag_False_only_shared_True_beta_0.001/"
#depmap_meta_file <- "/Volumes/iorio/lucia/datasets/DEPMAP_PORTAL/version_23Q2/Model.csv"
#file_purity = "/Volumes/iorio/lucia/Multiomic_VAE/data/raw/41467_2015_BFncomms9971_MOESM1236_ESM.xlsx"
#private_enc = FALSE
###########

print("Start")

# read data
df_sheet <- readxl::read_xlsx(file_purity, sheet="Supp Data 1", skip=3) %>% 
  rename_all(tolower) %>% 
  rename_all(~ gsub(" ", "_", .)) %>%
  mutate(estimate = as.numeric(estimate)) %>%
  mutate(sample_id = gsub(".$", "", sample_id)) # remove last charachter!

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

# refactor type, first xena then depmap
umap_df <- umap_df %>% 
  mutate(type = factor(type, levels = c("xena", "depmap"))) %>%
  arrange(type) %>%
  mutate(study = ifelse(study == "CL_depmap", "Cell Line - DepMap", paste0("Tissue - ", study))) %>%
  filter(!is.na(site)) %>%
  rename(sample_type = !!("_sample_type"))
# merge with tumour purity
umap_df <- umap_df %>% 
  left_join(df_sheet %>% select(sample_id, estimate), by = "sample_id")

pl1 <- ggplot(subset(umap_df), 
             aes(x = umap_1, y = umap_2, color = study, size = type)) + 
  geom_point(alpha = 0.5) +
  scale_size_manual(values = c(0.2, 1)) +
  scale_color_manual(values = c("#e8702a", "#6bd2db", "#9ed670", "#0c457d")) +
  theme_classic() + 
  theme(legend.title = element_blank(),
        legend.position = c(0.001, .001),
        legend.justification = c("left", "bottom"),
        legend.box.just = "left",
        legend.margin = margin(6, 6, 6, 6), 
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12), 
        legend.text = element_text(size = 10)) +
  xlab("UMAP 1") +
  ylab("UMAP 2") +
  guides(size = 'none')
# pl1
pl1
ggsave(sprintf("%s/umap_study_ggplot2.png", fold_plot_output), pl1, width = 6, height = 6)

# select random colors
n_colors <- length(unique(umap_df$site))
all_c <- colors()
set.seed(12344)
colors <- sample(all_c[!grepl("grey", all_c) & !grepl("gray", all_c)], n_colors)

pl2 <- ggplot(subset(umap_df), 
             aes(x = umap_1, y = umap_2, size = type)) + 
  geom_point(aes(fill = site, color = type), shape = 21, alpha = 0.8) + 
  scale_size_manual(values = c(0.7, 1)) +
  scale_color_manual(values = c("transparent", "black")) +
  # scale_fill_manual(values = colors) +
  theme_classic() + 
  theme(legend.title = element_blank(),
        legend.position = "right",
        legend.text = element_text(size = 10), 
        # reduce legend spacing
        legend.key.height = unit(0.1, "cm"),  
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12)) +
  xlab("UMAP 1") +
  ylab("UMAP 2") + 
  guides(fill = guide_legend(ncol = 1), size = 'none', color = 'none')
pl2
# pl2
ggsave(sprintf("%s/umap_site_ggplot2.png", fold_plot_output), pl2, width = 8, height = 6.2)

######################################
##### PLOTS FOR OT PRESENTATION ######
# #####################################
# xena_info <- read.table("/Volumes/iorio/lucia/datasets/XENA/TCGA_TARGET_GTEx/TcgaTargetGTEX_phenotype.txt.gz", sep = "\t", h=T, stringsAsFactors = F)
# umap_df_tot <- umap_df %>% 
#   left_join(xena_info %>% select(sample, detailed_category), by = c("sample_id" = "sample")) %>%
#   left_join(depmap_sample_df, by = c("sample_id" = "ModelID")) %>%
#   mutate(detailed_category = case_when(
#     type == "xena" ~ detailed_category,
#     type != "xena" ~ OncotreeLineage
#   )) %>%
#   rename(detailed_category_old = detailed_category) %>%
#   mutate(detailed_category = case_when(
#     detailed_category_old == "Cells - Ebv-Transformed Lymphocytes" ~ "Cells - Ebv-Transformed\nLymphocytes",
#     grepl("Brain - ", detailed_category_old) ~ "Brain",
#     TRUE ~ detailed_category_old
#   ))
# 
# pl2_ex1 <- ggplot(subset(umap_df_tot, umap_2 < -0.5 & umap_2 > -2.5 & umap_1 < 2.1), 
#               aes(x = umap_1, y = umap_2, size = type)) + 
#   geom_point(aes(fill = detailed_category, color = type), shape = 21, alpha = 0.8) + 
#   scale_size_manual(values = c(0.7, 1)) +
#   scale_color_manual(values = c("transparent", "black")) +
#   # scale_fill_manual(values = colors) +
#   theme_classic() + 
#   theme(legend.title = element_blank(),
#         legend.position = "right",
#         legend.text = element_text(size = 10), 
#         # reduce legend spacing
#         legend.key.height = unit(0.1, "cm"),  
#         axis.title = element_text(size = 14),
#         axis.text = element_text(size = 12)) +
#   xlab("UMAP 1") +
#   ylab("UMAP 2") + 
#   guides(fill = guide_legend(ncol = 1), size = 'none', color = 'none')
# pl2_ex1
# ggsave(sprintf("%s/umap_site_ggplot2_example1.png", fold_plot_output), pl2_ex1, width = 5, height = 4.2)
# 
# pl2_ex2 <- ggplot(subset(umap_df_tot, umap_2 > 9 & umap_2 < 10.5 & umap_1 > 0 & umap_1 < 2), 
#                   aes(x = umap_1, y = umap_2, size = type)) + 
#   geom_point(aes(fill = detailed_category, color = type), shape = 21, alpha = 0.8) + 
#   scale_size_manual(values = c(0.7, 1)) +
#   scale_color_manual(values = c("transparent", "black")) +
#   # scale_fill_manual(values = colors) +
#   theme_classic() + 
#   theme(legend.title = element_blank(),
#         legend.position = "right",
#         legend.text = element_text(size = 10), 
#         # reduce legend spacing
#         legend.key.height = unit(0.1, "cm"),  
#         axis.title = element_text(size = 14),
#         axis.text = element_text(size = 12)) +
#   xlab("UMAP 1") +
#   ylab("UMAP 2") + 
#   guides(fill = guide_legend(ncol = 1), size = 'none', color = 'none')
# pl2_ex2
# ggsave(sprintf("%s/umap_site_ggplot2_example2.png", fold_plot_output), pl2_ex2, width = 5, height = 4.2)
# 
# pl2_ex3 <- ggplot(subset(umap_df_tot, umap_2 > 9.6 & umap_2 < 9.9 & umap_1 > 2 & umap_1 < 4), 
#                   aes(x = umap_1, y = umap_2, size = type)) + 
#   geom_point(aes(fill = detailed_category, color = type), shape = 21, alpha = 0.8) + 
#   scale_size_manual(values = c(0.7, 1)) +
#   scale_color_manual(values = c("transparent", "black")) +
#   # scale_fill_manual(values = colors) +
#   theme_classic() + 
#   theme(legend.title = element_blank(),
#         legend.position = "right",
#         legend.text = element_text(size = 10), 
#         # reduce legend spacing
#         legend.key.height = unit(0.1, "cm"),  
#         axis.title = element_text(size = 14),
#         axis.text = element_text(size = 12)) +
#   xlab("UMAP 1") +
#   ylab("UMAP 2") + 
#   guides(fill = guide_legend(ncol = 1), size = 'none', color = 'none')
# pl2_ex3
# ggsave(sprintf("%s/umap_site_ggplot2_example3.png", fold_plot_output), pl2_ex3, width = 5, height = 4.2)
# 
# pl2_ex4 <- ggplot(subset(umap_df_tot, umap_2 > 10.5 & umap_1 > 4 & umap_1 < 9), 
#                   aes(x = umap_1, y = umap_2, size = type)) + 
#   geom_point(aes(fill = site, color = type), shape = 21, alpha = 0.8) + 
#   scale_size_manual(values = c(0.9, 1)) +
#   scale_color_manual(values = c("transparent", "black")) +
#   # scale_fill_manual(values = colors) +
#   theme_classic() + 
#   theme(legend.title = element_blank(),
#         legend.position = "right",
#         legend.text = element_text(size = 10), 
#         # reduce legend spacing
#         legend.key.height = unit(0.1, "cm"),  
#         axis.title = element_text(size = 14),
#         axis.text = element_text(size = 12)) +
#   xlab("UMAP 1") +
#   ylab("UMAP 2") + 
#   guides(fill = guide_legend(ncol = 1), size = 'none', color = 'none')
# pl2_ex4
# ggsave(sprintf("%s/umap_site_ggplot2_example4.png", fold_plot_output), pl2_ex4, width = 5, height = 4.2)
# 
# pl3_ex4 <- ggplot(subset(umap_df_tot, umap_2 > 10.5 & umap_1 > 4 & umap_1 < 9), 
#                   aes(x = umap_1, y = umap_2, size = type)) + 
#   geom_point(aes(fill = sample_type, color = type), shape = 21, alpha = 0.8) + 
#   scale_size_manual(values = c(0.9, 1)) +
#   scale_color_manual(values = c("transparent", "black")) +
#   # scale_fill_manual(values = colors) +
#   theme_classic() + 
#   theme(legend.title = element_blank(),
#         legend.position = "right",
#         legend.text = element_text(size = 10), 
#         # reduce legend spacing
#         legend.key.height = unit(0.1, "cm"),  
#         axis.title = element_text(size = 14),
#         axis.text = element_text(size = 12)) +
#   xlab("UMAP 1") +
#   ylab("UMAP 2") + 
#   guides(fill = guide_legend(ncol = 1), size = 'none', color = 'none')
# pl3_ex4
# ggsave(sprintf("%s/umap_sampletype_ggplot2_example4.png", fold_plot_output), pl3_ex4, width = 5, height = 4.2)
# 
# umap_df_tot <- umap_df_tot %>%
#   mutate(detailed_category_ex5 = case_when(
#     detailed_category %in% c("Brain", "Lung", "Brain Lower Grade Glioma", "CNS/Brain", "Glioblastoma Multiforme", "Lung") ~ detailed_category,
#     TRUE ~ "Other"
#   ))
# 
# pl2_ex5 <- ggplot(subset(umap_df_tot, umap_2 > 5 & umap_2 < 9 & umap_1 < -0.5), 
#                   aes(x = umap_1, y = umap_2, size = type)) + 
#   geom_point(aes(fill = detailed_category_ex5, color = type), shape = 21, alpha = 0.8) + 
#   scale_size_manual(values = c(0.9, 1)) +
#   scale_color_manual(values = c("transparent", "black")) +
#   # scale_fill_manual(values = colors) +
#   theme_classic() + 
#   theme(legend.title = element_blank(),
#         legend.position = "right",
#         legend.text = element_text(size = 10), 
#         # reduce legend spacing
#         legend.key.height = unit(0.1, "cm"),  
#         axis.title = element_text(size = 14),
#         axis.text = element_text(size = 12)) +
#   xlab("UMAP 1") +
#   ylab("UMAP 2") + 
#   guides(fill = guide_legend(ncol = 1), size = 'none', color = 'none')
# pl2_ex5
# ggsave(sprintf("%s/umap_site_ggplot2_example5.png", fold_plot_output), pl2_ex5, width = 6, height = 4.2)
# 
# pl3_ex5 <- ggplot(subset(umap_df_tot, umap_2 > 5 & umap_2 < 9 & umap_1 < -0.5), 
#                   aes(x = umap_1, y = umap_2, size = type)) + 
#   geom_point(aes(fill = sample_type, color = type), shape = 21, alpha = 0.8) + 
#   scale_size_manual(values = c(0.9, 1)) +
#   scale_color_manual(values = c("transparent", "black")) +
#   # scale_fill_manual(values = colors) +
#   theme_classic() + 
#   theme(legend.title = element_blank(),
#         legend.position = "right",
#         legend.text = element_text(size = 10), 
#         # reduce legend spacing
#         legend.key.height = unit(0.1, "cm"),  
#         axis.title = element_text(size = 14),
#         axis.text = element_text(size = 12)) +
#   xlab("UMAP 1") +
#   ylab("UMAP 2") + 
#   guides(fill = guide_legend(ncol = 1), size = 'none', color = 'none')
# pl3_ex5
# ggsave(sprintf("%s/umap_sampletype_ggplot2_example5.png", fold_plot_output), pl3_ex5, width = 5, height = 4.2)
# #############################################


pl3 <- ggplot(subset(umap_df, type == "xena"), 
              aes(x = umap_1, y = umap_2, size = type)) + 
  geom_point(aes(fill = sample_type, color = type), shape = 21, alpha = 0.8) + 
  scale_size_manual(values = c(0.7, 1)) +
  scale_color_manual(values = c("transparent", "black")) +
  # scale_fill_manual(values = colors) +
  theme_classic() + 
  theme(legend.title = element_blank(),
        legend.position = "right",
        legend.text = element_text(size = 10), 
        # reduce legend spacing
        legend.key.height = unit(0.1, "cm"),  
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12)) +
  xlab("UMAP 1") +
  ylab("UMAP 2") + 
  guides(fill = guide_legend(ncol = 1), size = 'none', color = 'none')
pl3
ggsave(sprintf("%s/umap_sampletype_ggplot2.png", fold_plot_output), pl3, width = 8, height = 6.2)

pl4 <- ggplot(subset(umap_df, !is.na(estimate)), 
              aes(x = umap_1, y = umap_2, color = estimate)) + 
  geom_point(alpha = 0.5, size = 0.3) + 
  # scale_fill_manual(values = colors) +
  scale_color_gradient2(low = "blue", mid = "grey80", high = "red", midpoint = mean(umap_df$estimate, na.rm = T))+
  theme_classic() + 
  theme(legend.position = "right",
        legend.text = element_text(size = 10), 
        # reduce legend spacing
        legend.key.height = unit(0.5, "cm"),  
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12)) +
  xlab("UMAP 1") +
  ylab("UMAP 2") + 
  # change legend title
  guides(color = guide_colorbar(title = "Tumour Purity"))
pl4
ggsave(sprintf("%s/umap_tumourpurity_ggplot2.png", fold_plot_output), pl4, width = 8, height = 6.2)
# ggarrange(plotlist = list(pl1, pl2), nrow = 1)

# # compute correlation among encoded features
# cor_enc_df <- cor(t(enc_df[, -1]))
# colnames(cor_enc_df) <- enc_df$sample_id
# rownames(cor_enc_df) <- enc_df$sample_id
# 
# sample_df <- umap_df %>%
#   dplyr::select(-umap_1, -umap_2) %>%
#   dplyr::rename(sample_type = !!("_sample_type")) %>%
#   dplyr::mutate(sample_type = case_when(
#     sample_type == "Solid Tissue Normal" ~ "Normal Tissue",
#     sample_type %in% c("Primary Solid Tumor", "Additional - New Primary") ~ "Primary Tumor",
#     sample_type == "Recurrent Solid Tumor" ~ "Recurrent Tumor",
#     sample_type == "Additional Metastatic" ~ "Metastatic",
#     sample_type %in% c("Primary Blood Derived Cancer - Peripheral Blood", "Primary Blood Derived Cancer - Bone Marrow") ~ "Primary Blood Derived Cancer",
#     sample_type %in% c("Recurrent Blood Derived Cancer - Bone Marrow", "Recurrent Blood Derived Cancer - Peripheral Blood") ~ "Recurrent Blood Derived Cancer",
#     sample_type %in% c("Post treatment Blood Cancer - Bone Marrow", "Post treatment Blood Cancer - Peripheral Blood", "Post treatment Blood Cancer - Blood") ~ "Post treatment Blood Cancer",
#     TRUE ~ sample_type
#   ))
# 
# 
# sample_df <- sample_df[match(colnames(cor_enc_df), sample_df$sample_id), ]
# id_depmap <- sample_df$type == "depmap"
# id_all_CL <- which(id_depmap)
# df <- list()
# corr_threshold <- 0.9
# for(i in 1:length(id_all_CL)){
#   
#   idx <- id_all_CL[i]
#   id_CL <- sample_df$sample_id[idx]
#   curr_corr <- cor_enc_df[id_CL, ]
#   
#   tmp_ann <- sample_df[curr_corr >= corr_threshold & !id_depmap, ]
#   max_study_perc <-  best_study <- max_sample_type_perc <- best_sample_type <- max_site_perc <- best_site <- NA
#   if(nrow(tmp_ann) > 0){
#     max_study_perc <- max(table(tmp_ann$study) / sum(table(tmp_ann$study)))
#     best_study <- paste0(names(table(tmp_ann$study)[table(tmp_ann$study) == max(table(tmp_ann$study))]), collapse = ",")
#     max_sample_type_perc <- max(table(tmp_ann$sample_type) / sum(table(tmp_ann$sample_type)))
#     best_sample_type <- paste0(names(table(tmp_ann$sample_type)[table(tmp_ann$sample_type) == max(table(tmp_ann$sample_type))]), collapse = ",")
#     max_site_perc <- max(table(tmp_ann$site) / sum(table(tmp_ann$site)))
#     best_site <- paste0(names(table(tmp_ann$site)[table(tmp_ann$site) == max(table(tmp_ann$site))]), collapse = ",")
#   }
#  
#   df[[i]] <- data.frame(sample_df[idx, ], 
#                         n_threshold = sum(curr_corr >= corr_threshold & !id_depmap),
#                         best_perc_study = max_study_perc, 
#                         best_study = best_study,
#                         best_perc_sample_type = max_sample_type_perc,
#                         best_sample_type = best_sample_type,
#                         best_perc_site = max_site_perc,
#                         best_site = best_site)
#   
# }
# 
# df <- do.call(rbind, df)
# # use color scale of reds
# # 
# png(sprintf("%s/heatmap_count_CL_site0.9corr.png", folder_model), width = 8, height = 10, units = "in", res = 300)
# pheatmap::pheatmap(t(table(df$site, df$best_site)), 
#                    cluster_rows = FALSE, 
#                    cluster_cols = FALSE, 
#                    color = c("grey80", colorRampPalette(brewer.pal(9, "Reds"))(100)))
# dev.off()
# 
# png(sprintf("%s/heatmap_count_CL_study0.9corr.png", folder_model), width = 5, height = 3, units = "in", res = 300)
# pheatmap::pheatmap(t(table(df$site, df$best_study)), 
#                    cluster_rows = FALSE, 
#                    cluster_cols = FALSE, 
#                    color = c("grey80", colorRampPalette(brewer.pal(9, "Reds"))(100)))
# # save
# dev.off()
# 
# png(sprintf("%s/heatmap_count_CL_sample_type0.9corr.png", folder_model), width = 5, height = 3, units = "in", res = 300)
# pheatmap::pheatmap(t(table(df$site, df$best_sample_type)), 
#                    cluster_rows = FALSE, 
#                    cluster_cols = FALSE, 
#                    fontsize = 7, 
#                    cellwidth = 6, 
#                    cellheight = 6, 
#                    color = c("grey80", colorRampPalette(brewer.pal(9, "Reds"))(100)))
# dev.off()

# compute euclidean distance
dist_enc <- as.matrix(dist(enc_df[, -1]))
colnames(dist_enc) <- enc_df$sample_id
rownames(dist_enc) <- enc_df$sample_id

sample_df <- umap_df %>%
  dplyr::select(-umap_1, -umap_2)
common_s <- intersect(sample_df$sample_id, enc_df$sample_id)
sample_df <- sample_df[match(common_s, sample_df$sample_id), ]

sample_df <- sample_df %>%
  dplyr::mutate(sample_type_macro = case_when(
    sample_type %in% c("Normal Tissue", "Cell Line") ~ "Non-cancerous",
    sample_type %in% c("Primary Tumor", "Recurrent Tumor", "Metastatic") ~ "Solid Tumor",
    sample_type %in% c("Primary Blood Derived Cancer", "Recurrent Blood Derived Cancer", "Post treatment Blood Cancer") ~ "Blood Tumor",
    TRUE ~ sample_type
  ))
id_depmap <- sample_df$type == "depmap"
id_all_CL <- which(id_depmap)

sample_df$sample_type_macro[id_depmap] <- "Solid Tumor"
sample_df$sample_type_macro[sample_df$sample_id %in% depmap_sample_df$ModelID[depmap_sample_df$OncotreePrimaryDisease == "Non-Cancerous"]] <- "Non-cancerous"
sample_df$sample_type_macro[id_depmap & sample_df$site == "Immune cells"] <- "Blood Tumor"

dist_enc <- dist_enc[common_s, common_s]

df_auc_site <- list()
df_auc_study <- list()
df_auc_type <- list()

for(i in 1:length(id_all_CL)){
  
  # print(i)
  idx <- id_all_CL[i]
  id_CL <- sample_df$sample_id[idx]
  curr_dist <- dist_enc[id_CL, ]
  
  # categorize based on study
  curr_s <- sample_df$study[idx]
  bin_study <- as.numeric(sample_df$study == curr_s)
  
  # compute AURC with pROC
  roc_obj <- pROC::roc(response = bin_study, predictor = curr_dist,
                       # arguments for ci
                       ci=TRUE, boot.n=1000, ci.alpha=0.95, stratified=TRUE, 
                       direction = ">", quiet = TRUE)
  ci <- roc_obj$ci
  # get auc and CI
  df_auc_study[[i]] <- data.frame(CI_low = as.numeric(ci)[1], 
                       AUC = as.numeric(ci)[2], 
                       CI_up = as.numeric(ci)[3], 
                       comp = "study", 
                       sample_id = id_CL, 
                       sample_info = sample_df$site[idx])
  
  # categorize based on site
  curr_t <- sample_df$site[idx]
  bin_site <- as.numeric(sample_df$site == curr_t)
  
  # compute AURC with pROC
  if(length(unique(bin_site[!id_depmap])) > 1){
    roc_obj <- pROC::roc(response = bin_site[!id_depmap], predictor = curr_dist[!id_depmap],
                         # arguments for ci
                         ci=TRUE, boot.n=1000, ci.alpha=0.95, stratified=TRUE, 
                         direction = ">", quiet = TRUE)
    ci <- roc_obj$ci
    # get auc and CI
    df_auc_site[[i]] <- data.frame(CI_low = as.numeric(ci)[1], 
                                   AUC = as.numeric(ci)[2], 
                                   CI_up = as.numeric(ci)[3], 
                                   comp = "site", 
                                   sample_id = id_CL, 
                                   sample_info = curr_t)
  }else{
    df_auc_site[[i]] <- data.frame(CI_low = NA, 
                                   AUC = NA, 
                                   CI_up = NA, 
                                   comp = "site", 
                                   sample_id = id_CL, 
                                   sample_info = curr_t)
  }
 
  # categorize based on type
  curr_tm <- sample_df$sample_type_macro[idx]
  bin_type <- as.numeric(sample_df$sample_type_macro == curr_tm)
  
  # compute AURC with pROC
  if(length(unique(bin_type[!id_depmap])) > 1){
    roc_obj <- pROC::roc(response = bin_type[!id_depmap], predictor = curr_dist[!id_depmap],
                         # arguments for ci
                         ci=TRUE, boot.n=1000, ci.alpha=0.95, stratified=TRUE, 
                         direction = ">", quiet = TRUE)
    ci <- roc_obj$ci
    # get auc and CI
    df_auc_type[[i]] <- data.frame(CI_low = as.numeric(ci)[1], 
                                   AUC = as.numeric(ci)[2], 
                                   CI_up = as.numeric(ci)[3], 
                                   comp = "type", 
                                   sample_id = id_CL, 
                                   sample_info = curr_t, 
                                   sample_type_macro = curr_tm)
  }else{
    df_auc_type[[i]] <- data.frame(CI_low = NA, 
                                   AUC = NA, 
                                   CI_up = NA, 
                                   comp = "type", 
                                   sample_id = id_CL, 
                                   sample_info = curr_t, 
                                   sample_type_macro = curr_tm)
  }
}
df_auc_site <- do.call(rbind, df_auc_site)
df_auc_study <- do.call(rbind, df_auc_study)
df_auc_type <- do.call(rbind, df_auc_type)

plot_auc_pertissue <- function(df_auc, title_pl, folder_model, save_csv = TRUE){
  
  # drop NA (not the same site name between tissue and cell lines)
  df_auc <- df_auc[!is.na(df_auc$AUC), ]
  # order class based on the median AUC in each sample info
  df_auc$sample_info <- factor(df_auc$sample_info, 
                               levels = names(sort(tapply(df_auc$AUC, df_auc$sample_info, median))))
  pl <- ggplot(df_auc, aes(x = sample_info, y = AUC, fill = sample_info)) + 
    geom_boxplot(outlier.shape = NA) + 
    geom_jitter(width = 0.2, alpha = 0.5, size = 0.7) + 
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", size = 1) +
    theme_bw() + 
    theme(axis.text.x = element_text(angle = 45, hjust = 1), 
          legend.position = "none") + 
    # facet_wrap(.~comp, scales = "free_x") + 
    labs(x = "", y = "AUCROC") + 
    ggtitle(title_pl)
  
  print(pl)
  ggsave(sprintf("%s/boxplotAUC_CLs_pertissue_%s.pdf", folder_model, gsub(" ", "_", title_pl)), 
         pl, width = 6, height = 4.5)
  if(save_csv){
    write.csv(df_auc, sprintf("%s/AUC_CLs_pertissue_%s.csv", folder_model, gsub(" ", "_", title_pl)), 
              row.names = F)
  }
  
  if( "sample_type_macro" %in% colnames(df_auc)){
    
  df_auc$sample_type_macro <- factor(df_auc$sample_type_macro, 
                                 levels = names(sort(tapply(df_auc$AUC, df_auc$sample_type_macro, median))))
   pl2 <- ggplot(df_auc, aes(x = sample_type_macro, y = AUC, fill = sample_type_macro)) + 
     geom_boxplot(outlier.shape = NA) + 
     geom_jitter(width = 0.2, alpha = 0.5, size = 0.7) + 
     geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", size = 1) +
     theme_bw() + 
     theme(axis.text.x = element_text(angle = 45, hjust = 1), 
           legend.position = "none") + 
     # facet_wrap(.~comp, scales = "free_x") + 
     labs(x = "", y = "AUCROC") + 
     ggtitle(title_pl)
   
   print(pl2)
   ggsave(sprintf("%s/boxplotAUC_CLs_pertype_%s.pdf", folder_model, gsub(" ", "_", title_pl)), 
          pl2, width =4.56, height = 4.5)
   if(save_csv){
     write.csv(df_auc, sprintf("%s/AUC_CLs_pertype_%s.csv", folder_model, gsub(" ", "_", title_pl)), 
               row.names = F)
   }
  }
  
}

plot_auc_pertissue(df_auc_site, "Same tissue", fold_plot_output)
plot_auc_pertissue(df_auc_study, "Same study", fold_plot_output)
plot_auc_pertissue(df_auc_type, "Same type", fold_plot_output)
