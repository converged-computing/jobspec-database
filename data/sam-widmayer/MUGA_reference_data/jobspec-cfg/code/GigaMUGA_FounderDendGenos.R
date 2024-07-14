#!/usr/bin/env Rscript
require(dplyr)
require(fst)
require(tidyr)
require(stringr)
require(vroom)

args <- commandArgs(trailingOnly = TRUE)

###########################################
# Function designed to recode genotype calls from letter format (i.e. G1, HET, or G2) to numeric format (i.e. 0, 1, 2)
# Inputs:
# x = Column of genotype values
# Outputs:
# Numeric vector of recoded genotypes
###########################################
recodeCalls <- function(x){
  y <- factor(c(as.matrix(x)))
  levels(y)[which(levels(y) == "H")] <- 1
  levels(y)[which(levels(y) != 1)] <- c(0,2)
  return(as.numeric(as.character(y)))
}

writeWideChrGenos <- function(x,y){
  if(dir.exists(paths = "data/GigaMUGA/GigaMUGA_founder_sample_dendrogram_genos/")){
    fst::write.fst(x, path = paste0("data/GigaMUGA/GigaMUGA_founder_sample_dendrogram_genos/gm_widegenos_chr_",y,".fst"))
  } else {
    dir.create(path = "data/GigaMUGA/GigaMUGA_founder_sample_dendrogram_genos/")
    fst::write.fst(x, path = paste0("data/GigaMUGA/GigaMUGA_founder_sample_dendrogram_genos/gm_widegenos_chr_",y,".fst"))
  }
}

load("data/GigaMUGA/GigaMUGA_QC_Results.RData")
load("data/GigaMUGA/GigaMUGA_BadSamples_BadMarkers.RData")

gm_metadata <- vroom::vroom("data/GigaMUGA/gm_uwisc_v4.csv",
                            progress = T)
print(paste("Writing wide genotype file for chromosome",args[1]))


wide_chr_genos <- fst::read.fst(paste0("data/GigaMUGA/GigaMUGA_reference_genotypes/gm_genos_chr_",args[1],".fst")) %>%
            dplyr::select(sample_id, marker, genotype) %>%
            dplyr::mutate(marker_flag = dplyr::if_else(condition = marker %in% above.cutoff$marker, true = "FLAG", false = "")) %>%
            dplyr::filter(sample_id %in% founderSamples$sample_id, marker_flag == "") %>% 
            dplyr::select(-marker_flag) %>%
            tidyr::pivot_wider(names_from = marker, values_from = genotype)

recoded_wide_sample_genos <- suppressWarnings(data.frame(apply(wide_chr_genos[,2:ncol(wide_chr_genos)], 2, recodeCalls)))
recoded_wide_sample_genos$sample_id <- wide_chr_genos$sample_id

writeWideChrGenos(x = recoded_wide_sample_genos, y = args[1])
