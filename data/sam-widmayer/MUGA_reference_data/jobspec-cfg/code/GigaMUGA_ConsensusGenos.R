#!/usr/bin/env Rscript
require(dplyr)
require(fst)
require(tidyr)
require(stringr)
require(parallel)
require(data.table)
require(purrr)
require(furrr)
require(magrittr)
require(vroom)
args <- commandArgs(trailingOnly = TRUE)

###########################################
# Function to form F1 genotypes from founder consensus genotypes
# Inputs:
# x = Row of genotype values
# Outputs:
# Numeric vector of single letter genotypes
###########################################
callGeno <- function(x){
  
  # Compare genotypes from both strains
  if(c(x[[2]] == x[[3]])){
    #If they are the same, keep the first value
    predicted.geno <- x[[2]]
  } else {
    #If they are different, code as a het
    predicted.geno <- "H"
  }
  return(predicted.geno)
}

###########################################
# Function to form F1 genotypes from founder consensus genotypes
# Inputs:
# x = Row of genotype values
# Outputs:
# Numeric vector of single letter genotypes
###########################################
callHemiGeno <- function(x){
  # Mitochondria comes from dam
  predicted.geno <- x[[2]]
  return(predicted.geno)
}

###########################################
# Function to form reference sample genotypes for comparison
# Inputs:
# Consensus genotypes for dam strain
# Consensus genotypes for sire strain

# Outputs:
# Numeric vector of recoded genotypes
###########################################
founder_background_QC <- function(dam, sire, GC_15 = F){
  
  # Extract strain names from genotype objects
  dam.df <- data.frame(dam)
  sire.df <- data.frame(sire)
  # dam_strain <- gsub(colnames(dam.df)[2], pattern = "X", replacement = "")
  # sire_strain <- gsub(colnames(sire.df)[2], pattern = "X", replacement = "")
  
  # Identify samples from supplied strains
  cross_bg <- paste0(colnames(dam)[2], colnames(sire)[2])
  
  if(GC_15 == F){
    maternalF1s <- founderSamples %>%
      dplyr::mutate(bg = stringr::str_sub(sample_id, start = 0, end = 2)) %>%
      dplyr::filter(bg == cross_bg)
    
  } else {
    maternalF1s <- founderSamples_GC_15_F %>%
      dplyr::mutate(bg = "EC") %>%
      dplyr::filter(bg == "EC")
  }
  
  print(paste0("Running QC: ", cross_bg))
  
  if(nrow(maternalF1s) == 0){
    # Some crosses don't exist in reference data, can't be QC'd
    return("No samples from this cross; skipping")
  } else {
    
    # Remove missing consensus genotypes for each strain
    mom <- dam.df[which(!dam.df[,2] == "-"),]
    dad <- sire.df[which(!sire.df[,2] == "-"),]
    
    # Identify informative markers common to both parental strains
    common_markers <- intersect(mom$marker, dad$marker)
    
    # Form the F1 hybrid
    cross <- dplyr::inner_join(mom, dad, "marker") %>%
      dplyr::filter(marker %in% common_markers)
    
    # Predict F1 genotypes from consensus genotypes for each strain
    predicted.genotypes <- apply(cross, 1, FUN = callGeno)
    cross$predicted_genotypes <- predicted.genotypes
  
    # Gather genotypes for F1 samples and remove bad markers
    sample_genos <- maternalF1s %>%
      dplyr::right_join(good_chr_genos,.,by = "sample_id") %>%
      dplyr::filter(marker %in% common_markers,
                    genotype != "N") %>%
      dplyr::distinct(marker, sample_id, genotype, .keep_all = TRUE)
    
    # Nest genotypes by sample and sex to prepare for calling Chr X genotypes
    nested_samples <- sample_genos %>%
      dplyr::group_by(sample_id, inferred.sex) %>%
      tidyr::nest()
    
    final_geno_comp_list <- list()
    for(i in 1:length(nested_samples$sample_id)){
      
      if(args[1] == "X" & nested_samples$inferred.sex[[i]] == "m"){
        print("Detected Male X Chromosome")
        # Male X Chrs
        new_predicted_genotypes <- apply(cross, 1, callHemiGeno)
        new_cross <- cross %>%
          dplyr::mutate(predicted_genotypes = new_predicted_genotypes)
          
          final_geno_comp_list[[i]] <- nested_samples$data[[i]] %>%
            dplyr::inner_join(., new_cross %>% 
                                dplyr::select(marker, predicted_genotypes), 
                              by = "marker") %>%
            dplyr::mutate(matching_genos = dplyr::if_else(genotype == predicted_genotypes,
                                                          true = "MATCH",
                                                          false = "NO MATCH"),
                          alt_chr = dplyr::case_when(chr == "M" ~ "M",
                                                     chr == "X" ~ "X",
                                                     chr == "Y" ~ "Y",
                                                     is.na(chr) ~ "Other",
                                                     TRUE ~ "Autosome"),
                          alt_chr = as.factor(alt_chr),
                          sample = nested_samples$sample_id[[i]],
                          inferred.sex = nested_samples$inferred.sex[[i]])
          } else {
        # Autosome
            print("Detected Female X Chromosome or Autosome")
            final_geno_comp_list[[i]] <- nested_samples$data[[i]] %>%
          dplyr::inner_join(., cross %>% 
                              dplyr::select(marker, predicted_genotypes), 
                            by = "marker") %>% 
          dplyr::mutate(matching_genos = dplyr::if_else(genotype == predicted_genotypes,
                                                        true = "MATCH",
                                                        false = "NO MATCH"),
                        alt_chr = dplyr::case_when(chr == "M" ~ "M",
                                                   chr == "X" ~ "X",
                                                   chr == "Y" ~ "Y",
                                                   is.na(chr) ~ "Other",
                                                   TRUE ~ "Autosome"),
                        alt_chr = as.factor(alt_chr),
                        sample = nested_samples$sample_id[[i]],
                        inferred.sex = nested_samples$inferred.sex[[i]])
        }
      }
    final_geno_comp <- Reduce(rbind,final_geno_comp_list)
    
    
    # Tabulate the percentage of concordant genotypes between theoretical and actual F1 samples
    geno_comp_summary <- final_geno_comp %>%
      dplyr::group_by(sample, inferred.sex, alt_chr, matching_genos) %>%
      dplyr::count() %>%
      tidyr::pivot_wider(names_from = matching_genos, values_from = n) %>%
      dplyr::mutate(bg = cross_bg,
                    chr = unique(good_chr_genos$chr))
    
    # Return genotypes and their summary statistics
    return(list(final_geno_comp,geno_comp_summary))
    }
}
# founder_background_QC(dam = dam_calls[[1]], 
#                       sire = sire_calls[[1]])



load("data/GigaMUGA/GigaMUGA_QC_Results.RData")
load("data/GigaMUGA/GigaMUGA_BadSamples_BadMarkers.RData")
load("data/GigaMUGA/GigaMUGA_SexCheck_Results.RData")
gm_metadata <- vroom::vroom("data/GigaMUGA/gm_uwisc_v4.csv",
                            progress = T)

# Sample Genotype Filepath
geno_path <- c("data/GigaMUGA/GigaMUGA_reference_genotypes")

# Consensus Genotype Filepath
consensus_geno_path <- c("data/GigaMUGA/GigaMUGA_founder_consensus_genotypes")

# Read in sample genotypes for a given chromosome and 
# filter out bad markers from sex check and marker QC.
# Using *ALL* reference samples, not just CC/DO founders and their F1s
good_chr_genos <- fst::read.fst(path = file.path(geno_path, 
                                                 paste0("gm_genos_chr_",args[1],".fst"))) %>%
  dplyr::mutate(marker_flag = dplyr::if_else(condition = marker %in% above.cutoff$marker,
                                             true = "FLAG",
                                             false = ""),
                genotype = dplyr::if_else(condition = genotype == "-",
                                          true = "N",
                                          false = genotype)) %>%
  dplyr::filter(marker_flag == "") %>%
  dplyr::select(-marker_flag)


# Read in consensus genotypes created from DO sample imputation
consensus_genotypes <- vroom::vroom(file = file.path(consensus_geno_path,
                                                     paste0("GigaMUGA_founder_consensus_imputed_genotypes_chr",
                                                            args[1],".csv")), delim = ",")
colnames(consensus_genotypes) <- c("marker",LETTERS[1:8])

founder_letters <- LETTERS[1:8]

# First form a list of dams that comprise each F1 cross type
parents <- data.frame(tidyr::expand_grid(founder_letters, founder_letters, .name_repair = "minimal")) %>%
  `colnames<-`(c("dams","sires"))

# Select dams
dams <- parents %>%
  dplyr::select(dams) %>%
  as.list()

# Select sires
sires <- parents %>%
  dplyr::select(sires) %>%
  as.list()

# Pull in consensus genotypes for each parental strain as a big list of objects
condenseFounderGenos <- function(x){
  # Identify founder in consensus file
  founder_consensus <- consensus_genotypes[,c("marker",x)]
  
  # Identify homozygous sites in consensus for the strain of interest
  fc_string <- strsplit(c(founder_consensus[,2])[[1]],"")
  homo_sites <- unlist(lapply(fc_string, function(x){x[1]==x[2]}))
  
  # Make founder strain genos the single letter genos
  single_letter_genos <- unlist(purrr::transpose(fc_string[which(homo_sites)])[[1]])
  founder_consensus[which(homo_sites),2] <- single_letter_genos
  
  return(founder_consensus)
}

dam_calls <- purrr::map(dams$dams,
                        condenseFounderGenos)
sire_calls <- purrr::map(sires$sires,
                         condenseFounderGenos)


# Compare the predicted genotypes (from consensus calls) to the actual genotypes of each sample in parallel
plan(multisession, workers = 16)
make_chunks <- furrr:::make_chunks
make_chunks(n_x = length(dam_calls), n_workers = 16)
bg_QC <- furrr::future_map2(dam_calls,
                            sire_calls, 
                            founder_background_QC,
                            .options = furrr_options(seed = TRUE))

# Keep outputs from the QC that are lists; if QC wasn't performed for a given background, the output was a character vector warning
founder_background_QC_tr <- bg_QC %>%
  purrr::keep(., is.list) %>%
  # Instead of having 64 elements of lists of two, have two lists of 64:
  # 1) All good genotypes from each cross with concordance values
  # 2) All concordance summaries for each cross
  purrr::transpose(.)

# Bind together all concordance summaries
founder_concordance_df <- Reduce(rbind, founder_background_QC_tr[[2]])

# If all markers for a given chromosome type were either concordant or discordant, NAs are returned
# This step assigns those NA values a 0
founder_concordance_df[is.na(founder_concordance_df)] <- 0

# Form concordance as a percentage
founder_concordance_df_2 <- founder_concordance_df %>%
  # dplyr::mutate(dam = gsub(dam, pattern = "[.]", replacement = "/"),
  #               sire = gsub(sire, pattern = "[.]", replacement = "/")) %>%
  dplyr::select(sample, inferred.sex, alt_chr, bg, chr, MATCH, `NO MATCH`)
founder_concordance_df_2$alt_chr <- factor(founder_concordance_df_2$alt_chr,
                                           levels = c("Autosome","X","Y","M","Other"))

# Performing background QC of GC_15 as the proper strain
founderSamples_GC_15_F <- founderSamples %>%
  dplyr::filter(sample_id == "GC_15_F")
GC_15_F_results <- founder_background_QC(dam = dam_calls[[33]],  # NZO
                                         sire = sire_calls[[3]], #129S1
                                         GC_15 = T) 


# Writing Concordance Files
if(dir.exists(paths = "data/GigaMUGA/GigaMUGA_founder_sample_concordance/")){
  fst::write.fst(founder_concordance_df_2, 
                 path = paste0("data/GigaMUGA/GigaMUGA_founder_sample_concordance/gm_founder_concordance_",args[1],".fst"))
} else {
  dir.create(path = "data/GigaMUGA/GigaMUGA_founder_sample_concordance/")
  fst::write.fst(founder_concordance_df_2, 
                 path = paste0("data/GigaMUGA/GigaMUGA_founder_sample_concordance/gm_founder_concordance_",args[1],".fst"))
}

# Write Results for Resexing of GC_15 to an NZO/129S1 mouse
if(dir.exists(paths = "data/GigaMUGA/GC_15_F_ReSex_Results")){
  write.csv(GC_15_F_results[[2]], file = paste0("data/GigaMUGA/GC_15_F_ReSex_Results/GC_15_F_",args[1],".csv"), row.names = F)
} else {
  dir.create(path = "data/GigaMUGA/GC_15_F_ReSex_Results/")
  write.csv(GC_15_F_results[[2]], file = paste0("data/GigaMUGA/GC_15_F_ReSex_Results/GC_15_F_",args[1],".csv"), row.names = F)
}


