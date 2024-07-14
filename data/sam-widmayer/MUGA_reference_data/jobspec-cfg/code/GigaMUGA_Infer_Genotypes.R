#!/usr/bin/env Rscript

################################################################################
# Using the full set of founder genotypes and the churchill lifespan data to
# infer missing founder genotypes.
# To do this, I need three data structures:
# 1. All founder & F1 data, including replicates.
# 2. Diplotype calls (from maxmarg) for a lot of samples.
# 3. 36 state genoprobs for the samples.

# Daniel Gatti
# dan.gatti@jax.org
# Sam Widmayer
# samuel.widmayer@jax.org

# 20240214
################################################################################

options(stringsAsFactors = FALSE)
args <- commandArgs(trailingOnly = TRUE)

library(dplyr)
library(tibble)
library(stringr)
library(tidyr)
library(fst)
library(qtl2)
library(qtl2convert)
library(qtl2fst)

##### VARIABLES #####

chr <- args[1]

if(chr == "X"){
  print("Specifying Input Files for Chromosome X")
} else {
  print(paste0("Specifying Input Files for Chromosome ",chr))
}


# Input data directory.
data_dir = '/projects/compsci/vmp/USERS/widmas/MUGA_reference_data/data/GigaMUGA'

# Sample data directory.
sample_dir = file.path(data_dir, 'DO_churchill_lifespan_1')

# Output directory for results.
out_dir  = '/projects/compsci/vmp/USERS/widmas/MUGA_reference_data/data/GigaMUGA'

# Marker file.
marker_file = file.path(data_dir, 'gm_uwisc_v2.csv')

# All founders & F1s in tall format.
if(chr == "X"){
  founder_all_file  = file.path(data_dir, 'GigaMUGA_reference_genotypes', 'gm_genos_chr_X.fst')
} else {
  founder_all_file  = file.path(data_dir, 'GigaMUGA_reference_genotypes', paste0('gm_genos_chr_',chr,'.fst'))
}


# Sample genotypes form DivDB.
sample_geno_file  = file.path(sample_dir, 'Churchill-churchill2-GigaMUGA_geno.csv')

# 36 state genoprobs.
sample_probs_file = file.path(sample_dir, 'churchill2__GigaMUGA_genoprobs_36state.rds')

# Sorted founder diplotype codes.
codes = outer(LETTERS[1:8], LETTERS[1:8], paste0)
codes = codes[upper.tri(codes, diag = TRUE)]

# Founder homozygous diplotypes.
founder_diplotypes = stringr::str_c(LETTERS[1:8], LETTERS[1:8])

# Homozygous genotypes.
homo_geno = c('AA', 'CC', 'GG', 'TT', '--')

##### MAIN #####

# Read in markers and create map.
markers = read.csv(marker_file) %>%
  mutate(pos = bp_mm10 * 1e-6) %>%
  as.data.frame()
map     = qtl2convert::map_df_to_list(map = markers, pos_column = 'pos')

# Read and wrangle founders & F1s.
# We want all of the allele codes sorted. (i.e. "AT", not "TA")
ff1 = fst::read_fst(founder_all_file)
ff1 = ff1 %>%
        select(sample_id, marker, chr, starts_with('allele')) %>%
        filter(str_detect(sample_id, str_c('^(', str_c(codes, collapse = '|'), ')_'))) %>%
        # filter(marker %in% names(map[[chr]])) %>%
        unite(gt, allele1, allele2, sep = '') %>%
        mutate(gt = if_else(gt == 'GC', 'CG', gt),
               gt = if_else(gt == 'TA', 'AT', gt),
               gt = if_else(gt == 'TC', 'CT', gt),
               gt = if_else(gt == 'TG', 'GT', gt)) %>%
        pivot_wider(names_from = sample_id, values_from = gt)
ff1 = ff1[ff1$marker %in% names(map[[chr]]),]

# We should have only markers from supplied chromosome
if(chr == "X"){
  stopifnot(all(ff1$chr == "X"))
} else {
  stopifnot(all(ff1$chr == chr))
}


ff1 = ff1 %>%
        select(-chr)
        
# Remove samples with too many missing calls.
nc = colMeans(ff1 == '--')

ff1 = ff1[,nc < 0.15]

# Create two letter founder codes for the founder/F1 samples.
# We can name two columns the same in a tibble.
founder_codes = str_sub(colnames(ff1)[-1], 1, 2)

# Convert to matrix to speed up operations.
ff1 = ff1 %>%
        column_to_rownames(var = 'marker') %>%
        as.matrix()

# Read in an wrangle sample genotypes.
sgeno = read.csv(sample_geno_file) 

# Convert to matrix to speed up operations.
sgeno = sgeno %>%
          column_to_rownames(var = 'marker') %>%
          as.matrix()
colnames(sgeno) <- gsub(colnames(sgeno), pattern = "[.]", replacement = "-")
sgeno[which(is.na(sgeno))] <- "-"

# Read in probs file and get diplotypes.
print("Reading Genotype Probabilities")
probs = readRDS(sample_probs_file)[[chr]]
rownames(probs) = str_replace_all(rownames(probs), '^Calico_Life_Sciences_Freund_MURGIGV01_[0-9]+_|_[A-H][0-9]+$', '')

# Synch up all of the datasets (samples and markers).
common_samples = intersect(rownames(probs), colnames(sgeno))
sgeno = sgeno[,common_samples]
probs = probs[common_samples,,]
stopifnot(all(colnames(sgeno) == rownames(probs)))

common_markers = intersect(names(map[[chr]]), rownames(ff1))
common_markers = intersect(common_markers,  rownames(sgeno))
common_markers = intersect(common_markers,  dimnames(probs)[[3]])

map[[chr]] = map[[chr]][names(map[[chr]]) %in% common_markers]
common_markers = names(map[[chr]])
ff1      = ff1[common_markers,]
sgeno    = sgeno[common_markers,]
probs    = probs[,,common_markers]

stopifnot(rownames(ff1) == rownames(sgeno))
stopifnot(rownames(ff1) == dimnames(probs)[[3]])

# Get the two-letter sample diplotypes.
sdiplo = maxmarg(list(probs), 
                 minprob = 0.01, 
                 return_char = TRUE)[[1]]

# Convert diplotypes to strings. Not sure why 'return_char = TRUE' didn't work.
sdiplo = matrix(colnames(probs)[sdiplo], 
                nrow = nrow(sdiplo), 
                ncol = ncol(sdiplo), 
                dimnames = list(rownames(probs), 
                                dimnames(probs)[[3]]))
sdiplo = t(sdiplo)



# Whew! Now we can begin the imputation.

# x: The founder/F1 matrix.
# cd: The corresponding diplotype codes.
consensus_geno = function(x, cd) {

  # Subset to keep founders only.
  wh = sapply(str_split(cd, pattern = ''), function(z) { z[1] == z[2] })
  cd = cd[wh]
  x  = x[,wh]
  
  # Using with_ties = FALSE, might lose some alleles.
  x = data.frame(x) %>%
        tibble::rownames_to_column(var = 'marker') %>%
        tidyr::pivot_longer(cols = -marker) %>%
        dplyr::mutate(name = str_sub(name, 1, 2)) %>%
        dplyr::count(marker, name, value) %>%
        group_by(marker, name) %>%
        slice_max(order_by = n, n = 1, with_ties = FALSE) %>%
        select(-n) %>%
        pivot_wider() %>%
        column_to_rownames(var = 'marker') %>%
        as.matrix()
        
  x[!x %in% homo_geno] = '--'
        
  return(x)
  
} # consensus_geno()

print("Imputing Consensus Genotypes")
founder_consensus = consensus_geno(x = ff1, 
                                   cd = founder_codes)
founder_consensus = founder_consensus[common_markers,]

# Get the rows containing missing data in the founder and at least
# 50% non-missing data in the samples.
wh = which(rowSums(founder_consensus == '--') > 0 & # FOUNDER consensus
           rowMeans(sgeno == '-') < 0.5) # DO samples

# Get allele frequencies and remove no-calls.
af = apply(sgeno[wh,], 1, table)
af = lapply(af, function(z) { z / sum(z) })
af = lapply(af, function(z) { z[names(z) != '-'] })

# Retain markers with three alleles and a minimum allele frequency of 1%.
# 1/36 is 2.8%.
wh = wh[sapply(af, length) == 3 & sapply(af, min) >= 0.01]

print(paste('Attempting to correct', length(wh), 'markers.'))

for(m in wh) {

  # Current founder consensus genotypes.
  fg = founder_consensus[m,]

  # Current sample diplotypes and genotypes.
  df = data.frame(id = colnames(sgeno), 
                  gt = sgeno[m,],   # Sample genotypes.
                  dt = sdiplo[m,])  # Sample diplotypes.
  
  # Summarize the genotype allele frequencies.
  # TBD: should we skip markers with too many no-calls?
  af = count(df, gt)
  print(af[af$gt == '-',])
  
  # Which founders are missing?
  missing = which(fg == '--')
  
  # Loop through and attempt to resolve each founder.
  for(i in seq_along(missing)) {

    # Get founder diplotype and single letter code.
    founder_dt     = names(missing)[i]
    founder_letter = str_sub(founder_dt, 1, 1)
    
    # Summarize the samples genotypes to include only the most common
    # genotype for each diplotype. 
    # NOTE: Excluding ties, which might cause us to miss some genotypes. 
    cand_gt = df %>%
                count(dt, gt) %>% 
                group_by(dt) %>% 
                slice_max(n, n = 1, with_ties = FALSE)
    
    # Get founders with homozygous calls.
    hom = cand_gt %>%
            filter(dt %in% founder_diplotypes) %>%
            filter(str_sub(gt, 1, 1) == str_sub(gt, 2, 2))

    # If the missing founder is in this set, replace the missing 
    # consensus genotype for this founder, with this genotype.
    if(founder_dt %in% hom$dt) {
    
      founder_consensus[m, founder_dt] = hom$gt[hom$dt == founder_dt]
    
    } else {
    # Otherwise, we try to use the hets and remaining homozygotes to 
    # infer the genotypes.

      # Get diplotypes which contain the current founder letter and
      # retain the homozygous genotypes.
      het = cand_gt %>%
              filter(str_detect(dt, founder_letter)) %>%
              filter(str_sub(gt, 1, 1) == str_sub(gt, 2, 2)) %>%
              mutate(a = str_sub(dt, 1, 1),
                     b = str_sub(dt, 2, 2),
                     a = if_else(b == founder_letter, a, b)) %>%
              select(a, dt, gt)
              
      hom = hom %>%
              mutate(a = str_sub(dt, 1, 1),
                     b = str_sub(dt, 2, 2)) %>%
              select(a, dt, gt)
              
      tmp = inner_join(hom, het, by = 'a', suffix = c("_hom", "_het"))

      # If there is only one homozygous call for all of the het diplotypes
      # for this founder, then this should be the founder genotype.
      # If there are more than one homozygous call for the het diplotypes,
      # then I think that the founder genotype is indeterminate.
      if(length(unique(het$gt)) == 1) {
      
        founder_consensus[m, founder_dt] = het$gt[1]
      
      } # if(length(unique(het$gt) == 1)

    } # else
     
  } # for(i)

} # for(m)

# Write chromosome-level founder consensus genotype file
if(chr == "X"){
  print("Writing Chromosome X Consensus Genotypes")
  write.csv(founder_consensus, 
            file = file.path(out_dir,"GigaMUGA_founder_consensus_genotypes","GigaMUGA_founder_consensus_imputed_genotypes_chrX.csv"), 
            quote = F)
} else {
  print(paste0("Writing Chromosome ", chr, " Consensus Genotypes"))
  write.csv(founder_consensus, 
            file = file.path(out_dir,"GigaMUGA_founder_consensus_genotypes",paste0("GigaMUGA_founder_consensus_imputed_genotypes_chr", chr, ".csv")), 
            quote = F)
  
}
          