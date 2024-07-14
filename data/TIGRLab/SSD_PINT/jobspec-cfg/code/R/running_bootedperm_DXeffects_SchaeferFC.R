'Calculating boots from for DX cohens D Schaefer

Usage:
  naval_fate.R <iteration>

Options:
  -h --help     Show this screen.

' -> doc

# install.packages('docopt', lib = "/home/edickie/R/x86_64-pc-linux-gnu-library/4.1")
# install.packages('tidyverse', lib = "/home/edickie/R/x86_64-pc-linux-gnu-library/4.1")
# install.packages('here', lib = "/home/edickie/R/x86_64-pc-linux-gnu-library/4.1")
# install.packages('modelr',  lib = "/home/edickie/R/x86_64-pc-linux-gnu-library/4.1")
# install.packages('effsize',  lib = "/home/edickie/R/x86_64-pc-linux-gnu-library/4.1")
# install.packages('furrr',  lib = "/home/edickie/R/x86_64-pc-linux-gnu-library/4.1")



library(docopt)
arguments <- docopt(doc)
print(arguments)



library(tidyverse)
library(knitr)
library(here)
library(modelr)
library(effsize)
library(furrr)


# These functions are for reading timeseries files
source(here('code/R/settings_helpers.R'))
#

pheno <- read_pheno_file()%>%
  drop_na(DX) %>%
  filter(in_matched_sample)
YeoNet_colours <- define_Yeo7_colours()

# these tables describe the subcortical data and combined node info
the_subcortical_guide <- get_subcortical_guide()

Schaefer_labels <- read_Schaefer_template()
Schaefer_node_annots <- get_Schaefer_node_annotations(Schaefer_labels, the_subcortical_guide)


lm_predictor_col = c("DX")
lm_covar_cols <- c("Age_match_pt", 
                   "Sex",
                   "fd_mean_match_pt",
                   "Site")

combat_data_corZ <- readRDS(file = file.path(output_base, "all_clinicalplusqa_group", "Rdata_cache", "06_wholebrain_FC_Schaefercombat_cache.rds"))

pheno <- pheno %>% mutate(subject_dataset = str_c(subject, "_", dataset))

results_pheno <- combat_data_corZ %>%
  pivot_longer(all_of(pheno$subject_dataset),
               names_to = "subject_dataset",
               values_to = "weight") %>%
  inner_join(pheno, by = "subject_dataset") %>%
  select(vertex_type, to, from, subject, dataset, weight, DX, all_of(lm_covar_cols)) %>%
  ungroup()

calc_DX_cohenD <- function(df, outcome, predictor, covars) { 
  m1 <- lm(formula(paste(outcome, '~', paste(covars, collapse = " + "))),
           data = df)
  result <-df %>% 
    add_residuals(m1) %>%
    cohen.d(formula(paste("resid ~", predictor)), data = .) %>% 
    .$estimate
  return(result)
}

# Create an empty data frame to store the results
roiwise_boot_cohenDDX_results <- data.frame()


print(paste("This is:", arguments$iteration))

ibase <- as.numeric(arguments$iteration)*10
ibasef <- sprintf('%03i',ibase)

outfile <- file.path(output_base, 
          "all_clinicalplusqa_group", 
          "Rdata_cache", 
          "06b_SchaeferFC_boots", 
          str_c("06b_wholebrain_FC_Schaefercombat_roi_DXboots_cache",
                ibasef,
                ".rds"))

if (file.exists(outfile)) {
  print(paste("Already found output", outfile))
} else {

print(paste('running boots starting at i', ibasef))


# Run the code snippet 100 times
for (i in ibase:(ibase + 9)) {
  
  set.seed(i)
  
  #for (this_to in unique(results_pheno$to)) {
    
    print(paste("boot",i))
    # Execute the code snippet
    this_subsample <- pheno %>%
      group_by(DX) %>%
      sample_frac(size = 0.5) %>%
      select(subject, dataset, DX) 
    
    this_result <-  results_pheno %>%
      semi_join(this_subsample, by = c("subject", "dataset")) %>%
      #filter(to == this_to) %>%
      mutate(corZ = weight) %>%
      ungroup() %>%
      select(vertex_type, to, from, subject, corZ, DX, all_of(lm_covar_cols)) %>%
      group_by(vertex_type, to, from) %>%
      nest() %>%
      #slice(1:3) %>%
      mutate(DX_cohenD = future_map(data, ~calc_DX_cohenD(.x, 
                                                   outcome = "corZ", 
                                                   predictor = "DX",
                                                   covars = lm_covar_cols))) %>%
      unnest(DX_cohenD) %>%
      ungroup() %>%
      select(vertex_type, to, from, DX_cohenD) %>%
      pivot_wider(id_cols = c("to", "from"),
                  names_from = "vertex_type",
                  values_from = DX_cohenD) %>%
      mutate("surf_minus_vol" = surfschaefer - volschaefer,
              i = i)
  #}          
  
  # Combine the results with the existing data frame
  roiwise_boot_cohenDDX_results <- bind_rows(roiwise_boot_cohenDDX_results, this_result)
  
}

print(paste("Writing outputfile: ",outfile))
saveRDS(roiwise_boot_cohenDDX_results, 
        file = outfile)

}

print("All done!")
