library(tidyverse)
library(brms)
library(projpred)
library(loo)

# load utility functions
source('R/real-world/config.R')
source('R/forward-search/forward_search.R')
source('R/forward-search/projpred_search.R')

# read command line arguments
args <- commandArgs(trailingOnly = TRUE)
datasets_file <- args[[1]]
cv_fold <- args[[2]]
cv_fold <- as.numeric(cv_fold)

print(datasets_file)
print(cv_fold)

# read the dataset
datasets <- readRDS(datasets_file)
current_data <- datasets[[cv_fold]]
df <- current_data$train_df
df_test <- current_data$test_df
n <- current_data$n_train
n_test <- current_data$n_test
data_name <- current_data$data

# define the priors
prior = c(
  prior(horseshoe(df = heart_df, scale_slab = heart_scale_slab, 
                  par_ratio = heart_par_ratio), 
        class = 'b'),
  prior('normal(0, 2.5)', class='Intercept')
)
prior_normal = c(
  prior('normal(0, 1)', class = 'b'),
  prior('normal(0, 2.5)', class='Intercept')
)

# fit the reference model
reference = brm(y ~ ., 
                data = df, 
                prior = prior, 
                family = bernoulli(link = "probit"),
                chains = 4, 
                cores = Sys.getenv('SLURM_CPUS_PER_TASK'), 
                backend = "cmdstanr",
                stan_model_args=list(stanc_options = list("O1")),
                #control=list(adapt_delta=0.95),
                silent=2, 
                refresh=0)
ref_elpd_loo = get_elpd_loo(reference)
ref_elpd_test = get_elpd_test(reference, df_test)

base = update(reference, 
              formula = y ~ 1, 
              chains = 4, 
              cores = Sys.getenv('SLURM_CPUS_PER_TASK'), 
              backend = "cmdstanr",
              refresh = 0, 
              silent = 2)
base$variable = 'base'
base$loo = loo(base)

# Run forward selection with LOO-ELPD as the selection criteria
out = run_forward_selection(
  model = base,
  train_data = df,
  test_data = df_test,
  prior = prior,
  steps = 10, # define maximum number of predictors 
  # Following columns are added as metadata to the final output table
  prior_name='R2D2 prior',
  data_name=data_name,
  fold=cv_fold,
  elpd_loo_ref=ref_elpd_loo,
  elpd_test_ref=ref_elpd_test,
  n=n,
  n_test=n_test)

# Run forward selection with LOO-ELPD as the selection criteria
out_normal = run_forward_selection(
  model = base,
  train_data = df,
  test_data = df_test,
  prior = prior_normal,
  steps = 10, # define maximum number of predictors 
  # Following columns are added as metadata to the final output table
  prior_name='Normal prior',
  data_name=data_name,
  fold=cv_fold,
  elpd_loo_ref=ref_elpd_loo,
  elpd_test_ref=ref_elpd_test,
  n=n,
  n_test=n_test)

# Run projpred
out_projpred = run_projpred_varsel(
  reference, 
  df_test, 
  steps = 10, # define maximum number of predictors 
  # Following columns are added as metadata to the final output table
  prior_name='projpred', 
  data_name=data_name,
  fold=cv_fold,
  elpd_loo_ref=ref_elpd_loo,
  elpd_test_ref=ref_elpd_test,
  n=n,
  n_test=n_test
)

# save the data
output_file <- paste0("real_results_",data_name,"_",cv_fold)
saveRDS(list(forward_search=out$results,
             candidates=out$candidates,
             forward_search_normal=out_normal$results,
             candidates_normal=out_normal$candidates,
             projpred=out_projpred), 
        file = paste0("data/results/real/", output_file, ".RDS"))
