library(tidyverse)
library(brms)
library(loo)

# Config loads dataset configs, hyperparameters and utility functions
source('R/forward-search/config.R')
source('R/forward-search/forward_search.R')

args <- commandArgs(trailingOnly = TRUE)

# read command line arguments
datasets_file <- args[[1]]
n <- args[[2]]
rho <- args[[3]]
dataset_iter <-  args[[4]]

print(paste0("n: ", n, " rho: ", rho))

# ensure correct data types
n <- as.numeric(n)
rho <- as.numeric(rho)
dataset_iter <- as.numeric(dataset_iter)

# read the dataset
datasets <- readRDS(datasets_file)
current_data <- datasets[[dataset_iter]]
df <- current_data$train
df_test <- current_data$test

prior = c(
  prior(R2D2(mean_r2, prec_r2, cons_d2), class = 'b'),
  prior('normal(0, 2.5)', class='Intercept'),
  prior('exponential(1)', class='sigma')
)

prior_normal = c(
  prior('normal(0, 1)', class = 'b'),
  prior('normal(0, 2.5)', class='Intercept'),
  prior('exponential(1)', class='sigma')
)

reference = brm(y ~ ., 
                data = df, 
                prior = prior, 
                backend = 'cmdstanr',
                stan_model_args=list(stanc_options = list("O1")),
                control=list(adapt_delta=0.95),
                silent=2, 
                refresh=0)
ref_elpd_loo = get_elpd_loo(reference)
ref_elpd_test = get_elpd_test(reference, df_test)

base = update(reference, formula=y ~ 1, refresh=0, silent=2)
base$variable = 'base'
base$loo = loo(base)

# Run forward selection with LOO-ELPD as the selection criteria
out = run_forward_selection(
  model = base,
  train_data = df,
  test_data = df_test,
  prior = prior,
  # Following columns are added as metadata to the final output table
  name='R2D2 prior',
  iter=dataset_iter,
  elpd_loo_ref=ref_elpd_loo,
  elpd_test_ref=ref_elpd_test,
  n=n,
  rho=rho,
  n_test=n_test)
results <- out$results
candidates <- out$candidates

# Run forward selection with LOO-ELPD as the selection criteria
out_normal = run_forward_selection(
  model = base,
  train_data = df,
  test_data = df_test,
  prior = prior_normal,
  # Following columns are added as metadata to the final output table
  name='Normal prior',
  iter=dataset_iter,
  elpd_loo_ref=ref_elpd_loo,
  elpd_test_ref=ref_elpd_test,
  n=n,
  rho=rho,
  n_test=n_test)
results_normal <- out_normal$results
candidates_normal <- out_normal$candidates

output_file <- paste0("forward_results_",dataset_iter,"_",n,"_",rho)
saveRDS(list(forward_search=results,
             candidates=candidates,
             forward_search_normal=results_normal,
             candidates_normal=candidates_normal), 
        file = paste0("data/results/forward-search/", output_file, ".RDS"))

print(output_file)
print("done!")
