.libPaths(c('/ncf/mclaughlin/users/jflournoy/R/x86_64-pc-linux-gnu-library/verse-4.2.1', .libPaths()))

# To do:
# - reaction time as predictor separate from prepotency effect somehow
# - preconditioning should be part of main combined model
# - plots should all come from the full comvbined model.

library(data.table)
library(brms)
library(argparse)
source('fit_model_functions.R')
# create parser object
parser <- ArgumentParser()

# specify our desired options 
# by default ArgumentParser will add a help option 
parser$add_argument(
  "--model", type = "character", default = 'rt_age',
  help = sprintf("Which model to fit. Valid options are: %s", 
                 paste(get_model_options(model_name = NULL, names = TRUE), collapse = ', ')))
parser$add_argument('--id', type = "integer", default = 1, help = 'Chain ID')
parser$add_argument("--test", action = "store_true", 
                    help = "Run on subset of data.")
parser$add_argument('--testprop', type = "double", default = .0125, help = 'Proportion sample to use in test')
parser$add_argument("--refit", action = "store_true", 
                    help = "Overwrite existing model.")
parser$add_argument("--kfold", action = "store_true", 
                    help = "Run k-fold CV using participant clusters.")
parser$add_argument('--nfolds', type = "integer", default = 5, help = 'Number of folds')
parser$add_argument('--foldid', type = "integer", default = 1, help = 'Fold ID')
parser$add_argument("--par_chains", action = "store_true", 
                    help = "Ignore `--id` and run 4 parallel chains.")
parser$add_argument("--long", action = "store_true", 
                    help = "Use longitudinal data.")
parser$add_argument("--onlylong", action = "store_true", 
                    help = "Use _only_ longitudinal data.")
if(interactive()){
  #parser$parse_args('--help')
  args <- parser$parse_args(c('--model', 'acc_prevcond', '--test', '--testprop', '.05','--id', '1', '--kfold', '--foldid', '1'))
  CPUS <- 16
} else {
  args <- parser$parse_args()
  CPUS <- as.numeric(Sys.getenv('SLURM_CPUS_PER_TASK'))
}

MODEL <- args$model
TEST <- args$test
TESTPROP <- args$testprop
CHAINID <- args$id
REFIT <- args$refit
PARCHAINS <- args$par_chains
KFOLD <- args$kfold
FOLDID <- args$foldid
NFOLDS <- args$nfolds
LONG <- args$long | args$onlylong
ONLYLONG <- args$onlylong

if(PARCHAINS){
  message('Ignoring `--id` option, running 4 parallel chains...')
  CHAINID <- 1234
}

model_data <- get_model_data(MODEL = MODEL, 
                             TEST = TEST, 
                             TESTPROP = TESTPROP, 
                             LONG = LONG, 
                             ONLYLONG = ONLYLONG)

if(KFOLD){
  set.seed(92104)
  folds <- loo::kfold_split_grouped(K = NFOLDS, x = as.character(model_data$sID_factor))
  omitted <- predicted <- which(folds == FOLDID)
  full_data <- model_data
  model_data <- model_data[-omitted]
}

if ('exact_prepotency_ofactor' %in% names(model_data)){
  message("Unique values of exact prepotency ordered factor: ", 
          unique(model_data$exact_prepotency_ofactor))
}

#This sets up the specification for all models described in the preregistration.
brm_model_options <- get_model_options(MODEL)

file_refit <- ifelse(REFIT, 'always', 'on_change')

brm_model_options$file <- sprintf('%s%s%s%s_c%02d', 
                                  brm_model_options$file,
                                  ifelse(ONLYLONG, '_onlylong',
                                         ifelse(LONG, '_long', '')),
                                  ifelse(KFOLD, sprintf('_k%02d', FOLDID), ''),
                                  ifelse(TEST, sprintf('_testp%03d', round(TESTPROP*1000, 0)), ''), CHAINID)

if(PARCHAINS){
  chain_opts <- list(chains = 4, cores = 4)
  if(CPUS > 4){
    chain_opts <- c(chain_opts, list(threads = CPUS / 4))
  }
} else {
  chain_opts <- list(chains = 1, cores = 1, threads = CPUS, chain_ids = CHAINID)
  if(TEST){
    chain_opts$threads <- CPUS
  }
}

brm_options <- c(brm_model_options, chain_opts,
                 list(data = model_data,
                      file_refit = file_refit,
                      backend = 'cmdstanr',
                      iter = 2500, warmup = 1500, 
                      control = list(adapt_delta = .9999999999999, max_treedepth = 20)))
print('\n\nbrms options ///')
print(brm_options)
print('\n\nDefault priors ///')
print(get_prior(formula = brm_options$formula, data = brm_options$data))
print('\n\nOur priors ///')
print(brm_options$prior)
if(file.exists(sprintf('%s.rds', brm_options$file))){
  message(sprintf('Model already exists. %s', 
                  ifelse(REFIT, 
                         'Refitting and overwriting...',
                         'Will refit only if model has changed...')))

}
fit <- do.call(brm, brm_options)

if(KFOLD){
  fit$manual_kfold <- brms:::nlist(full_data, omitted, predicted, folds)
  saveRDS(fit, sprintf('%s.rds', brm_options$file))
}

summary(fit)

sessionInfo()

##
# plot(conditional_effects(fit), points = FALSE, ask = FALSE)
# ranefplots <- visibly::plot_coefficients(fit, ranef = TRUE, which_ranef = 'sID_factor:runN_factor')
# library(ggplot2)
# ranefplots[[6]] + theme(axis.text.x = element_text(angle = 45, hjust = 1))
# variables(fit)
# plot(fit)
# library(priorsense)
# psens <- priorsense::powerscale_sensitivity(fit)
# print(psens$sensitivity, n = 118)
# pss <- priorsense::powerscale_sequence(fit)
# powerscale_plot_ecdf(pss, variables = variables(fit)[c(1:12)])
# fit_post <- as_draws_array(fit)
# np <- nuts_params(fit)
# if(FALSE){
#   library(ggplot2)
#   ggplot(model_data, aes(x = time, y = RT.shape, group = sID_factor)) + 
#     geom_line(stat = 'smooth', method = 'gam', formula = y ~ s(x), alpha = .8) + 
#     geom_smooth(aes(group = NULL), method = 'gam', formula = y ~ s(x), alpha = .8) +
#     facet_grid(~runN_factor)
#   
# }
