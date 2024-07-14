#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
print(args)

# Setup -------------------------------------------------------------------

ncores = as.integer(args[1])
# Number of MC replications in the sensitivity analysis
n_sim = 5e3
# Number of MC samples for computing the ICA and related measures.
n_prec = 1e4
# Number of bootstrap replications for computing uncertainty intervals.
B = 5e2

library(Surrogate)
library(dplyr)
library(tidyr)
library(copula)
library(FNN)

# We need the best fitted vine copula model.
best_fitted_model = readRDS("results/best-fitted-model.rds")

# We define all different scenarios for the set of sensitivity analysis in the
# scenarios_tbl.
sensitivity_ranges = tibble(
  ranges = list(list(
    lower = c(0.5, 0, 0, 0.2),
    upper = c(0.95, 0, 0, 0.8)
  ), list(
    lower = c(0.40, 0, 0, 0.15),
    upper = c(0.975, 0.20, 0.20, 0.90)
  )),
  range_class = c("Main Assumptions", "Relaxed Assumptions"),
  cond_ind = c(TRUE, FALSE)
)
# We consider all combinations of parameter ranges, unidentifiable copula
# families, and ICA (Spearman's rho or SICC).
scenarios_tbl = expand_grid(
  sensitivity_ranges,
  copula_family = c("gaussian", "frank", "gumbel", "clayton"),
  ICA_type = c("SICC", "Spearman's correlation")
)
# The SICC can be replaced with any measure by replacing the mutual information
# estimator with an estimator of -0.5 * log(1 - measure).
scenarios_tbl = scenarios_tbl %>%
  mutate(mutinfo_estimator = list(function(x, y) {
    -0.5 * log(1 - stats::cor(x, y, method = "spearman"))
  }))
scenarios_tbl$mutinfo_estimator[scenarios_tbl$ICA_type == "SICC"] = list(NULL)

# Sensitivity Analysis ----------------------------------------------------

# We use a wrapper function for the sensitivity analysis such that we set the
# same seed for each different version of the sensitivity analysis.
wrapper_sensitivity_analysis = function(cond_ind, copula_family, lower, upper, mutinfo_estimator) {
  set.seed(1)
  sensitivity_analysis_SurvSurv_copula(
    fitted_model = best_fitted_model,
    n_sim = n_sim,
    n_prec = n_prec,
    ncores = ncores,
    marg_association = TRUE,
    eq_cond_association = TRUE,
    composite = TRUE,
    copula_family2 = copula_family,
    degrees = 0,
    lower = lower,
    upper = upper,
    mutinfo_estimator = mutinfo_estimator
  )
}
# Similarly for the uncertainty intervals.
wrapper_uncertainty_intervals = function(sens_results, mutinfo_estimator, measure) {
  set.seed(1)
  # We save some computational time if we do not compute the ICA when we're only
  # looking at Spearman's rho in the full population.
  if (measure == "sp_rho") mutinfo_estimator = function(x, y) return(0)
  sensitivity_intervals_Dvine(
    fitted_model = best_fitted_model,
    sens_results = sens_results,
    measure = measure,
    mutinfo_estimator = mutinfo_estimator,
    n_prec = n_prec,
    B = B,
    ncores = ncores
  )
}

# The sensitivity analysis is implemented in this file, but the results of the
# sensitivity analysis are saved into an .RData file and processed elsewhere.
# This is done because the sensitivity analysis is computer intensive and not
# interactive; whereas processing the results is not computer intensive, but
# interactive.
a = Sys.time()
sens_results_tbl = scenarios_tbl %>%
  rowwise(everything()) %>%
  summarize(
    sens_results = list(wrapper_sensitivity_analysis(
      cond_ind,
      copula_family,
      ranges$lower,
      ranges$upper,
      mutinfo_estimator
    ))
  ) %>%
  ungroup()
# The uncertainty intervals are computed as well.
sens_results_tbl = sens_results_tbl %>%
  rowwise(everything()) %>%
  summarize(
    sens_interval_ICA_subset = list(
      wrapper_uncertainty_intervals(sens_results, mutinfo_estimator, measure = "ICA")
    ),
    sens_interval_sprho_full = list(
      wrapper_uncertainty_intervals(sens_results, mutinfo_estimator, measure = "sp_rho")
    )
  )
print(Sys.time() - a)

# Saving Results ----------------------------------------------------------

# The results of the sensitivity analysis are saved to a file. These results are
# analyzed in a separate file.
saveRDS(
  object = sens_results_tbl,
  file = "results/sensitivity-analysis-results-relaxed.rds"
)
