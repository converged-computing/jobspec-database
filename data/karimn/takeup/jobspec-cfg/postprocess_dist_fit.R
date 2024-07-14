#!/usr/bin/Rscript
#
# This script is used to postprocess Stan fit for the various models, reduced form and structural. In addition to putting our analysis in a format that 
# allows for easy extraction of all levels and treatment effects, it allows handles imputing take-up levels for counterfactuals using Stan-generated cost-benefits 
# and reputational returns parameters: using these we can calculate the probability of take-up after calculating the v^* fixed point solution.
#

script_options <- docopt::docopt(
  stringr::str_glue(
"Usage:
  postprocess_dist_fit.R <fit-version> [--full-outputname --cores=<num-cores> --output-path=<path> --input-path=<path> --load-from-csv --no-prior --no-rate-of-change --keep-fit --models=<models> --single-chain]
  
Options:
  --cores=<num-cores>  Number of cores to use [default: 12]
  --input-path=<path>  Path to find results [default: {file.path('data', 'stan_analysis_data')}]
  --output-path=<path>  Path to find results [default: temp-data]
  --keep-fit 
  --single-chain  For debugging purposes, only use the first chain
  
  "), 
  args = if (interactive()) "
  95
  --cores=1 
  --output-path=tmp
  --load-from-csv 
  " else commandArgs(trailingOnly = TRUE)
)

  # --single-chain
  # --no-prior
  # --no-rate-of-change
  # --models=STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_FOB

library(magrittr)
library(tidyverse)
library(rlang)
library(cmdstanr)
library(tidybayes)
library(furrr)

source("analysis_util.R")
source(file.path("multilvlr", "multilvlr_util.R"))
source("dist_structural_util.R")

fit_version <- script_options$fit_version
postprocess_cores <- as.integer(script_options$cores)

if (postprocess_cores > 1) {
  if (interactive()) {
    plan(multisession, workers = postprocess_cores)
  } else {
    plan(multicore, workers = postprocess_cores)
  }
} else {
  plan(sequential)
}

# Analysis Data --------------------------------------------------------------------

load(file.path("data", "analysis.RData"))

standardize <- as_mapper(~ (.) / sd(.))
unstandardize <- function(standardized, original) standardized * sd(original)

monitored_nosms_data <- analysis.data %>% 
  filter(mon_status == "monitored", sms.treatment.2 == "sms.control") %>% 
  left_join(village.centers %>% select(cluster.id, cluster.dist.to.pot = dist.to.pot),
            by = "cluster.id") %>% 
  mutate(standard_cluster.dist.to.pot = standardize(cluster.dist.to.pot)) %>% 
  group_by(cluster.id) %>% 
  mutate(cluster_id = cur_group_id()) %>% 
  ungroup()

nosms_data <- analysis.data %>% 
  filter(sms.treatment.2 == "sms.control") %>% 
  left_join(village.centers %>% select(cluster.id, cluster.dist.to.pot = dist.to.pot),
            by = "cluster.id") %>% 
  mutate(standard_cluster.dist.to.pot = standardize(cluster.dist.to.pot)) %>% 
  group_by(cluster.id) %>% 
  mutate(cluster_id = cur_group_id()) %>% 
  ungroup()

analysis_data <- monitored_nosms_data

# Load Data ---------------------------------------------------------------

roc_param <- c("cluster_roc_diff", "cluster_roc_diff_diffdist", 
               # str_c(rep(c("cluster_roc", "cluster_rep_return", "cluster_social_multiplier", "cluster_w_cutoff", "cluster_takeup_prop"), each = 2), c("_left", "_right")))
               "cluster_roc", "cluster_rep_return", "cluster_rep_return_dist", "cluster_social_multiplier", "cluster_w_cutoff", "cluster_takeup_prop")

param_used <- c(
  "total_error_sd", "u_sd", "cluster_cf_cutoff", roc_param, "sim_delta", "obs_cluster_mu_rep", # "mu_beliefs_effect", 
  "group_dist_mean", "group_dist_sd", "group_dist_mix", "missing_cluster_standard_dist", 
  "prob_prefer_calendar", "strata_wtp_mu", "hyper_wtp_mu",
  "prob_1ord", "prob_2ord", "ate_1ord", "ate_2ord",
   "wtp_travel_dist", "calendar_preference_in_dist" 
)

# param_used %<>% c("structural_cluster_takeup_prob", "missing_cluster_standard_dist") 

load_from_csv <- function(fit_file, input_path, param) {
  if (script_options$single_chain) {
    dir(input_path, pattern = str_c(str_remove(basename(fit_file), fixed(".rds")), r"{-1.csv}"), full.names = TRUE) %>% 
        read_cmdstan_csv(variables = param) 
  } else {
    dir(input_path, pattern = str_c(str_remove(basename(fit_file), fixed(".rds")), r"{-\d+.csv}"), full.names = TRUE) %>% 
        read_cmdstan_csv(variables = param) 
  }
}

load_fit <- function(fit_file, input_path = script_options$input_path, load_from_csv = script_options$load_from_csv, param = param_used) {
  if (load_from_csv) {
    load_from_csv(fit_file, input_path, param) %>% 
      pluck("post_warmup_draws") %>% 
      posterior::as_draws_df() %>% 
      mutate(iter_id = .draw) %>% 
      pivot_longer(!c(iter_id, .draw, .iteration, .chain), names_to = "variable", values_to = "iter_est") %>% 
      nest(iter_data = !variable)
  } else {
    read_rds(file.path(input_path, basename(fit_file)))
  }
}

# Metadata on the models fit
model_info <- tribble(
  ~ model,                                             ~ model_name,                                         ~ model_type,
  
  "REDUCED_FORM_NO_RESTRICT",                          "Reduced Form",                                       "reduced form",
  "REDUCED_FORM_NO_RESTRICT_DIST_CTS",                 "Reduced Form",                                       "reduced form",
  "STRUCTURAL_LINEAR_U_SHOCKS",                        "Structural",                                         "structural",
  # "STRUCTURAL_LINEAR",              "Structural",                                "structural",
  # "STRUCTURAL_QUADRATIC",           "Structural Quadratic Cost",                 "structural",
  # "STRUCTURAL_QUADRATIC_NO_SHOCKS", "Structural Quadratic Cost (No Shocks)",     "structural",
  # "STRUCTURAL_LINEAR_NO_SHOCKS",    "Structural Linear Cost (No Shocks)",        "structural",
  # "STRUCTURAL_QUADRATIC_SALIENCE",  "Structural Quadratic Cost With Salience",   "structural",
  # "STRUCTURAL_LINEAR_SALIENCE",     "Structural With Salience",                 "structural",


  "STRUCTURAL_LINEAR_U_SHOCKS_LOG_MU_REP",             "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_LINEAR_MU_REP",          "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_NO_REP",                 "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP",             "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_HIGH_MU_WTP_VAL",          "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_HIGH_SD_WTP_VAL",          "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_NO_WTP_SUBMODEL",             "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_NO_BELIEFS_SUBMODEL",          "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_NO_SUBMODELS",          "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_HIER_FOB",             "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_HIER_SOB",             "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_STRATA_SOB",             "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_SOB",             "Structural",                                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_FOB",             "Structural",                                         "structural",


  
  "STACKED",                                          "Stacked Model",                                       "combined", # Includes both reduced form and structural models
  "STRUCTURAL_STACKED",                               "Structural Stacked Model",                            "structural", 
  "STRUCTURAL_LINEAR_U_SHOCKS_NO_SUBMODELS",          "Structural without Any Submodel Data",                "structural", 
  "STRUCTURAL_LINEAR_U_SHOCKS_NO_BELIEFS_SUBMODEL",   "Structural without Beliefs Data",                     "structural", 
  "STRUCTURAL_LINEAR_U_SHOCKS_NO_WTP_SUBMODEL",       "Structural without WTP Data",                         "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_NO_WTP_TAKEUP",         "Structural without Takeup or WTP Data",               "structural",
  "STRUCTURAL_LINEAR_U_SHOCKS_NO_BELIEFS_DIST",       "Structural without Distance in Beliefs Model",        "structural",
) %>% 
  mutate(model_type = factor(model_type, levels = c("reduced form", "structural", "combined")))

# Stan fit 
dist_fit_data <- tryCatch({
  if (script_options$full_outputname) {
    load(file.path(script_options$input_path, str_interp("${fit_version}.RData")))
  } else {
    load(file.path(script_options$input_path, str_interp("dist_fit${fit_version}.RData")))
  }
  
  if (!is_null(script_options$models)) {
    dist_fit %<>%  magrittr::extract(script_options$models)
  }



#  dist_fit_present_lgl = dist_fit %>% map(~str_c(str_remove(.x, fixed(".rds")), r"{-1.csv}")) %>%
#    map_lgl(file.exists)

#  dist_fit = dist_fit[dist_fit_present_lgl]
  
  dist_fit %<>% 
    map_if(is.character, ~ tryCatch(load_fit(.x), error = function(err) load_fit(.x, param = NULL)))  
    

  if (has_name(dist_fit, "value")) {
    dist_fit_warnings <- dist_fit$warning
    dist_fit %<>% 
      list_modify(!!!.$value, value = NULL, warning = NULL)
  }

  # Convert to tibble with list column of fit objects
  dist_fit_data <- enframe(dist_fit, name = "model", value = "fit") %>% 
    mutate(stan_data = list(stan_data))
  
  rm(dist_fit, stan_data)
  
  dist_fit_data
}, error = function(err) NULL)

# Prior predicted fit for same models
dist_fit_data <- if (!script_options$no_prior) { 
  tryCatch({
    load(file.path(script_options$input_path, str_interp("dist_prior${fit_version}.RData")))
    
    if (!is_null(script_options$models)) {
      dist_fit %<>%  magrittr::extract(script_options$models)
    }
    
    dist_fit %<>% 
      map_if(is.character, ~ tryCatch(load_fit(.x), error = function(err) load_fit(.x, param = NULL))) 
    
    dist_fit_data %<>%
      bind_rows("fit" = .,
                "prior-predict" = enframe(dist_fit, name = "model", value = "fit") %>% 
                  mutate(stan_data = list(stan_data)),
                .id = "fit_type") 
    
    rm(dist_fit, stan_data)
    
    dist_fit_data
  },
  error = function(err) {
    dist_fit_data %>% 
      mutate(fit_type = "fit")
  })
} else {
  dist_fit_data %>% 
    mutate(fit_type = "fit")
}
dist_fit_data %<>% 
  mutate(fit_type = factor(fit_type, levels = c("fit", "prior-predict")))
    
dist_fit_data <- tryCatch({
  # Load cross-validation data
  load(file.path(script_options$input_path, str_interp("dist_kfold${fit_version}.RData")))
  
  dist_fit_data <- enframe(dist_kfold, name = "model", value = "kfold") %>% 
    inner_join(select(model_info, model, model_type), by = "model") %>% 
    mutate(
      fit_type = factor("fit", levels = c("fit", "prior-predict")),
      stacking_weight = map(kfold, pluck, "pointwise") %>% 
        do.call(rbind, .) %>% 
        t() %>% 
        loo::stacking_weights(),
      
      kfold = set_names(kfold, model)
    ) %>% 
    # group_by(model_type) %>% 
    # mutate(
    #   stacking_weight_by_type = map(kfold, pluck, "pointwise") %>% 
    #     do.call(rbind, .) %>% 
    #     t() %>% { 
    #     tryCatch(loo::stacking_weights(.), error = function(err) NA)
    #     }
    # ) %>% 
    # ungroup() %>% 
    left_join(kfold_compare(x = discard(.$kfold, is_null)), by = "model") %>% 
    left_join(dist_fit_data, ., by = c("model", "fit_type")) %>% 
    select(-one_of("model_type")) %>% 
    mutate(across(where(~ is(.x, "stacking_weights")), as.numeric))

  rm(dist_kfold)
  
  return(dist_fit_data)
},
error = function(error) dist_fit_data)
dist_fit_data %<>% 
  inner_join(select(model_info, model, model_type), by = "model") 

observed_takeup <- monitored_nosms_data %>% 
  group_by(cluster_id, assigned_treatment = assigned.treatment, assigned_dist = cluster.dist.to.pot) %>% 
  summarize(prop_takeup = mean(dewormed), 
            se = sqrt(prop_takeup * (1 - prop_takeup) / n()),
            prop_takeup_ub = prop_takeup + se,
            prop_takeup_lb = prop_takeup - se, 
            .groups = "drop") 
# Functions ---------------------------------------------------------------

quant_probs <- c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

calculate_prob_and_num_takeup <- function(cutoffs, total_error_sd, force_draw_takeup = FALSE) {
  if (is_null(cutoffs)) {
    return(NULL)
  }
  
  prob_iter_data <- if (!is_null(total_error_sd)) {
    left_join(cutoffs, total_error_sd, by = c("assigned_treatment"), suffix = c("", "_total_error_sd")) %>%
      mutate(
        iter_data = map2(iter_data, iter_data_total_error_sd, ~ {
          left_join(.x, .y, by = "iter_id", suffix = c("", "_total_error_sd")) %>% 
            mutate(prob = pnorm(- iter_est, sd = iter_est_total_error_sd)) 
        })
      ) %>% 
      select(!ends_with("total_error_sd"))
  } else {
    mutate(cutoffs, iter_data = map(iter_data, ~ mutate(.x, prob = pnorm(- iter_est)))) 
  }
  
  prob_iter_data %>% 
    mutate(
      iter_data = future_pmap(lst(iter_data, cluster_size, obs_num_takeup), function(iter_data, cluster_size, obs_num_takeup, force_draw_takeup) {
        if (!is.na(obs_num_takeup) && !force_draw_takeup) {
          mutate(iter_data, iter_num_takeup = obs_num_takeup)
        } else {
          mutate(iter_data, iter_num_takeup = rbinom(n(), cluster_size, prob)) # If not observed treatment, draw random number of takers for cluster
        }
      }, force_draw_takeup = force_draw_takeup, .options = furrr_options(seed = TRUE))
    )
}

# Combine cluster level nested data to produce treatment level nested data
organize_by_treatment <- function(.data, condition_on_dist, ...) {
  if (is_null(.data)) {
    return(NULL)
  }
  
  .data %>% {
    if (condition_on_dist) { # Only use observed distance
      filter(., assigned_dist_group_obs == assigned_dist_group)
    } else .
  } %>% 
    select(..., cluster_size, iter_data) %>% 
    mutate(iter_data = map(iter_data, select, iter_id, prob, iter_num_takeup)) %>%
    unnest(iter_data) %>% 
    group_by(iter_id, ...) %>% 
    summarize(iter_prop_takeup = sum(iter_num_takeup) / sum(cluster_size), .groups = "drop") %>% 
    nest(iter_data = c(iter_id, iter_prop_takeup)) %>%
    mutate(
      mean_est = map_dbl(iter_data, ~ mean(.$iter_prop_takeup, na.rm = TRUE)),
      takeup_quantiles = map(iter_data, quantilize_est, iter_prop_takeup, wide = TRUE, quant_probs = c(quant_probs), na.rm = TRUE)
    ) %>% 
    unnest(takeup_quantiles)
}

level_stack_reducer <- function(accum, est_takeup_level, stacking_weight) {
  if (is.na(stacking_weight)) {
    return(accum)
  } else {
    weighed_level <- est_takeup_level %>% 
      select(-mean_est, -starts_with("per_")) %>% 
      mutate(iter_data = map(iter_data, mutate, iter_prop_takeup = iter_prop_takeup * stacking_weight),
             mu_assigned_treatment = if (!is.factor(mu_assigned_treatment)) factor(mu_assigned_treatment) else mu_assigned_treatment,
             mu_assigned_treatment = if_else(is.na(mu_assigned_treatment), assigned_treatment, mu_assigned_treatment))
     
    if (is_empty(accum)) {
      return(weighed_level)
    } else {
      inner_join(accum, weighed_level, by = setdiff(intersect(names(accum), names(weighed_level)), "iter_data"), suffix = c("_accum", "_weighed")) %>% 
        mutate(iter_data = map2(iter_data_accum, iter_data_weighed, inner_join, by = "iter_id", suffix = c("_accum", "_weighed")) %>% 
                 map(mutate, iter_prop_takeup = iter_prop_takeup_accum + iter_prop_takeup_weighed) %>% 
                 map(select, -ends_with("_accum"), -ends_with("_weighed"))) %>% 
        select(-ends_with("_accum"), -ends_with("_weighed"))
    }
  }
}

rate_of_change_stack_reducer <- function(accum, rate_of_change, stacking_weight) {
  if (is.na(stacking_weight)) {
    return(accum)
  } else {
    weighed_level <- rate_of_change %>% 
      select(-c(prob_takeup, social_multiplier, partial_bbar)) %>% 
      mutate(iter_data = map(iter_data, mutate_at, vars(iter_prob_takeup, iter_social_multiplier, iter_partial_bbar), ~ . * stacking_weight))
     
    if (is_empty(accum)) {
      return(weighed_level)
    } else {
      inner_join(accum, weighed_level, by = setdiff(intersect(names(accum), names(weighed_level)), "iter_data"), suffix = c("_accum", "_weighed")) %>% 
        mutate(iter_data = map2(iter_data_accum, iter_data_weighed, inner_join, by = "iter_id", suffix = c("_accum", "_weighed")) %>% 
                 map(mutate, 
                     iter_prob_takeup = iter_prob_takeup_accum + iter_prob_takeup_weighed,
                     iter_social_multiplier = iter_social_multiplier_accum + iter_social_multiplier_weighed,
                     iter_partial_bbar = iter_partial_bbar_accum + iter_partial_bbar_weighed) %>% 
                 map(select, -ends_with("_accum"), -ends_with("_weighed"))) %>% 
        select(-ends_with("_accum"), -ends_with("_weighed"))
    }
  }
}

summarize_roc <- function(param_data) {
  group_col <- c("roc_distance_index", "roc_distance", "treatment_id", "assigned_treatment", "iter_id") 
  
  unnest(param_data, iter_data) %>%
    group_by(across(all_of(intersect(group_col, names(.))))) %>%
    summarize(iter_est = mean(iter_est), .groups = "drop") %>% 
    ungroup() %>% 
    nest(iter_data = c(iter_id, iter_est)) %>% 
    mutate(
      mean_est = map_dbl(iter_data, ~ mean(.x$iter_est, na.rm = TRUE)),
      quants = map(iter_data, quantilize_est, iter_est, quant_probs = quant_probs, na.rm = TRUE)
    ) %>% 
    unnest(quants)
}

extract_sim_delta <- function(fit, stan_data) {
  temp <- fit %>% 
    filter(str_detect(variable, "sim_delta"))
  
  if (nrow(temp) > 0) {
    temp %>% 
      transmute(w = stan_data$sim_delta_w, iter_data) %>% 
      mutate(
        mean_est = map_dbl(iter_data, ~ mean(.x$iter_est, na.rm = TRUE)),
        quants = map(iter_data, quantilize_est, iter_est, quant_probs = quant_probs, na.rm = TRUE)
      ) %>% 
      unnest(quants)
  } else return(NULL)
}

ate_pivot <- function(level_data, ate_combo, other_ate_join_col = NULL) {
  if (!is_null(level_data)) {
    present_col <- intersect(names(level_data), c("mu_assigned_treatment", "assigned_treatment", "assigned_dist_group"))
    # ate_combo_col <- str_c(rep(present_col, each = 2), c("_left", "_right"))
    ate_combo_col <- paste0(rep(present_col, each = 2), c("_left", "_right"))
    
    left_data <- inner_join(select(ate_combo, all_of(ate_combo_col)) %>% distinct_all(.keep_all = TRUE),
                            level_data,
                            by = present_col %>% set_names(str_c(.,"_left")))
    
    inner_join(left_data,
               level_data,
               by = c(present_col %>% set_names(str_c(., "_right")), other_ate_join_col),
               suffix = c("_left", "_right"))
  }
}

calculate_ate <- function(level_list, model_type, other_ate_join_col = NULL) {
  level_list %>% 
    map_if(~ !is_null(.x), select_if, ~ !all(is.na(.))) %>% # Get rid of the NA mu_assigned_treatment in the reduced form results 
        map(ate_pivot, ate_combo = ate_combo, other_ate_join_col = other_ate_join_col) %>% 
    map_if(
      fct_match(model_type, "structural"), ~ {
        if (!is_null(.x)) {
          filter(., 
            mu_assigned_treatment_left != mu_assigned_treatment_right |
            assigned_treatment_left != assigned_treatment_right |
            (is.na(assigned_dist_group_left) & is.na(assigned_dist_group_right)) |
            assigned_dist_group_left != assigned_dist_group_right)
        }
      },
      .else = ~ filter(., 
        assigned_treatment_left != assigned_treatment_right |
        (is.na(assigned_dist_group_left) & is.na(assigned_dist_group_right)) |
        assigned_dist_group_left != assigned_dist_group_right)) %>% 
    map_if(~ !is_null(.x), mutate, iter_data = map2(iter_data_left, iter_data_right, inner_join, by = "iter_id", suffix = c("_left", "_right")) %>% 
          map(mutate, iter_takeup_te = iter_prop_takeup_left - iter_prop_takeup_right)) %>% 
    map_if(~ !is_null(.x), select, -iter_data_left, -iter_data_right) %>% 
    map_if(~ !is_null(.x), mutate, 
        mean_est = map_dbl(iter_data, ~ mean(.$iter_takeup_te, na.rm = TRUE)),
        takeup_te_quantiles = map(iter_data, quantilize_est, iter_takeup_te, wide = TRUE, quant_probs = c(quant_probs), na.rm = TRUE)) %>% 
    map_if(~ !is_null(.x), unnest, takeup_te_quantiles)
}

calculate_dist_ate <- function(ate_list, model_type, other_ate_join_col = NULL) {
  ate_list %>%
    map_if(~ !is_null(.x), filter, !is.na(assigned_dist_group_left), !is.na(assigned_dist_group_right)) %>% 
    map_if(fct_match(model_type, "structural") & map_lgl(., ~ !is_null(.x)), filter, mu_assigned_treatment_left == assigned_treatment_left, mu_assigned_treatment_right == assigned_treatment_right) %>% 
    map_if(~ !is_null(.x), select, !starts_with("mu_assigned_treatment")) %>% 
    map_if(~ !is_null(.x), filter, assigned_treatment_left == assigned_treatment_right, fct_match(assigned_dist_group_left, "far"), fct_match(assigned_dist_group_right, "close")) %>% 
    map_if(~ !is_null(.x), mutate, assigned_treatment_right = "control") %>% 
    map_if(~ !is_null(.x), select, !c(mean_est, starts_with("per_"), starts_with("assigned_dist_group"))) %>% 
    map_if(~ !is_null(.x), ~ inner_join(., select(., !assigned_treatment_right), by = c("assigned_treatment_right" = "assigned_treatment_left", other_ate_join_col), suffix = c("_left", "_right"))) %>% 
    map_if(~ !is_null(.x), filter, assigned_treatment_left != assigned_treatment_right) %>% 
    map_if(~ !is_null(.x), mutate, iter_data = map2(iter_data_left, iter_data_right, inner_join, by = "iter_id", suffix = c("_left", "_right")) %>% 
          map(transmute, iter_id, iter_takeup_dist_te = iter_takeup_te_left - iter_takeup_te_right)) %>% 
    map_if(~ !is_null(.x), select, !c(iter_data_left, iter_data_right)) %>% 
    map_if(~ !is_null(.x), mutate, 
        mean_est = map_dbl(iter_data, ~ mean(.$iter_takeup_dist_te)),
        takeup_te_quantiles = map(iter_data, quantilize_est, iter_takeup_dist_te, wide = TRUE, quant_probs = quant_probs, na.rm = TRUE)) %>% 
    map_if(~ !is_null(.x), unnest, takeup_te_quantiles)
}

#' Calculate Difference in ATEs across treatments, within distance groups
#'
#' Filtering process:
#' Ensure treat_left != treat_right. Ensure mu_assigned either matches treatment 
#' or is set to control in RF. Make sure distance groups match or are both NA
#'
#'
calculate_ate_diff = function(level_list, model_type, other_ate_join_col = NULL){
  level_list %>%
    map_if(~ !is_null(.x), select_if, ~ !all(is.na(.))) %>% # Get rid of the NA mu_assigned_treatment in the reduced form results 
        map(ate_pivot, ate_combo = ate_combo, other_ate_join_col = other_ate_join_col) %>% 
    map_if(
      fct_match(model_type, "structural"), ~ {
        if (!is_null(.x)) {
          filter(.,
            (assigned_treatment_left != assigned_treatment_right) &
            mu_assigned_treatment_left == assigned_treatment_left &
            mu_assigned_treatment_right == assigned_treatment_right &
            ((is.na(assigned_dist_group_left) & is.na(assigned_dist_group_right)) |
            (assigned_dist_group_left == assigned_dist_group_right))
          )
        }
      },
      .else = ~ filter(.,
        (assigned_treatment_left != assigned_treatment_right) &
        mu_assigned_treatment_left == "control" &
        mu_assigned_treatment_right == "control" &
        ((is.na(assigned_dist_group_left) & is.na(assigned_dist_group_right)) |
        (assigned_dist_group_left == assigned_dist_group_right)))
    ) %>%
    map_if(
      ~ !is_null(.x), mutate, iter_data = map2(iter_data_left, iter_data_right, inner_join, by = "iter_id", suffix = c("_left", "_right")) %>%
          map(mutate, iter_takeup_te_diff = iter_prop_takeup_left - iter_prop_takeup_right)
    ) %>%
      map_if(~ !is_null(.x), select, -iter_data_left, -iter_data_right) %>% 
      map_if(~ !is_null(.x), mutate, 
          mean_est = map_dbl(iter_data, ~ mean(.$iter_takeup_te_diff, na.rm = TRUE)),
          takeup_te_quantiles = map(iter_data, quantilize_est, iter_takeup_te_diff, wide = TRUE, quant_probs = c(quant_probs), na.rm = TRUE)) %>% 
      map_if(~ !is_null(.x), unnest, takeup_te_quantiles)
}
# Postprocessing ----------------------------------------------------------

dist_fit_data %<>% 
  mutate(
    wtp_travel_dist = map2(fit, stan_data, ~ extract_obs_fit_level(.x, par = "wtp_travel_dist", stan_data = .y, iter_level = "none", quant_probs = quant_probs)) %>% 
      map2(stan_data, ~ {
      if (!is_null(.x)) {
        unnest(.x, iter_data) %>% 
          mutate(iter_est = iter_est / unstandardize(1, .y$analysis_data$cluster.dist.to.pot))
      }
    }),
    calendar_preference_in_dist = map2(fit, stan_data, ~ extract_obs_fit_level(.x, par = "calendar_preference_in_dist", stan_data = .y, iter_level = "none", quant_probs = quant_probs)) %>% 
      map2(stan_data, ~ {
      if (!is_null(.x)) {
        unnest(.x, iter_data) %>% 
          mutate(iter_est = unstandardize(iter_est, .y$analysis_data$cluster.dist.to.pot))
      }
    }),
    
    total_error_sd = map2(fit, stan_data, ~ extract_obs_fit_level(.x, par = "total_error_sd", stan_data = .y, iter_level = "none", by_treatment = TRUE, summarize_est = FALSE, quant_probs = quant_probs)) %>% 
      map(., ~ {
        if (!is_null(.x)) {
          select(.x, !assigned_dist_group)
        }
      }),
    u_sd = map2(fit, stan_data, ~ extract_obs_fit_level(.x, par = "u_sd", stan_data = .y, iter_level = "none", quant_probs = quant_probs)),
    
    obs_cluster_mu_rep = map(fit, filter, str_detect(variable, "obs_cluster_mu_rep")),
    
    cluster_cf_cutoff = pmap(lst(fit, stan_data, model_type), extract_obs_cluster_cutoff_cf, quant_probs = quant_probs) %>% 
      lst(cutoffs = ., total_error_sd, force_draw_takeup = fct_match(fit_type, "prior-predict")) %>%  
      pmap(calculate_prob_and_num_takeup), 
    
    # cluster_age_group_cf_cutoff = pmap(lst(fit, stan_data, model_type), extract_obs_cluster_age_group_cutoff_cf, quant_probs = quant_probs) %>% 
    #   lst(cutoffs = ., total_error_sd, force_draw_takeup = fct_match(fit_type, "prior-predict")) %>%  
    #   pmap(calculate_prob_and_num_takeup), 
    
    # for structural model we can just average across all distances (covered on Stan side)
    # for RF model, we only want to use close distances in the close group and 
    # far distances in the far group when averaging across all clusters to get 
    # takeup levels.
    est_takeup_level = list(cluster_cf_cutoff, TRUE) %>% # FALSE, map_lgl(model_type, fct_match, "structural")) %>%  
      pmap(organize_by_treatment,  mu_assigned_treatment, assigned_treatment, assigned_dist_group) %>% 
      map2(cluster_cf_cutoff,
           ~ filter(.y, assigned_dist_group_obs == assigned_dist_group) %>%
             organize_by_treatment(condition_on_dist = TRUE, mu_assigned_treatment, assigned_treatment) %>%
             bind_rows(.x, .)),
    
    # est_age_group_takeup_level = list(cluster_age_group_cf_cutoff, FALSE) %>% # map_lgl(model_type, fct_match, "structural")) %>%  
    #   pmap(organize_by_treatment, mu_assigned_treatment, assigned_treatment, assigned_dist_group, age_group) %>% 
    #   map2(cluster_age_group_cf_cutoff, ~ {
    #     if (!is_null(.x)) {
    #       filter(.y, assigned_dist_group_obs == assigned_dist_group) %>%
    #         organize_by_treatment(condition_on_dist = TRUE, mu_assigned_treatment, assigned_treatment, age_group) %>%
    #         bind_rows(.x, .)
    #     }
    #   }),
    
    obs_cluster_takeup_level = map(cluster_cf_cutoff, filter, !is.na(obs_prop_takeup)) %>% 
      map(mutate, quants = map(iter_data, quantilize_est, prob, quant_probs = quant_probs, na.rm = TRUE)) %>% 
      map(select, !iter_data) %>% 
      map(unnest, quants),
    
    # obs_age_group_cluster_takeup_level = map_if(cluster_age_group_cf_cutoff, fct_match(model_type, "reduced form"), filter, !is.na(obs_prop_takeup)) %>% 
    #   map_if(fct_match(model_type, "reduced form"), mutate, quants = map(iter_data, quantilize_est, prob, quant_probs = quant_probs, na.rm = TRUE)) %>% 
    #   map_if(fct_match(model_type, "reduced form"), select, !iter_data) %>% 
    #   map_if(fct_match(model_type, "reduced form"), unnest, quants),
    
    group_dist_param = pmap(lst(fit, stan_data, model_type), get_dist_results),
    # imputed_dist = pmap(lst(fit, stan_data, model_type), get_imputed_dist),
    beliefs_results = map2(fit, stan_data, get_beliefs_results),
    wtp_results = map(fit, get_wtp_results),
  )

if (!script_options$no_rate_of_change) {
  

  # Back to Karim work
  dist_fit_data %<>% 
    mutate(
      pmap(list(param = roc_param, incr = if_else(str_detect(roc_param, "diff"), 1, 0), diff_diff = str_detect(roc_param, "diff_diffdist")), 
           function(param, incr, diff_diff, fit, stan_data) map2(fit, stan_data, extract_roc_param, param, incr, diff_diff), fit = fit, stan_data = stan_data) %>% 
        set_names(roc_param) %>% 
        map(map_if, ~ !is_null(.x), summarize_roc) %>%
        as_tibble(), 
      cluster_rep_return_dist = map2(cluster_rep_return_dist, stan_data, ~ {
        if (!is_null(.x)) {
          mutate(.x, iter_data = map(iter_data, ~ mutate(.x, iter_est = iter_est * .y), sd(.y$analysis_data$cluster.dist.to.pot))) %>% 
            summarize_roc()
        }
      })
      # sim_delta = map2(fit, stan_data, extract_sim_delta),
    )

  any_structural_models = any(dist_fit_data$model_type == "structural")
  if (any_structural_models) {
    ## Ed edits because Anne wants rep return vs control  
    cluster_rep_return_dist_control = dist_fit_data %>%
      select(cluster_rep_return_dist) %>%
      unnest(cols = c(cluster_rep_return_dist)) %>%
      filter(assigned_treatment == "control")

    cluster_rep_return_te_df = dist_fit_data %>%
        select(cluster_rep_return_dist) %>%
        unnest(cols = c(cluster_rep_return_dist))  %>%
        # filter(assigned_treatment != "control")  %>%
        unnest(iter_data) %>%
        left_join(
          cluster_rep_return_dist_control %>% 
            select(roc_distance_index, iter_data) %>%
            unnest(iter_data) %>%
            rename(iter_est_right = iter_est),
          by = c("roc_distance_index", "iter_id")
        ) %>%
        mutate( 
          iter_est_te = iter_est - iter_est_right
        ) %>%
        nest(iter_data = c(iter_id, iter_est_te, iter_est, iter_est_right)) %>%
        select(-contains("per"), -mean_est) %>%
        mutate(
          mean_est = map_dbl(iter_data, ~ mean(.$iter_est_te)),
          takeup_quantiles = map(iter_data, quantilize_est, iter_est_te, wide = TRUE, quant_probs = c(quant_probs), na.rm = TRUE)
        ) %>% 
        unnest(takeup_quantiles)

    cluster_rep_return_te_df %>%
      select(-iter_data) %>%
      write_csv(file.path(script_options$output_path, str_interp("processed_rep_return_dist_fit${fit_version}.csv")))


  }


}

dist_fit_data %<>%
  mutate(
    cluster_cf_cutoff = map(cluster_cf_cutoff, ~ {
      mutate(.x, quants = map(iter_data, quantilize_est, prob, quant_probs = quant_probs, na.rm = TRUE)) %>% 
        select(!iter_data) %>% 
        unnest(quants)
    })
  ) %>% 
  select(!c(ends_with("_dist_cost"), any_of("structural_cluster_benefit")))

if (!script_options$keep_fit) {
  dist_fit_data %<>% 
    select(!fit)
}

## Combine models using stacking -------------------------------------------

if (has_name(dist_fit_data, "stacking_weight")) {
  # The two functions below are used to combined multiple takeup levels, from different models, into a single (stacked) takeup level.
  
  dist_fit_data %<>% 
    add_row(
      model = factor("STACKED"), 
      model_type = factor("combined"),
      fit_type = factor("fit", levels = c("fit", "prior-predict")),
      est_takeup_level = list(
        reduce2(.$est_takeup_level, .$stacking_weight, level_stack_reducer, .init = tibble()) %>% 
          mutate(
            mean_est = map_dbl(iter_data, ~ mean(.$iter_prop_takeup)),
            takeup_quantiles = map(iter_data, quantilize_est, iter_prop_takeup, wide = TRUE, quant_probs = c(quant_probs))
          ) %>% 
          unnest(takeup_quantiles)
      ),
    ) %>%  
    add_row(
      model = factor("STRUCTURAL_STACKED"), 
      model_type = factor("structural"),
      fit_type = factor("fit", levels = c("fit", "prior-predict")),
      est_takeup_level = list(
        reduce2(.$est_takeup_level, .$stacking_weight_by_type, level_stack_reducer, .init = tibble()) %>% 
          mutate(
            mean_est = map_dbl(iter_data, ~ mean(.$iter_prop_takeup)),
            takeup_quantiles = map(iter_data, quantilize_est, iter_prop_takeup, wide = TRUE, quant_probs = c(quant_probs))
          ) %>% 
          unnest(takeup_quantiles)
      ),
    ) 
}

dist_fit_data %<>%
  inner_join(select(model_info, model, model_name), by = "model") %>% 
  mutate(model = factor(model, levels = model_info$model)) %>% 
  arrange(model)


## ATE ---------------------------------------------------------------------

# Select all the estimate differences that need to be calculated
ate_combo <- dist_fit_data %>% 
  # filter(fct_match(model_type, "structural")) %>% 
  slice(1) %$% 
  select(est_takeup_level[[1]], mu_assigned_treatment:assigned_dist_group) %>% {
    bind_cols(rename_all(., str_c, "_left"), rename_all(., str_c, "_right"))
  } %>% {
    bind_rows(
      expand(., crossing(!!!syms(names(.)))) %>%
        filter(mu_assigned_treatment_left != mu_assigned_treatment_right |
                 assigned_treatment_left != assigned_treatment_right |
                 (fct_match(assigned_dist_group_left, "far") & fct_match(assigned_dist_group_right, "close"))),
      select(., -contains("dist_group")) %>% 
        expand(crossing(!!!syms(names(.)))) %>%
        filter(mu_assigned_treatment_left != mu_assigned_treatment_right |
                 assigned_treatment_left != assigned_treatment_right) 
    )
  }




dist_fit_data %<>% 
  mutate(
    # Calculate treatment effects
    est_takeup_te = est_takeup_level %>% 
      map(select, mu_assigned_treatment:assigned_dist_group, iter_data) %>%
      calculate_ate(model_type),

    # calculate differences in TEs across arms but within same distance type
    est_takeup_te_diff = est_takeup_level %>%
      map(select, mu_assigned_treatment:assigned_dist_group, iter_data) %>%
      calculate_ate_diff(model_type),

    # est_age_group_takeup_te = est_age_group_takeup_level %>% 
    #   map_if(fct_match(model_type, "reduced form"), select, mu_assigned_treatment:age_group, iter_data) %>%
    #   calculate_ate(model_type, other_ate_join_col = "age_group"),
   
    # Calculate treatment effects based on distance 
    est_takeup_dist_te = est_takeup_te %>% calculate_dist_ate(model_type), 
    # est_age_group_takeup_dist_te = est_age_group_takeup_te %>% calculate_dist_ate(model_type, other_ate_join_col = "age_group"), 
    
    across(c(
      est_takeup_level, 
      est_takeup_te_diff
    # est_age_group_takeup_level
    ), ~ map_if(.x, ~ !is_null(.x), mutate, iter_data = map(iter_data, as_tibble)))  
  )

save(dist_fit_data, file = file.path(script_options$output_path, str_interp("processed_dist_fit${fit_version}.RData")))

dist_fit_data %<>% 
  mutate(across(where(is_list), map_if, ~ is_tibble(.x) && has_name(.x, "iter_data"), ~ select(.x, !iter_data)))

save(dist_fit_data, file = file.path(script_options$output_path, str_interp("processed_dist_fit${fit_version}_lite.RData")))

plan(sequential)

cat(str_glue("Post processing completed [version {fit_version}]\n\n"))




##### Checks ####
# dist_fit_env = new.env()
# with_env = function(f, e = parent.frame()) {
#     stopifnot(is.function(f))
#     environment(f) = e
#     f
# }

# load_df_function = function(x){
#     load(file.path("temp-data", str_interp("processed_dist_fit${x}.RData")))
#     return(dist_fit_data)
# }


# dist_fit_data_71 = with_env(load_df_function, dist_fit_env)(71) 

# n_nan_71 = dist_fit_data_71 %>%
#   filter(model == "STRUCTURAL_LINEAR_U_SHOCKS") %>%
#   select(fit_type, model, wtp_results)  %>%
#   unnest(wtp_results) %>%
#   slice(1) %>%
#   unnest(iter_data) %>%
#   summarise(
#     frac_nan = sum(is.nan(iter_est)), 
#     frac_not_nan = sum(!is.nan(iter_est)), 
#     frac_zero = sum(iter_est == 0, na.rm = TRUE), 
#     frac_not_zero = sum(iter_est != 0, na.rm = TRUE), 
#     n_iter = n()
#   ) %>% 
#   mutate(across(contains("frac"), ~.x/n_iter)) %>%
#   mutate(
#     fit_version = 71
#   )

# n_nan_75 = dist_fit_data %>%
#   select(fit) %>%
#   unnest() %>%
#   filter(str_detect(variable, r"{^(prob_prefer_calendar|(strata|hyper)_wtp_mu)}")) %>% 
#   tidyr::extract(variable, c("variable", "index"), r"{([^\[]+)(?:\[(\d+)\])?}", convert = TRUE) %>%
#   slice(1) %>%
#   unnest(iter_data) %>%
#   summarise(
#     frac_nan = sum(is.nan(iter_est)), 
#     frac_not_nan = sum(!is.nan(iter_est)), 
#     frac_zero = sum(iter_est == 0, na.rm = TRUE), 
#     frac_not_zero = sum(iter_est != 0, na.rm = TRUE), 
#     n_iter = n()
#   ) %>% 
#   mutate(across(contains("frac"), ~.x/n_iter)) %>%
#   mutate(
#     fit_version = 75
#   )

# bind_rows(
#   n_nan_71,
#   n_nan_75
# )

# dist_fit_data = dist_fit_data %>%
#   mutate(
#     wtp_results = map(fit, get_wtp_results)
#   )

# dist_fit_data_75 %>%
#   filter(model_type == "structural") %>%
#   select(fit_type, model, wtp_results) %>%
#   unnest(wtp_results) %>%
#   filter(variable == "prob_prefer_calendar") %>%
#   ggplot(aes(
#     x = index, 
#     y = per_0.5, 
#     ymin = per_0.05, 
#     ymax = per_0.95
#   )) +
#   geom_pointrange() +
#   facet_wrap(~fit_type)

# get_wtp_results <- function(wtp_draws) {
#   wtp_draws %>% 
#     filter(str_detect(variable, r"{^(prob_prefer_calendar|(strata|hyper)_wtp_mu)}")) %>% 
#     tidyr::extract(variable, c("variable", "index"), r"{([^\[]+)(?:\[(\d+)\])?}", convert = TRUE) %>% 
#     mutate(
#       quants = map(iter_data, quantilize_est, iter_est, quant_probs = c(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95), na.rm = TRUE),
#       mean_est = map_dbl(iter_data, ~ mean(.x$iter_value, na.rm = TRUE)),
#     ) %>% 
#     unnest(quants)
# }

# wtp_test = dist_fit_data %>%
#   select(fit) %>%
#   unnest() %>%
#   filter(str_detect(variable, r"{^(prob_prefer_calendar|(strata|hyper)_wtp_mu)}")) %>% 
#   tidyr::extract(variable, c("variable", "index"), r"{([^\[]+)(?:\[(\d+)\])?}", convert = TRUE) %>%
#   mutate(
#     mean_est = map_dbl(iter_data, mean, na.rm = TRUE), 
#     conf.low = map_dbl(iter_data, quantile, 0.05, na.rm = TRUE), 
#     conf.high = map_dbl(iter_data, quantile, 0.95, na.rm = TRUE)
#   )


# dist_fit_data %>%
#   select(fit) %>%
#   unnest() %>%
#   filter(str_detect(variable, r"{^(prob_prefer_calendar|(strata|hyper)_wtp_mu)}")) %>% 
#   tidyr::extract(variable, c("variable", "index"), r"{([^\[]+)(?:\[(\d+)\])?}", convert = TRUE) %>%
#   slice(1) %>%
#   unnest(iter_data) %>%
#   filter(iter_est == 0)


# cmdstan_fit_obj = as_cmdstan_fit(
#   str_glue("data/stan_analysis_data/dist_fit{script_options$fit_version}_STRUCTURAL_LINEAR_U_SHOCKS-1.csv"))

# library(tidybayes)
# library(posterior)

# prefer_cal_draws = gather_rvars(
#   cmdstan_fit_obj,
#   prob_prefer_calendar[index]
# )

# rm(cmdstan_fit_obj)
# gc()

# prefer_cal_draws %>%
#   # filter(index == 1) %>%
#   unnest_rvars() %>%
#   ggplot(aes(
#     x = .value, 
#     fill = factor(index)
#   )) +
#   geom_histogram(bins = 60) +
#   guides(fill = "none") +
#   labs(
#     title = "Prob Prefer Calendar, Posterior Draws", 
#     subtitle = "Fill just denotes different monetary values"
#   ) +
#   theme_bw()
  
# ggsave(
#   "temp-plots/tmp.png", 
#   width = 8,
#   height = 6, 
#   dpi = 500
# )


# stop()

