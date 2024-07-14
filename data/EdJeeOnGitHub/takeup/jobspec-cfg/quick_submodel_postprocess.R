#!/usr/bin/Rscript
script_options <- docopt::docopt(
  stringr::str_glue(
"Usage:
  quick_submodel_postprocess.R <fit-version> [options] [<chain>...]
  
Options:
  --input-path=<path>  Path to find results [default: {file.path('data', 'stan_analysis_data')}]
  --output-path=<path>  Path to find results [default: temp-data]
  --model=<model>  Which model to postprocess
  --prior  Postprocess the prior predictive
  
  "), 
  args = if (interactive()) "
  95
  --output-path=temp-data
  --model=STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_FOB
  1 2 
  " else commandArgs(trailingOnly = TRUE)
)

library(tidyverse)
library(cmdstanr)
library(posterior)
library(tidybayes)



fit_type_str = if_else(script_options$prior, "prior", "fit")
if (length(script_options$chain) > 1) {
  chain_str = str_glue("{min(script_options$chain)}-{max(script_options$chain)}")
} else {
  chain_str = script_options$chain
}
source("quick_postprocess_functions.R")
##
dist_idx_mapper = tibble(
  dist_treat_idx = 1:8,
  treatment = rep(c("control", "ink", "calendar", "bracelet"), 2),
  dist_group = rep(c("close", "far"), each = 4)
) %>%
  mutate(
    treatment = factor(treatment, levels = c("bracelet", "calendar", "ink", "control")) %>% fct_rev,
    treatment = fct_relabel(treatment, str_to_title)
  )

prob_draws_raw = load_param_draws(
    fit_version = script_options$fit_version,
    model = script_options$model,
    chain = script_options$chain,
    prior_predictive = script_options$prior,
    input_path = script_options$input_path,
    prob_1ord[dist_treat_idx],
    prob_2ord[dist_treat_idx]
)

prob_draws = prob_draws_raw %>%
    left_join(dist_idx_mapper, by = "dist_treat_idx") 

prob_draws = bind_rows(
  prob_draws,
  prob_draws %>%
    group_by(fit_version, model, fit_type, treatment) %>%
    summarise(
      across(where(is_rvar), rvar_mean)
    ) %>%
    mutate(dist_group = "combined")
)

prob_draws %>%
  pivot_longer(where(is_rvar), names_to = "variable") %>%
  saveRDS(
    file.path(
      script_options$output_path,
      str_glue(
        "rvar_processed_dist_{fit_type_str}{script_options$fit_version}_belief_probs_{script_options$model}_{chain_str}.rds"
      )
    ) 
  )


cluster_treatment_map = distinct(analysis_data, assigned_treatment, assigned_dist_group) %>% 
  arrange(assigned_dist_group, assigned_treatment) # We must arrange by distance first

beliefs_ate_pairs <- cluster_treatment_map %>% 
  mutate(treatment_id = seq(n())) %>% {
      bind_rows(
        left_join(., filter(., fct_match(assigned_treatment, "control")), by = c("assigned_dist_group"), suffix = c("", "_control")) %>% 
          filter(assigned_treatment != assigned_treatment_control) %>% 
          select(treatment_id, treatment_id_control),
        
        left_join(., filter(., fct_match(assigned_dist_group, "close")), by = c("assigned_treatment"), suffix = c("", "_control")) %>% 
          filter(assigned_dist_group != assigned_dist_group_control) %>% 
          select(treatment_id, treatment_id_control),
      )
} %>%
  arrange(treatment_id, treatment_id_control) 



belief_ate_idx_mapper = beliefs_ate_pairs %>%
    mutate(belief_ate_idx = seq(n())) %>%
    right_join(mutate(cluster_treatment_map, treatment_id = seq(n())), ., by = c("treatment_id")) %>%
    right_join(mutate(cluster_treatment_map, treatment_id = seq(n())), ., by = c("treatment_id" = "treatment_id_control"), suffix = c("_right", "_left"))  %>%
    filter(assigned_treatment_right == "control") %>%
    filter(assigned_treatment_left != "control") %>%
    filter(assigned_dist_group_left == assigned_dist_group_right) %>%
    select(
        belief_ate_idx,
        treatment = assigned_treatment_left, 
        dist_group = assigned_dist_group_left
    )  %>%
    mutate(
      treatment = factor(treatment, levels = c("bracelet", "calendar", "ink", "control")) %>% fct_rev,
      treatment = fct_relabel(treatment, str_to_title)
    )


ate_draws_raw = load_param_draws(
    fit_version = script_options$fit_version,
    model = script_options$model,
    chain = script_options$chain,
    prior_predictive = script_options$prior,
    input_path = script_options$input_path,
    ate_1ord[belief_ate_idx],
    ate_2ord[belief_ate_idx]
)


ate_draws_cf = ate_draws_raw %>%
    right_join(
        belief_ate_idx_mapper,
        by = "belief_ate_idx"
    )  %>%
    select(-belief_ate_idx)
ate_draws_combined = ate_draws_cf %>%
  group_by(
    model,
    fit_version,
    fit_type,
    treatment) %>%
  summarise(
    across(where(is_rvar), rvar_mean),
    .groups = "drop"
  )  %>%
  mutate(dist_group = "combined")

ate_draws = bind_rows(
  ate_draws_cf,
  ate_draws_combined
)

ate_draws %>%
  pivot_longer(where(is_rvar), names_to = "variable") %>%
  saveRDS(
    file.path(
      script_options$output_path,
      str_glue(
        "rvar_processed_dist_{fit_type_str}{script_options$fit_version}_belief_ates_{script_options$model}_{chain_str}.rds"
      )
    ) 
  )


## WTP
wtp_draws_raw = load_param_draws(
    fit_version = script_options$fit_version,
    model = script_options$model,
    chain = script_options$chain,
    prior_predictive = script_options$prior,
    input_path = script_options$input_path,
    prob_prefer_calendar[pref_value_diff_idx],
    hyper_wtp_mu
)

wtp_draws_raw %>%
  pivot_longer(where(is_rvar), names_to = "variable") %>%
  saveRDS(
    file.path(
      script_options$output_path,
      str_glue(
        "rvar_processed_dist_{fit_type_str}{script_options$fit_version}_wtp_params_{script_options$model}_{chain_str}.rds"
      )
    ) 
  )



