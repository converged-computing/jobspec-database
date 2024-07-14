#!/usr/bin/Rscript
script_options <- docopt::docopt(
  stringr::str_glue(
"Usage:
  quick_roc_postprocess.R <fit-version> [options] [<chain>...]
  
Options:
  --input-path=<path>  Path to find results [default: {file.path('data', 'stan_analysis_data')}]
  --output-path=<path>  Path to find results [default: temp-data]
  --model=<model>  Which model to postprocess
  --prior  Postprocess the prior predictive
  --cluster-roc
  --fix-cluster-roc
  --cluster-takeup-prop
  --cluster-rep-return-dist
  --sm
  
  "), 
  args = if (interactive()) "
  95
  --output-path=temp-data
  --model=STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_FOB
  --sm
  1
  " else commandArgs(trailingOnly = TRUE)
)
library(tidyverse)
library(cmdstanr)
library(posterior)
library(tidybayes)


source("quick_postprocess_functions.R")

# N.B. treat_idx (the second idx, is the mu (signalling) idx)
treat_idx_mapper = tibble(
  treat_idx = 1:4,
  treatment = c("control", "ink", "calendar", "bracelet")
) %>%
  mutate(
    treatment = factor(treatment, levels = c("bracelet", "calendar", "ink", "control")) %>% fct_rev,
    treatment = fct_relabel(treatment, str_to_title)

  )
dist_idx_mapper = tibble(
  dist_treat_idx = 1:8,
  dist_treatment = rep(c("control", "ink", "calendar", "bracelet"), 2),
  dist_group = rep(c("close", "far"), each = 4)
) %>%
  mutate(dist_treatment = factor(dist_treatment, levels = c("bracelet", "calendar", "ink", "control")))
cluster_mapper = analysis_data %>%
  select(
    cluster_id,
    assigned_treatment,
    assigned_dist_group
  ) %>% unique()

roc_dist_idx_mapper = tibble(
    roc_distance = seq(0, 5000, 100)) %>% 
    mutate(roc_dist_idx = seq(n()))


        
fit_type_str = if_else(script_options$prior, "prior", "fit")
if (length(script_options$chain) > 1) {
  chain_str = str_glue("{min(script_options$chain)}-{max(script_options$chain)}")
} else {
  chain_str = script_options$chain
}

if (script_options$cluster_roc) {

    cluster_roc_draws_raw = load_param_draws(
      fit_version = script_options$fit_version,
      model = script_options$model,
      chain = script_options$chain,
      prior_predictive = script_options$prior,
      input_path = script_options$input_path,
      cluster_roc[roc_dist_idx, cluster_idx, treat_idx],
      cluster_roc_no_vis[roc_dist_idx, cluster_idx, treat_idx]
    )

    cluster_roc_draws = cluster_roc_draws_raw %>%
        left_join(roc_dist_idx_mapper, by = "roc_dist_idx") %>%
        left_join(treat_idx_mapper, by = "treat_idx")

    roc_draws = cluster_roc_draws  %>%
        group_by(
            model,
            fit_version,
            fit_type,
            treatment, 
            roc_distance
        ) %>%
        summarise(
            cluster_roc = rvar_mean(cluster_roc),
            cluster_roc_no_vis = rvar_mean(cluster_roc_no_vis)
        )

  roc_draws %>%
    pivot_longer(where(is_rvar), names_to = "variable") %>%
    saveRDS(
      file.path(
        script_options$output_path,
        str_glue(
          "rvar_processed_dist_{fit_type_str}{script_options$fit_version}_roc_draws_{script_options$model}_{chain_str}.rds"
        )
      ) 
    )



        
    summ_roc_draws = roc_draws %>%
        pivot_longer(where(is_rvar), names_to = "variable") %>%
        median_qi(value) %>%
        to_broom_names()
        
    summ_roc_draws %>%
      saveRDS(
        file.path(
          script_options$output_path,
          str_glue(
            "rvar_processed_dist_{fit_type_str}{script_options$fit_version}_roc_summ_{script_options$model}_{chain_str}.rds"
          )
        ) 
      )

}


if (script_options$fix_cluster_roc) {
    cluster_fix_roc_draws_raw = load_param_draws(
        fit_version = script_options$fit_version,
        model = script_options$model,
        chain = script_options$chain,
        prior_predictive = script_options$prior,
        input_path = script_options$input_path,
        cluster_roc_no_vis[roc_dist_idx, cluster_idx, treat_idx],
        cluster_roc_delta_deriv[roc_dist_idx, cluster_idx, treat_idx]
    ) 

    cluster_fix_roc_draws = cluster_fix_roc_draws_raw  %>%
        left_join(roc_dist_idx_mapper, by = "roc_dist_idx") %>%
        left_join(treat_idx_mapper, by = "treat_idx")

    cluster_fix_roc_draws = cluster_fix_roc_draws %>%
        mutate(
            fix_cluster_roc = cluster_roc_no_vis / (1 + cluster_roc_delta_deriv)
        )

    fix_roc_draws = cluster_fix_roc_draws  %>%
        group_by(
            model,
            fit_version,
            fit_type,
            treatment, 
            roc_distance
        ) %>%
        summarise(
            fix_cluster_roc = rvar_mean(fix_cluster_roc)
        )

    summ_fix_roc_draws = fix_roc_draws %>%        
        median_qi(fix_cluster_roc) %>%
        to_broom_names()


    summ_fix_roc_draws %>%
      filter(roc_distance <= 2500) %>%
      ggplot(aes(
        x = roc_distance,
        y = fix_cluster_roc,
        colour = treatment
      )) +
      geom_point()

}



if (script_options$cluster_takeup_prop) {

    cluster_prop_draws_raw = load_param_draws(
      fit_version = script_options$fit_version,
      model = script_options$model,
      chain = script_options$chain,
      prior_predictive = script_options$prior,
      input_path = script_options$input_path,
      cluster_takeup_prop[roc_dist_idx, cluster_idx, treat_idx]
    )

    cluster_prop_draws = cluster_prop_draws_raw %>%
        left_join(roc_dist_idx_mapper, by = "roc_dist_idx") %>%
        left_join(treat_idx_mapper, by = "treat_idx")

    prop_draws = cluster_prop_draws  %>%
        group_by(
            model,
            fit_version,
            fit_type,
            treatment, 
            roc_distance
        ) %>%
        summarise(
            cluster_takeup_prop = rvar_mean(cluster_takeup_prop)
        )

  prop_draws %>%
    pivot_longer(where(is_rvar), names_to = "variable") %>%
    saveRDS(
      file.path(
        script_options$output_path,
        str_glue(
          "rvar_processed_dist_{fit_type_str}{script_options$fit_version}_prop_draws_{script_options$model}_{chain_str}.rds"
        )
      ) 
    )
        
    summ_prop_draws = prop_draws %>%
        median_qi(cluster_takeup_prop) %>%
        to_broom_names()
        
    summ_prop_draws %>%
      saveRDS(
        file.path(
          script_options$output_path,
          str_glue(
            "rvar_processed_dist_{fit_type_str}{script_options$fit_version}_prop_summ_{script_options$model}_{chain_str}.rds"
          )
        ) 
      )
}


if (script_options$cluster_rep_return_dist) {
    cluster_rep_return_dist_raw = load_param_draws(
      fit_version = script_options$fit_version,
      model = script_options$model,
      chain = script_options$chain,
      prior_predictive = script_options$prior,
      input_path = script_options$input_path,
      cluster_rep_return_dist[roc_dist_idx, cluster_idx, treat_idx]
    )

    cluster_rep_return_dist_draws = cluster_rep_return_dist_raw %>%
        left_join(roc_dist_idx_mapper, by = "roc_dist_idx") %>%
        left_join(treat_idx_mapper, by = "treat_idx")

    rep_return_dist_draws = cluster_rep_return_dist_draws  %>%
        group_by(
            model,
            fit_version,
            fit_type,
            treatment, 
            roc_distance
        ) %>%
        summarise(
            cluster_rep_return_dist = rvar_mean(cluster_rep_return_dist*sd_of_dist)
        )

    rep_return_dist_draws = rep_return_dist_draws %>%
      group_by(
            model,
            fit_version,
            fit_type,
            roc_distance
      ) %>%
      mutate(
        cluster_rep_return_dist_te = cluster_rep_return_dist - cluster_rep_return_dist[treatment == "Control"]
      )

  rep_return_dist_draws %>%
    pivot_longer(where(is_rvar), names_to = "variable") %>%
    saveRDS(
      file.path(
        script_options$output_path,
        str_glue(
          "rvar_processed_dist_{fit_type_str}{script_options$fit_version}_rep_return_dist_draws_{script_options$model}_{chain_str}.rds"
        )
      ) 
    )

}

if (script_options$sm) {
    sm_raw = load_param_draws(
      fit_version = script_options$fit_version,
      model = script_options$model,
      chain = script_options$chain,
      prior_predictive = script_options$prior,
      input_path = script_options$input_path,
      cbind(
        cluster_social_multiplier,
        cluster_mu_rep,
        cluster_mu_rep_deriv,
        cluster_delta,
        cluster_delta_deriv
      )[roc_dist_idx, cluster_idx, treat_idx],
      dist_beta_v
    )




    sm_draws = sm_raw %>%
        left_join(roc_dist_idx_mapper, by = "roc_dist_idx") %>%
        left_join(treat_idx_mapper, by = "treat_idx")

    sm_draws_clean = sm_draws  %>%
        group_by(
            model,
            fit_version,
            fit_type,
            treatment, 
            roc_distance
        ) %>%
        summarise(
            sm = rvar_mean(cluster_social_multiplier),
            dist_beta_v = rvar_mean(dist_beta_v),
            sm_delta_part = rvar_mean(
              -dist_beta_v / (1 + cluster_mu_rep*cluster_delta_deriv)
            ),
            sm_mu_part = rvar_mean(
              cluster_mu_rep_deriv*cluster_delta / (1 + cluster_mu_rep*cluster_delta_deriv)
            )
        ) %>%
      mutate(across(
        c(sm, sm_delta_part, sm_mu_part),
        ~.x/dist_beta_v,
        .names = "{.col}_rescaled"
      )) 

  sm_draws_clean %>%
    pivot_longer(where(is_rvar), names_to = "variable") %>%
    saveRDS(
      file.path(
        script_options$output_path,
        str_glue(
          "rvar_processed_dist_{fit_type_str}{script_options$fit_version}_sm_draws_{script_options$model}_{chain_str}.rds"
        )
      ) 
    )


        
    summ_sm_draws = sm_draws_clean %>%
      pivot_longer(where(is_rvar), names_to = "variable") %>%
      median_qi(value) %>%
      to_broom_names()
        
    summ_sm_draws %>%
      saveRDS(
        file.path(
          script_options$output_path,
          str_glue(
            "rvar_processed_dist_{fit_type_str}{script_options$fit_version}_sm_summ_{script_options$model}_{chain_str}.rds"
          )
        ) 
      )

}