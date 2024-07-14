#!/usr/bin/Rscript
print(commandArgs(trailingOnly = TRUE))
script_options = docopt::docopt(
    stringr::str_glue("Usage:
    predict-takeup-for-optim.R <fit-version> <private-benefit-z> <visibility-z> [options] [ --from-csv | --to-csv ]

    Takes posterior fits and calculates estimated takeup for a given distance.


    Options:
        --from-csv  Load parameter draws from a csv file
        --to-csv   Load stanfit object and write draws to csv
        --input-path=<path>  Path to find results [default: {file.path('data', 'stan_analysis_data')}]
        --output-path=<path>  Path to find results [default: {file.path('optim', 'data')}]
        --output-name=<output-name>  Prepended to output file
        --pred-distance  Estimate takeup on a grid from 0 to 5km
        --dist-cutoff=<dist-cutoff>  Don't estimate takeup past this cutoff (set takeup to 0) [default: 2500]
        --rep-cutoff=<rep-cutoff>  Don't let reputational returns increase past this point. [default: Inf]
        --num-post-draws=<num-post-draws>  Number of posterior draws to use [default: 200]
        --num-cores=<num-cores>  Number of cores to use [default: 8]
        --type-lb=<type-lb>  Lower bound of type distribution [default: -Inf]
        --type-ub=<type-ub>  Upper bound of type distribution [default: Inf]
        --model=<model>  Which model to use [default: STRUCTURAL_LINEAR_U_SHOCKS]
        --fit-rf  Estimate a simple brms reduced form model with distance entering continuously 
        --run-estimation  Whether th run estimation stuff
        --data-input-path=<data-input-path>  Where village and PoT data is stored [default: {file.path('optim', 'data')}]
        --data-input-name=<data-input-name>  Filename of village and PoT data [default: full-experiment-4-extra-pots.rds]
        --suppress-reputation  Suppress reputational returns
        --single-chain  Only use first chain for draws (useful for debugging) 

        --static-signal-pm  Policy maker only estimates v* across some distance and doesn't realise v* a function of distance
        --static-signal-distance=<static-signal-distance>  Distance over which PM estimates v* in meters [default: NA]
        --fit-type=<fit-type>  Which fit type to use - prior predictive or posterior draws [default: fit] 

        --fix-mu-at-1  Fix mu at 1 (for B&T model) when running --pred-distance N.B. This is currently bugged and sets x dot lambda > 1

        --fix-mu-distance=<fix-mu-distance>  Distance to fix mu at in meters. 
        --fix-delta-distance=<fix-delta-distance>  Distance to fix delta at in meters

    "),
    args = if (interactive()) "
                            86
                            ink
                            ink
                            --output-name=TEST-mu-fix-0-cutoff-b-ink-mu-ink-STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP
                            --from-csv
                            --num-post-draws=200
                            --rep-cutoff=Inf
                            --dist-cutoff=3500
                            --type-lb=-Inf
                            --type-ub=Inf
                            --num-cores=12
                            --model=STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP
                            --data-input-name=full-many-pots-experiment.rds
                            --single-chain

                            --run-estimation
                            --static-signal-distance=Inf
                            --fix-delta-distance=0
                              " 
           else commandArgs(trailingOnly = TRUE)
)


# Loading functions
run_estimation = script_options$run_estimation
set.seed(19484)

library(posterior)
library(tidyverse)
library(tidybayes)
library(broom)
library(rstan)
library(sf)
library(nleqslv)
library(cmdstanr)
library(econometr)
library(furrr)

source(file.path("optim", "optim-functions.R"))
source(file.path("rct-design-fieldwork", "takeup_rct_assign_clusters.R"))
source(file.path("analysis_util.R"))
source(file.path( "dist_structural_util.R"))

mu_rep_type = switch(
    script_options$model,
    STRUCTURAL_LINEAR_U_SHOCKS = 0,
    STRUCTURAL_LINEAR_U_SHOCKS_LOG_MU_REP = 1,
    STRUCTURAL_LINEAR_U_SHOCKS_LINEAR_MU_REP = 2,
    STRUCTURAL_LINEAR_U_SHOCKS_NO_REP = 3,
    STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP = 4
)

cat(
    str_glue(
        "model: {script_options$model} \n
        mu_rep_type: {mu_rep_type} \n,
        fit_type: {script_options$fit_type} \n
        "
    )
)


source(file.path("multilvlr", "multilvlr_util.R"))
source(file.path("stan_models", "stan_owen_t.R"))

stan_owen_t = exposed_stan_func$stan_owen_t


script_options = script_options %>%
    modify_at(c("dist_cutoff", 
                "rep_cutoff",
                "num_post_draws", 
                "num_cores",
                "type_lb",
                "type_ub", 
                "static_signal_distance",
                "fix_mu_distance",
                "fix_delta_distance"
                ), as.numeric)

if (length(script_options$fix_mu_distance) == 0) script_options$fix_mu_distance = NULL
if (length(script_options$fix_delta_distance) == 0) script_options$fix_delta_distance = NULL

fit_version = script_options$fit_version

script_options$bounds = c(script_options$type_lb, script_options$type_ub)

append_output = if (!is.null(script_options$output_name)) {
    paste0("-", script_options$output_name)
} else {
    ""
}


models_we_want = c(
    script_options$model
)




if (script_options$to_csv) {
    if (script_options$single_chain) {
        struct_model_files = fs::dir_ls(
            script_options$input_path, 
            regex = str_glue("dist_{script_options$fit_type}{fit_version}_{script_options$model}-1\\.csv"))
    } else {
        struct_model_files = fs::dir_ls(
            script_options$input_path, 
            regex = str_glue("dist_{script_options$fit_type}{fit_version}_{script_options$model}-.*csv"))
    }
    
    struct_model_fit = as_cmdstan_fit(struct_model_files)
    # N.B. this is very hardcoded for vanilla structural model
    struct_param_draws = struct_model_fit %>%
        gather_draws(
            beta[k],
            dist_beta_v[j],
            centered_cluster_beta_1ord[j, k],
            centered_cluster_dist_beta_1ord[j, k],
            base_mu_rep, 
            total_error_sd[k],
            u_sd[k]
        )
    # model_fit is huge so get rid of it asap
    rm(struct_model_fit) 
    gc()
    # N.B. we use input path for `to_csv` output and reserve output_path for 
    # output later in script
    struct_param_draws %>%
        write_csv(file.path(script_options$input_path, str_interp("param_posterior_draws_dist_fit${fit_version}_${script_options$model}.csv") ))
    ## Now reduced form  - we actually fit rf model continuous in distance here
    # since main analysis RF just uses close/far
    if (script_options$fit_rf) {

        # Fit Loading
        load(file.path("temp-data", str_interp("processed_dist_fit${fit_version}_lite.RData")))

        # N.B. This uses old postprocessing output so won't currently run from 
        # scratch - just load analysis_data here and run from that will work however.
        # TODO
        stan_data = (dist_fit_data %>%
            slice(1) %>% # just take first model as only using for analysis_data
            pull(stan_data))[[1]]
        rf_analysis_data = stan_data$analysis_data

        library(brms)
        brms_rf_analysis_data = rf_analysis_data %>%
            select(
                dewormed,
                assigned.treatment, 
                cluster.id, 
                county, 
                cluster.dist.to.pot
            ) %>%
            mutate(sd_dist = cluster.dist.to.pot/sd(cluster.dist.to.pot))


        options(mc.cores = script_options$num_cores)
        rf_model_fit = brm(
            data = brms_rf_analysis_data,
            dewormed ~ sd_dist*assigned.treatment, 
            bernoulli(link = "probit")
        )

        rf_model_fit %>%
            saveRDS(
                file.path(
                    script_options$input_path, 
                    str_interp("param_posterior_draws_dist_fit${fit_version}_REDUCED_FORM_NO_RESTRICT.rds") ))

    } 
} 

if (script_options$from_csv) {
    struct_param_draws = read_csv(
        file.path(
            script_options$input_path, 
            str_interp(
                "param_posterior_draws_dist_${script_options$fit_type}${fit_version}_${script_options$model}.csv"
            )
        )
    )
    if (!script_options$fit_rf) {
        rf_model_fit = read_rds(
            file.path(
                script_options$input_path, 
                str_interp(
                    "param_posterior_draws_dist_fit${fit_version}_REDUCED_FORM_NO_RESTRICT.rds"
                )
            )
        )
    }
}

## B(z,d):
# \beta is treatment effect
# dist_beta_v is distance cost


## Create estimated demand functions
max_draw = max(struct_param_draws$.draw)
draw_treat_grid = expand.grid(
    draw = 1:min(max_draw, script_options$num_post_draws)
    # treatment = script_options$treatment
)

dist_data = read_rds(
    file.path(
        script_options$data_input_path,
        script_options$data_input_name
    )
)

brms_long_distance_mat = dist_data$brms_long_distance_mat
long_distance_mat = dist_data$long_distance_mat
sd_of_dist = dist_data$sd_of_dist
sd_of_dist %>%
    saveRDS("temp-data/sd_of_dist.rds")

if (script_options$static_signal_pm == TRUE | !is.null(script_options$fix_delta_distance)) {
    # generate dynamic (non-static) prediction functions to calculate static signal 
    # at required cutoff
    true_pred_functions = map(
        draw_treat_grid$draw,
        ~extract_params(
            param_draws = struct_param_draws,
            private_benefit_treatment = script_options$private_benefit_z,
            visibility_treatment = script_options$visibility_z,
            draw_id = .x,
            dist_sd = sd_of_dist,
            j_id = 1,
            rep_cutoff = script_options$rep_cutoff,
            dist_cutoff = script_options$dist_cutoff, 
            bounds = script_options$bounds,
            mu_rep_type = mu_rep_type,
            suppress_reputation = script_options$suppress_reputation, 
            static_signal = NA,
            fix_mu_at_1 = FALSE,
            fix_mu_distance = script_options$fix_mu_distance,
            static_delta_v_star = NA
        ) %>% find_pred_takeup()
    )


    if (script_options$static_signal_pm == TRUE) {
        static_pred_outputs = map(
            true_pred_functions, 
            ~.x(script_options$static_signal_distance)
        )
    } 

    if (!is.null(script_options$fix_delta_distance)) {
        static_pred_outputs = map(
            true_pred_functions,
            ~.x(script_options$fix_delta_distance)
        )
    }

    delta_v_stars = map_dbl(static_pred_outputs, "delta_v_star")
    mu_reps = map_dbl(static_pred_outputs, "mu_rep")

    draw_treat_grid$static_signal_value = mu_reps*delta_v_stars
    draw_treat_grid$static_delta_v_star = delta_v_stars
}  else {
    draw_treat_grid$static_signal_value = NA
    draw_treat_grid$static_delta_v_star = NA
}


if (run_estimation == TRUE){

pred_functions = pmap(
    list(
        draw_treat_grid$draw,
        draw_treat_grid$static_signal_value,
        draw_treat_grid$static_delta_v_star
    ),
    ~extract_params(
        param_draws = struct_param_draws,
        private_benefit_treatment = script_options$private_benefit_z,
        visibility_treatment = script_options$visibility_z,
        draw_id = ..1,
        dist_sd = sd_of_dist,
        j_id = 1,
        rep_cutoff = script_options$rep_cutoff,
        dist_cutoff = script_options$dist_cutoff, 
        bounds = script_options$bounds,
        mu_rep_type = mu_rep_type,
        suppress_reputation = script_options$suppress_reputation, 
        static_signal = ..2,
        fix_mu_at_1 = FALSE,
        fix_mu_distance = script_options$fix_mu_distance,
        static_delta_v_star = ..3
    ) %>% find_pred_takeup()
)

if (script_options$fit_rf) {

    rf_pred_df = map_dfr(
        script_options$private_benefit_z,
        ~{
            add_epred_draws(
                brms_long_distance_mat %>% mutate(assigned.treatment = .x),
                rf_model_fit,
                ndraws = script_options$num_post_draws
            )
        }
    ) %>%
        rename(
            treatment = assigned.treatment, 
            draw = .draw, 
            demand = .epred
        )  %>%
        ungroup() 

}


subset_long_distance_mat = long_distance_mat %>%
    filter(dist < script_options$dist_cutoff)


print(str_glue("Running {length(pred_functions)} pred functions"))

tictoc::tic()
plan(multicore, workers = script_options$num_cores)
pred_df = future_imap_dfr(
    pred_functions,
    ~{
        subset_long_distance_mat %>%
            mutate( 
                as_tibble(.x(dist)),
                draw = draw_treat_grid[.y, "draw"]
            )
    },
    .options = furrr_options(
        seed = TRUE
    ),
    .progress = TRUE
)
tictoc::toc()

cutoff_pred_df = future_map_dfr(
    draw_treat_grid$draw,
    ~{
        long_distance_mat %>%
            filter(dist >= script_options$dist_cutoff) %>%
            mutate(
                pred_takeup = NA,
                linear_pred = NA,
                b = NA, 
                mu_rep = NA, 
                delta_v_star = NA, 
                v_star = NA, 
                draw = .x
            )
    }
)

pred_df = bind_rows(
    pred_df,
    cutoff_pred_df
)


structural_demand_df = long_distance_mat %>%
    left_join(
        pred_df %>% 
            select(-dist, -dist_km),
        by = c("index_i", "index_j")
    ) %>%
    mutate(
        across(
            pred_takeup, replace_na, 0.0
        )) %>%
  rename(
    village_i = index_i, 
    pot_j = index_j, 
    demand = pred_takeup) %>%
  group_by(village_i) %>%
  mutate(closest_pot = dist == min(dist)) %>%
  ungroup() %>%
  mutate( 
    model = script_options$model 
  )

if (script_options$fit_rf) {

    rf_demand_df = rf_pred_df %>%
        rename(
            village_i = index_i, 
            pot_j = index_j, 
        ) %>%
        group_by(village_i, closest_pot = dist == min(dist)) %>%
        ungroup() %>%
        mutate(
            model = "REDUCED_FORM_NO_RESTRICT"
        ) %>%
        mutate(
            demand = if_else(
                dist > script_options$dist_cutoff,
                0,
                demand
            )
        ) %>%
        select(-`.row`, -`.chain`, -`.iteration`)
    rf_demand_df$private_benefit_z = script_options$private_benefit_z
    rf_demand_df$visibility_z = script_options$visibility_z
}




structural_demand_df$private_benefit_z = script_options$private_benefit_z
structural_demand_df$visibility_z = script_options$visibility_z






structural_demand_df %>%
    write_csv(
        file.path(
            script_options$output_path, 
            str_interp("pred-demand-dist-${script_options$fit_type}${fit_version}${append_output}.csv")
        )
    )

if (script_options$fit_rf) {

    rf_demand_df %>%
        write_csv(
            file.path(
                script_options$output_path, 
                str_interp("pred-demand-dist-fit${fit_version}-REDUCED_FORM_NO_RESTRICT.csv")
            )
        )
}

        

}

if (script_options$pred_distance) {

plan(multicore, workers = script_options$num_cores)
pred_functions = map2(
    draw_treat_grid$draw,
    draw_treat_grid$static_signal_value,
    ~extract_params(
        param_draws = struct_param_draws,
        private_benefit_treatment = script_options$private_benefit_z,
        visibility_treatment = script_options$visibility_z,
        draw_id = .x,
        dist_sd = sd_of_dist,
        j_id = 1,
        rep_cutoff = script_options$rep_cutoff,
        dist_cutoff = script_options$dist_cutoff, 
        bounds = script_options$bounds,
        mu_rep_type = mu_rep_type,
        suppress_reputation = script_options$suppress_reputation, 
        static_signal = .y,
        fix_mu_at_1 = script_options$fix_mu_at_1
    ) %>% find_pred_takeup()
)

dist_df = tibble(
        dist = seq(from = 0, to = 2500, by = 100)
) 

sm_df = future_map_dfr(
    pred_functions,
    ~{
        dist_df %>%
            mutate(
                as_tibble(.x(dist))
            ) 
    }, 
    .id = "draw", 
    .progress = TRUE,
    .options = furrr_options(seed = TRUE)
)

sm_df = sm_df %>%
    mutate(
        sm = (-delta + mu_rep_deriv*delta_v_star) / (1 + mu_rep*delta_v_star_deriv)
    ) 

fix_mu_str = if_else(script_options$fix_mu_at_1, "fix-mu-", "")

sm_df %>%
    write_csv(
        file.path(
            script_options$output_path, 
            str_interp("pred-social-multiplier-${fix_mu_str}${script_options$fit_type}${fit_version}${append_output}.csv")
        )
    )


 }
