#!/usr/bin/Rscript
script_options <- docopt::docopt(
    stringr::str_glue("Usage:
        create-village-target.R <model-fit> [options]

        Options:
          --input-path=<input-path>  Path where input data is stored.
          --output-path=<output-path>  Path where output should be saved.
          --output-basename=<output-basename>  Output basename.
          --num-cores=<num-cores>  Number of cores [default: 4]
          --util-function=<util-function>  Utility function [default: log]
          --force
          --assign-closest
"),
  args = if (interactive()) "
                            pred-demand-dist-fit86-cutoff-b-control-mu-control-STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP.csv \
                            --input-path=optim/data/STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP/agg-log-full-many-pots \
                            --output-path=optim/data/STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP/agg-log-full-many-pots \
                            --output-basename=target-closest-cutoff-b-control-mu-control-STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP \
                            --util-function=identity \
                            --assign-closest
                             " else commandArgs(trailingOnly = TRUE)
) 


library(tidyverse)

script_options$num_cores = as.numeric(script_options$num_cores)


util_function = eval(parse(text = script_options$util_function))

demand_df = read_csv(
    file.path(
        script_options$input_path,
        script_options$model_fit
    )
)

target_demand_df = demand_df %>%
    group_by(village_i) %>%
    # adjustment for 3 villages which don't seem to have anyone close by
    mutate(all_far = all(dist > 2500 )) %>%
    ungroup() %>%
    filter(dist < 2500 | all_far == TRUE)


summ_output_file = file.path(
            script_options$output_path,
            paste0("summ-agg-log-", script_options$output_basename, ".csv")
        )


# Compute expected social welfare if planner had randomly chosen villages like 
# in the experiment.
# slightly computationally costly so we skip if the file already exists, 
# if --force is passed we fit anyway.
if (!file.exists(summ_output_file) | script_options$force) {
    if (script_options$assign_closest) {
        summ_target_df = target_demand_df %>%
            group_by(village_i) %>%
            filter(dist == min(dist)) %>%
            group_by(village_i, draw) %>%
            arrange(dist) %>%
            slice(1) %>%
            mutate(
                util = util_function(demand)
            ) %>%
            group_by(draw) %>%
            summarise(
                social_welfare = sum(util), 
                mean_takeup = mean(demand)
            ) 
        summ_target_df %>%
            write_csv(
                summ_output_file
            )
   } else {
        library(furrr)
        plan(multicore, workers = script_options$num_cores)
        summ_target_df = target_demand_df  %>%
            group_by(village_i, draw) %>%
            sample_n(20, replace = TRUE)  %>%
            mutate(random_village_draw = 1:n()) %>%
            group_by(draw, random_village_draw) %>%
            group_nest() %>%
            mutate(
            social_welfare = future_map(
                data, 
                ~{
                mutate(
                    .x,
                util = log(demand)
                ) %>%
                summarise(social_welfare = sum(util), mean_takeup = mean(demand)) 
                }
            )
            )



        summ_target_df %>%
            select(-data) %>%
            unnest(social_welfare) %>%
            write_csv(
                summ_output_file
            )


   }

}

target_demand_df %>%
    write_csv(
        file.path(
            script_options$output_path, 
            paste0(script_options$output_basename, ".csv")
        )
    )

