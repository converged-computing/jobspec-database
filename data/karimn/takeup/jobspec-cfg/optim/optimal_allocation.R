#!/usr/bin/Rscript
script_options <- docopt::docopt(
    stringr::str_glue("Usage:
        optimal_allocation.R  [options] [--min-cost | --max-takeup]

        Options:
          --min-cost  Flag to minimise programme cost for a given takeup level.
          --max-takeup  Flag to maximise vaccine takeup for a given cost level.
          --optim-type=<optim-type>  What sort of optimisation problem to solve
          --constraint-type=<constraint-type>  Aggregate or individual village level constraints
          --target-constraint=<target-constraint>  Amount of takeup/budget constraint depending on [--min-cost|--max-takeup]  [default: 0.5]
          --input-path=<input-path>  Path where input data is stored.
          --output-path=<output-path>  Path where output should be saved.
          --output-filename=<output-filename>  Output filename.
          --village-input-filename=<village-input-filename>  Index and location of each village - a csv path.
          --pot-input-filename=<pot-input-filename>  Index and location of each PoT - a csv path. 
          --demand-input-filename=<demand-input-filename>  Estimated demand for every village i, PoT j pair.  
          --num-cores=<num-cores>  Number of cores to use in parallel [default: 8]
          --time-limit=<time-limit>  GLPK solver time limit [default: 1000]
          --posterior-median  Use median demand rather than solve across all draws
          --welfare-function=<welfare-function>  Welfare function to use for takeup [default: identity]
          --data-input-path=<data-input-path>  Where village and PoT data is stored [default: {file.path('optim', 'data')}]
          --data-input-name=<data-input-name>  Filename of village and PoT data [default: full-experiment-4-extra-pots.rds]
          --solver=<solver>  MILP solver [default: glpk]
          --distance-constraint=<distance-constraint>  Maximum distance a villager can be forced to walk by PM [default: 3500]
"),
  args = if (interactive()) "
                                --num-cores=12 \
                                --min-cost  \
                                --constraint-type=agg \
                                --target-constraint=summ-agg-identity-experiment-target-constraint.csv \
                                --output-path=optim/data/STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP/agg-full-many-pots \
                                --input-path=optim/data/STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP/agg-full-many-pots  \
                                --data-input-name=full-many-pots-experiment.rds
                                --data-input-path=optim/data
                                --time-limit=10000 \
                                --output-filename=TEST-cutoff-b-control-mu-control-STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP \
                                --demand-input-filename=pred-demand-dist-fit86-cutoff-b-control-mu-control-STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP.csv
                                --welfare-function=identity
                                --solver=gurobi
                                --posterior-median
                                --distance-constraint=4500

                             " else commandArgs(trailingOnly = TRUE)
) 

                                # --welfare-function=log
library(tidyverse)
library(data.table)
library(ompr)
library(ompr.roi)
library(sf)
library(ROI)
library(Matrix)
library(slam)

if (script_options$solver == "gurobi") {
    # library(gurobi)
    library(ROI.plugin.gurobi)
} else {
    library(ROI.plugin.glpk)
}

source("optim/optim-functions.R")




expected_functions = c(
  "identity",
  "log"
)
if (!(script_options$welfare_function %in% expected_functions)) {
  warning(str_glue(
    "{script_options$welfare_function} isn't an expected function."
  ))
}
swf = eval(parse(text = script_options$welfare_function))


numeric_options = c(
  "dry_run_subsidy",
  "num_cores",
  "time_limit", 
  "distance_constraint"
)

script_options = script_options %>%
  modify_at(numeric_options, as.numeric)


stat_type = if_else(script_options$posterior_median, "median", "post-draws")


################################################################################
# Loading Data #################################################################
################################################################################

wgs.84 <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"
dist_data = read_rds(
  file.path(
    script_options$data_input_path,
    script_options$data_input_name
  )
)

village_data = dist_data$village_df
pot_data = dist_data$pot_df

demand_input_path = file.path(script_options$input_path, 
                              script_options$demand_input_filename)

initial_demand_data = read_csv(demand_input_path) %>%
  as.data.table() 

n_max = Inf
m_max = Inf
initial_demand_data = initial_demand_data[village_i <= n_max & pot_j <= m_max]
pot_data = pot_data %>%
  filter(id <= m_max)
village_data = village_data %>%
  filter(id <= n_max)



if (script_options$posterior_median) {
  initial_demand_data = initial_demand_data[
    , 
    .(demand = median(demand), dist = unique(dist)), 
    .(village_i,
      pot_j, 
      private_benefit_z, 
      visibility_z,
      model)]
}
# kinda weird, have to take this after finding median due to a bug
# increasing function so should be fine??
initial_demand_data[, util := swf(demand)]

n = nrow(village_data)
m = nrow(pot_data)
data = list(
  n = n,
  m = m,
  pot_locations = pot_data,
  village_locations = village_data
) 

target_df = read_csv(
  file.path(
    script_options$input_path,
    script_options$target_constraint
  ) 
)  %>% 
  rename(target_demand = demand)

library(furrr)
plan(multicore, workers = script_options$num_cores)

if (script_options$constraint_type == "indiv") {
  target_optim = target_df %>% 
    arrange(village_i) %>%
    mutate(target_util = swf(target_demand)) %>%
    group_by(village_i) %>%
    summarise(target_util = mean(target_util)) %>%
    pull(target_util)
}


# Have precomputed SWF in this case so just load that csv
if (script_options$constraint_type == "agg") {
  summ_target_df = read_csv(
    file.path(
      script_options$input_path,
      paste0(script_options$target_constraint)
    )
  )

  target_optim = mean(summ_target_df$social_welfare)

}


if (script_options$constraint_type == "indiv") {
  if (nrow(target_df) !=  nrow(village_data)) {stop("Target mismatch.")}

  n_infeasible = initial_demand_data %>%
    left_join(target_df, by = "village_i") %>%
    group_by(village_i) %>%
    summarise(
      infeasible = all(util < target_util)
    ) %>%
    filter(infeasible == TRUE) %>%
    nrow()

}

if (script_options$constraint_type == "agg") {
  n_infeasible = -Inf # hard to tell in this case a priori?
}

if (n_infeasible > 0) {
  stop("Infeasible allocation desired.")
}

demand_data = initial_demand_data %>%
  nest(demand_data = -any_of(c("draw", 
                               "private_benefit_z", 
                               "visibility_z", 
                               "model")))
# typically good to remove this as it's huge and the optimisation needs RAM if 
# using full posterior
rm(initial_demand_data)
gc()

baseline_constraints = create_base_constraints(
  data$n,
  data$m
)

sorted_distance = demand_data %>%
  slice(1) %>%
  unnest(demand_data) %>%
  arrange(village_i, pot_j) %>%
  pull(dist)

baseline_constraints = add_distance_constraints(
  distance = sorted_distance,
  distance_constraint = script_options$distance_constraint ,
  baseline_constraints = baseline_constraints, 
  n = data$n, 
  m = data$m
)

# Have to set -Inf utility from very 0 takeup in no-cutoff to -1e10 since 
# optimiser gets upset by negative infinity
v_neg_value = demand_data %>%
  head(1) %>%
  unnest(demand_data) %>%
  filter(demand > 0) %>%
  summarise(min_util = min(util)) %>%
  pull() 

v_neg_value = -abs(v_neg_value)

demand_data = demand_data %>%
  mutate(
    demand_data = map(
      demand_data,
      ~{
        mutate(.x, util = ifelse(!is.finite(util), v_neg_value*300, util))
      }
      )
  )


target_optim

if (script_options$solver == "glpk") {
  control_args = list(tm_limit = script_options$time_limit)
} else {
  control_args = NULL
}
tictoc::tic()
tidy_output = demand_data %>%
  mutate(
    optim_problem = future_map(
      demand_data,
    ~{
        define_model(
          takeup = .x,
          data = data,
          target_takeup = target_optim,
          baseline_constraints = baseline_constraints,
          constraint_type = script_options$constraint_type
        )
    },
      .progress = TRUE,
      .options = furrr_options(seed = TRUE)
    ))
tidy_output = tidy_output %>%
  mutate(
    optim_fit = map(
      optim_problem,
      ~ROI_solve(
        .x, 
        solver = script_options$solver,
        verbose = TRUE,
        control = control_args
      ),
      .progress = TRUE,
      .options = furrr_options(seed = TRUE)
    )
  )   

tidy_output = tidy_output %>%
  mutate(
    model_output = future_map2( 
      optim_fit, 
      demand_data,
      ~clean_solution(.x, data = data, takeup = .y ) %>%
        mutate(target_optim = target_optim), 
        .progress = TRUE, 
        .options = furrr_options(seed = TRUE)
    )
  )
tictoc::toc()

#' Create Best Solution if OA is Infeasible
#' 
#'  Use all PoTs and assign to closest when overall allocation is infeasible
create_infeasible_soln = function(status_code, demand_data, model_output) {
  if (status_code == 0) {
    return(model_output)
  } 
  subset_demand_data = demand_data %>%
    group_by(
      village_i
    ) %>%
    filter(
      dist == min(dist)
    ) %>%
    slice(1) %>%
    ungroup()

    infeasible_optimal_df = subset_demand_data %>%
      group_by(village_i, pot_j) %>%
      summarise(
        demand = median(demand), 
        dist = unique(dist)
      ) %>%
      left_join(
        village_data %>%
          select(id, village_lon = lon, village_lat = lat), 
        by = c("village_i" = "id")
      ) %>%
      left_join(
          pot_data %>%
              select(id, pot_lon = lon, pot_lat = lat), 
          by = c("pot_j" = "id")
      ) %>%
      rename(
        i = village_i, 
        j = pot_j
      ) %>%
      mutate(target_optim = target_optim) %>%
      ungroup()
    return(infeasible_optimal_df) 

}

tidy_output = tidy_output %>%
  mutate(
    status_code = map_dbl(optim_fit, ~.x$status$code)
  )


tidy_output = tidy_output %>%
  mutate(
    model_output = pmap(
      list(
        status_code, 
        demand_data,
        model_output
      ), 
      ~create_infeasible_soln(
        ..1, 
        ..2, 
        ..3
      )
    )
  )

summ_output = tidy_output %>% 
  head(1) %>%
  unnest(model_output) %>%
  summarise(
    util = sum(swf(demand)),
    mean_demand = mean(demand), 
    min_demand = min(demand), 
    n_pot = n_distinct(j))

if (script_options$constraint_type == "agg") {
  if (summ_output$util < target_optim) {
    warning("Allocation utility below target utility.")
  }
}

plot_res = FALSE
if (plot_res) {


tidy_output %>% 
  unnest(model_output) %>%
  select(dist) %>%
  ggplot() +
  geom_histogram(aes(x = dist), colour = "white") +
  theme_bw() +
  labs(
    x = "Distance Walked (m)", 
    title = "Distance Walked, No Cutoff - B Bracelet Mu Bracelet"
  )

tidy_output %>% 
  unnest(model_output) %>%
  select(demand) %>%
  ggplot() +
  geom_histogram(aes(x = demand))

tidy_output %>% 
  unnest(model_output) %>%
  select(util) %>%
  ggplot() +
  geom_histogram(aes(x = util))


tidy_output %>% 
  unnest(model_output) %>%
  ggplot() +
  geom_point(aes(
    x = pot_lon,
    y = pot_lat
  ), shape = 17, size = 3) +
  geom_point(aes(
    x = village_lon,
    y = village_lat
  )) +
  geom_point(
    data = . %>%
      filter(demand == 0),
    aes(
    x = village_lon,
    y = village_lat
  ), size = 10, colour = "hotpink") +
  geom_segment(
    data = . %>%
      filter(demand > 0),
    aes(
    x = village_lon,
    y = village_lat,
    xend = pot_lon, 
    yend = pot_lat
  ))

  ## Double checking model
  summ_output
}

# #########################################
# stop()




output_path = file.path(
  script_options$output_path, 
  str_glue("{script_options$output_filename}-{stat_type}-optimal-allocation.rds"))



saveRDS(
  tidy_output,
  output_path
)




#### Manual Stuff Checks
# takeup = demand_data %>%
#   unnest(demand_data) %>%
#   arrange(village_i, pot_j) %>%
#   pull(demand)
# # stop()
# # script_options$constraint_type = "agg"
# # script_options$target_constraint = rep(0.32, n)

# problem_constraints = add_takeup_constraints(
#   takeup = takeup,
#   takeup_target = script_options$target_constraint,
#   n = data$n,
#   m = data$m,
#   baseline_constraints = baseline_constraints,
#   constraint_type = script_options$constraint_type
# )


# cm = simple_triplet_matrix(
#   i = problem_constraints$constraint_matrix$i,
#   j = problem_constraints$constraint_matrix$j,
#   v = problem_constraints$constraint_matrix$v,
#   nrow = problem_constraints$constraint_matrix$nrow,
#   ncol = problem_constraints$constraint_matrix$ncol
# )

# # # data$n*data$m + 10


# best_pot_vill_pairs = demand_data %>%
#   unnest(demand_data) %>%
#   arrange(village_i, pot_j) %>%
#   group_by(village_i) %>%
#   filter(demand == max(demand)) %>%
#   select(pot_j, village_i, demand) %>%
#   mutate(
#     x_on_index = find_x_index(
#       n = n,
#       m = m, 
#       i = village_i, 
#       j = pot_j 
#       )
#   )

# # best_pot_vill_pairs

# # best_pot_vill_pairs

# y_on = matrix(
#   data = 0,
#   nrow = m,
#   ncol = 1
# )
# y_on[best_pot_vill_pairs$pot_j, 1] = 1

# x_on = matrix(
#   data = 0, 
#   nrow = data$n*data$m, 
#   ncol = 1
# )

# x_on[best_pot_vill_pairs$x_on_index, 1] = 1


# x_mat = rbind(
#   y_on, 
#   x_on
# )
# ## ompr soln
# # ompr_demand = (demand_data %>%
# #     pull(demand_data))[[1]] %>%
# #     group_by(old_village_i = village_i) %>%
# #     mutate(village_i = cur_group_id()) %>%
# #     as.data.table()
# # # OMPR
# # baseline_model = define_baseline_MIPModel(data)
# # fit_ompr_model = define_and_solve_model(
# #   baseline_model = baseline_model,
# #   data = data,
# #   demand_data = ompr_demand,
# #   optim_type = "min_cost",
# #   target_constraint = script_options$target_constraint
# #    )


# # ompr_soln = tidy_output %>%
# #   mutate(soln = map(optim_fit, ~.x$solution)) %>%
# #   unnest(soln) %>%
# #   pull()


# # cbind(
# #   x_mat,
# #   ompr_soln)
# # look into structure of constraints
# # dim(cm)
# # dim(x_mat)
# # names(x_mat) = names(ompr_soln)
# ed = as.matrix(cm) %*% x_mat
# # rownames(ed) = names(ompr_soln)
# ed



  # model_soln = (tidy_output %>%
  #   pull(optim_fit))[[1]]$solution

  # takeup = demand_data %>%
  #   unnest(demand_data) %>%
  #   arrange(village_i, pot_j) %>%
  #   pull(demand)

  # problem_constraints = add_takeup_constraints(
  #   takeup = takeup,
  #   takeup_target = script_options$target_constraint,
  #   n = data$n,
  #   m = data$m,
  #   baseline_constraints = baseline_constraints,
  #   constraint_type = script_options$constraint_type
  # )


  # cm = simple_triplet_matrix(
  #   i = problem_constraints$constraint_matrix$i,
  #   j = problem_constraints$constraint_matrix$j,
  #   v = problem_constraints$constraint_matrix$v,
  #   nrow = problem_constraints$constraint_matrix$nrow,
  #   ncol = problem_constraints$constraint_matrix$ncol
  # )
  # stop()
  # length(model_soln)

  # manual_fit = as.matrix(cm) %*% model_soln

  # manual_fit[length()]
