library(cmdstanr)
library(loo)
library(dplyr)
stanfit <- function(fit) rstan::read_stan_csv(fit$output_files())

args <- commandArgs(trailingOnly = TRUE)

datasets_file <- args[[1]]
dataset_iter <- args[[2]]
dataset_batch <- args[[3]]
dataset_iter <- as.numeric(dataset_iter)
dataset_batch <- as.numeric(dataset_batch)
exec_file <- args[[4]]

datasets <- readRDS(datasets_file)

current_data <- datasets[[dataset_iter]]

compute_loo_elpd_difference <- function(Ma_fit, Mb_fit) {
  # compute LOO-CV elpd difference
  log_lik_Ma <- extract_log_lik(Ma_fit, merge_chains = FALSE)
  log_lik_Mb <- extract_log_lik(Mb_fit, merge_chains = FALSE)
  r_eff_Ma <- relative_eff(exp(log_lik_Ma))
  r_eff_Mb <- relative_eff(exp(log_lik_Mb))
  loo_Ma <- loo(log_lik_Ma, r_eff = r_eff_Ma, cores = 2)
  loo_Mb <- loo(log_lik_Mb, r_eff = r_eff_Mb, cores = 2)
  loo_elpd_diff <- sum(loo_Ma$pointwise[,"elpd_loo"] 
                       - loo_Mb$pointwise[,"elpd_loo"])
  return(loo_elpd_diff)
}

compute_test_elpd_difference <- function(Ma_fit, Mb_fit) {
  # compute test elpd difference
  test_log_lik_Ma <- extract_log_lik(Ma_fit, 
                                     parameter_name = "log_lik_test")
  test_log_lik_Mb <- extract_log_lik(Mb_fit, 
                                     parameter_name = "log_lik_test")
  elpd_Ma <- elpd(test_log_lik_Ma)
  elpd_Mb <- elpd(test_log_lik_Mb)
  elpd_diff <- sum(elpd_Ma$pointwise[,"elpd"] 
                   - elpd_Mb$pointwise[,"elpd"])
  return(elpd_diff)
}

experiment <- function(exec_file, data) {
  
  n_train <- n_test <- as.numeric(data$n)
  beta_delta <- as.numeric(data$beta_delta)
  
  # define stan data
  stan_data_Ma <- list(
    N_train = n_train,
    N_test = n_test,
    d = 2,
    x_train = data$train[,c("x1","x2")],
    y_train = data$train$y,
    x_test = data$test[,c("x1","x2")],
    y_test = data$test$y
  )
  stan_data_Mb <- list(
    N_train = n_train,
    N_test = n_test,
    d = 3,
    x_train = data$train[,c("x1","x2","x3")],
    y_train = data$train$y,
    x_test = data$test[,c("x1","x2","x3")],
    y_test = data$test$y
  )
  
  # retrieve models
  exec <- cmdstan_model(exe_file = exec_file)
  
  # compile and fit the models
  Ma_fit <- stanfit(
    exec$sample(data = stan_data_Ma,
                chains = 4,
                parallel_chains = 4,
                refresh = 0)
  )
  Mb_fit <- stanfit(
    exec$sample(data = stan_data_Mb,
                chains = 4,
                parallel_chains = 4,
                refresh = 0)
  )
  
  # extract mean posterior value of beta_delta
  posterior_mean <- posterior::summarise_draws(Mb_fit) %>% 
    filter(variable == "beta[3]") %>% select(mean)
  
  # compute elpd difference
  loo_elpd_diff <- compute_loo_elpd_difference(Ma_fit, Mb_fit)
  test_elpd_diff <- compute_test_elpd_difference(Ma_fit, Mb_fit)
  
  # save results
  out <- data.frame(beta_delta = beta_delta, 
                    loo_elpd_diff = loo_elpd_diff, 
                    elpd_diff = test_elpd_diff,
                    posterior_mean = posterior_mean)
  return(out)
}

# run experiment
results <- experiment(exec_file, current_data)

# save the results
write.csv(results, file = paste0("data/results/joint-predictive/jointpred_iter",dataset_iter,"_batch",dataset_batch,".csv"),
          row.names = FALSE)


list.files('./data/results/joint-predictive/', pattern = "jointpred_iter*", full.names = TRUE) %>% 
  lapply(read_csv) %>% 
  bind_rows %>%
  write_csv('data/results/joint-predictive/jointpred_all.csv')

list.files('./data/results/many-irrelevant/', pattern = "many_models_results*", full.names = TRUE) %>% 
  lapply(read_csv) %>% 
  bind_rows %>%
  write_csv('data/results/many-irrelevant/many_models_all.csv')
