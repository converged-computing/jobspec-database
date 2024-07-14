library(cmdstanr)
library(purrr)
library(loo)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)

datasets_file <- args[[1]]
dataset_iter <- args[[2]]
dataset_iter <- as.numeric(dataset_iter)
exec <- args[[3]]

datasets <- readRDS(datasets_file)

current_data <- datasets[[dataset_iter]]

compute_loo_elpd <- function(model_fit) {
  loo_obj <- model_fit$loo()
  loo_obj$estimates["elpd_loo", "Estimate"]
}

compute_test_elpd <- function(model_fit) {
  test_log_lik <- model_fit$draws("log_lik_test")
  elpd_model <- elpd(test_log_lik)
  sum(elpd_model$pointwise[,"elpd"])
}

# define model-fitting method
fit_candidate_model <- function(k, exec, data, n, n_test) {
  stan_data <- list(N_train = n,
                    N_test = n_test,
                    d = 2,
                    x_train = as.matrix(data$train)[, paste0("x", c(0, k))],
                    x_test = as.matrix(data$test)[, paste0("x", c(0, k))],
                    y_train = data$train$y,
                    y_test = data$test$y)
  model_fit <- exec$sample(data = stan_data,
                           chains = 4,
                           parallel_chains = 4,
                           refresh = 0)
  return(model_fit)
}

fit_all_models <- function(exec_file, data) {
  n <- as.numeric(data$n)
  n_test <- as.numeric(data$n_test)
  sigma <- as.numeric(data$sigma)
  snr <- as.numeric(data$snr)
  K <- as.numeric(data$K)
  beta_delta <- data$beta_delta
  iter <- data$rep_id
  
  # build model
  exec <- cmdstan_model(exe_file = exec_file)
  
  # fit all models
  print("fitting baseline model ...")
  baseline_stan_data <- list(N_train = n,
                             N_test = n_test,
                             d = 1,
                             x_train = as.matrix(data$train$x0),
                             x_test = as.matrix(data$test$x0),
                             y_train = data$train$y,
                             y_test = data$test$y)
  baseline_model <- exec$sample(data = baseline_stan_data,
                                chains = 4,
                                parallel_chains = 4,
                                refresh = 0)
  print("baseline model done.")
  print("fitting candidate models ...")
  fitted_models <- 1:(K - 1) |> 
    map(\(k) fit_candidate_model(k, exec, data, n, n_test))
  print("candidate models done.")
  
  # compute the difference between all models and the baseline model
  print("computing elpd ...")
  loo_elpds <- fitted_models |>
    map(\(Ma_fit) compute_loo_elpd(Ma_fit))
  test_elpds <- fitted_models |>
    map(\(Ma_fit) compute_test_elpd(Ma_fit))
  baseline_loo_elpd <- compute_loo_elpd(baseline_model)
  baseline_test_elpd <- compute_test_elpd(baseline_model)
  print("elpd done.")
  
  # build dataframe of results
  out <- cbind(data.frame(sapply(loo_elpds,c)),
               data.frame(sapply(test_elpds,c)),
               data.frame(sapply(baseline_loo_elpd,c)),
               data.frame(sapply(baseline_test_elpd,c)))
  names(out) <- c("loo_elpd", "test_elpd", "baseline_loo_elpd", "baseline_test_elpd")
  out$model <- 1:(K - 1)
  out$K <- K
  out$n <- n
  out$n_test <- n_test
  out$sigma <- sigma
  out$snr <- snr
  out$iter <- iter
  out$beta <- beta_delta
  return(out)
}

# fit all models and compute stats
out <- fit_all_models(exec, current_data)

# save results
K <- as.numeric(current_data$K)
print("writing results ...")
output_file <- paste0("all_irrelevant_results_","K",K,
                      "_iter", dataset_iter)
write.csv(out, file = paste0("data/results/all-irrelevant/", output_file, ".csv"), 
          row.names = FALSE)
print("writing done.")

