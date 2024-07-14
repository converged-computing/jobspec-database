library(jsonlite)
library(dplyr)
library(tidyr)
library(arrow)
library(posterior)
library(future.apply)
plan(multisession)

source("R/ksc_results.r")

config <- jsonlite::read_json("configs/ksc.json")
tmp_data_location <- paste("simulation_output/", config$simulation_name, "/tmp", sep ="") 

file_metadata <- as.data.frame(list.files(tmp_data_location))
names(file_metadata) <- c("files")

file_metadata <- tidyr::separate(file_metadata, 
                     col = files, 
                     sep = "_", 
                     into = c("dataset", "mcmc_seed"),
                     remove = FALSE) %>% 
    tidyr::separate(., col = mcmc_seed, sep = "\\.", into = c("mcmc_seed", "parquet"), remove = TRUE) %>% 
    select(-parquet)

simulation_results <- function(file_number){
    results <- list()
    results[["all_chains"]] <- list()
    results[["one_chain"]] <- list()

    iteration_list <- list()
    iter_files <- file_metadata[file_metadata["dataset"] == file_number, ][["files"]]
    iter_files_path <- paste(tmp_data_location, "/", iter_files, sep = "")
    iteration_list <- lapply(iter_files_path, arrow::read_parquet, as_data_frame=TRUE)
    parameter_names <- names(iteration_list[[1]])

    all_chains = sapply(parameter_names, combine_chains, list_obj = iteration_list, simplify=FALSE)
    names(all_chains)[names(all_chains) == "sigma2"] <- "sigma_sqd"

    post_weights <- all_chains[['weights']] %>% 
            tibble::rowid_to_column("index") %>% 
            pivot_longer(cols = -c(index),  names_to = "chain", values_to = "weights")

    post_weights_ess <- post_weights %>% 
                            group_by(chain) %>% 
                            summarise(ess_weights=1/sum(weights^2))

    all_chains <- within(all_chains, rm(weights))
    
    one_chain = sapply(all_chains, one_chain_f, simplify=FALSE)

    # All chain diagnostics
    if(length(iteration_list)>1){
    all_chains_diagnostics <- sapply(all_chains, diagnostics, simplify=FALSE)
    } else{
    all_chains_diagnostics <- "Only one chain"
    }

    # One chain diagnostics
    ess_basic <- sapply(one_chain, posterior::ess_basic, USE.NAMES = TRUE)
    ess_bulk <- sapply(one_chain, posterior::ess_bulk, USE.NAMES = TRUE)
    ess_tail <- sapply(one_chain, posterior::ess_tail, USE.NAMES = TRUE)
    one_chain_diagnostic <- as.data.frame(cbind(ess_bulk, ess_tail, ess_basic))
    one_chain_diagnostic["variable"] <- rownames(one_chain_diagnostic)
    rownames(one_chain_diagnostic) <- NULL

    # Calculate ranks
    data_location <- here::here("data/simulated/sbc", config$data_location, paste(file_number, "json", sep = "."))
    prior_params <- jsonlite::read_json(data_location)

    if(length(iteration_list)>1){
    results[["all_chains"]][["agg_ranks"]] <- sapply(names(all_chains), rank_stats, draws = all_chains, prior_parameters = prior_params, USE.NAMES = TRUE)
    results[["all_chains"]][["weighted_ranks"]] <- sapply(names(all_chains), weighted_ranks, draws = all_chains, prior_parameters = prior_params, USE.NAMES = TRUE, posterior_weights = post_weights, one_chain = FALSE)
    }
    results[["one_chain"]][["agg_ranks"]] <- sapply(names(one_chain), rank_stats, draws = one_chain, prior_parameters = prior_params, USE.NAMES = TRUE)
    results[["one_chain"]][["weighted_ranks"]] <- sapply(names(all_chains), weighted_ranks, draws = all_chains, prior_parameters = prior_params, USE.NAMES = TRUE, posterior_weights = post_weights, one_chain = TRUE)
    
    # Append results
    results[["all_chains"]][["diagnostics"]] <- all_chains_diagnostics
    results[["one_chain"]][["diagnostics"]] <- one_chain_diagnostic
    results[["config"]] <- config
    results[["seed_files"]] <- file_metadata[file_metadata["dataset"] == file_number, ]
    results[["weights_ess"]] <- post_weights_ess

    saveRDS(results,
            file = here::here("simulation_output",
                config$simulation_name,
                "output",
                paste("seed_index",
                    file_number,
                    ".RDS",
                    sep = "_")))
}

datasets <- unique(file_metadata$dataset)
future_lapply(datasets, simulation_results, future.seed = NULL)
