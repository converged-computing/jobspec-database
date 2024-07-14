# Pipeline for method adapted from
# Pulliam, JRC, C van Schalkwyk, N Govender, A von Gottberg, C 
# Cohen, MJ Groome, J Dushoff, K Mlisana, and H Moultrie. (2022) Increased
# risk of SARS-CoV-2 reinfection associated with emergence of Omicron in
# South Africa. _Science_ <https://www.science.org/doi/10.1126/science.abn4947>
# 
# Repository: <https://github.com/jrcpulliam/reinfections>
# Adapted for third infections 

args = commandArgs(trailingOnly=TRUE)

results <- list()

#set i, the index in the parameter set
i<-strtoi(args[1])

if (!exists("i") | is.na(i))
  i <- 1

method <- 'third'

dir.create(paste0('sbv/raw_output'))
dir.create(paste0('sbv/raw_output/ml2third'))


load(file=paste0("sbv/third_infections/parameters.RData"))
data_source <- 'data/inf_for_sbv_v4.RDS'
configpth <- paste0('sbv/third_infections/config_third_infections.json')
settingspth <- 'utils/settings.RData'

load('utils/settings.RData')

# load required packages & files
library(data.table)
library(iterators)
library(foreach)
library(doParallel)
library(coda)
library(parallel)
library(dplyr)
library(ggplot2)

lapply(required_files_third_l2, load, envir = .GlobalEnv)

load('utils/mcmc_functions_l2.RData')

parameters.r <- save_params

attach(jsonlite::read_json(configpth))

wave_split <- omicron_date 

seed_arg <-strtoi(args[2])
if (!exists("seed_arg") | is.na(seed_arg)) {
  print("Keep seed batch from config -- do nothing")
} else {
  print("Change seed batch from args")
  seed_batch <- seed_arg
}

dir.create('sbv/third_infections/output')
target_mcmc <- paste0('sbv/third_infections/output/posterior_null_ouput_i_', i, '_seed_', seed_batch, '.RData' )


results <- list()



ts <- generate_data_third(data_source, seed = seed_batch)
ts_adjusted <- ts[, c("date", "observed", "ma_tot", "cases" )]

#Run MCMC

disease_params <- function(lambda = .000000015 ## hazard coefficient
                           , kappa = 0.1 ## dispersion (inverse)
                           , lambda2 = .000000018
) return(as.list(environment()))


output <- do.mcmc.l2(mcmc$n_chains, ts_adjusted)



#Save posterior
lambda.post <- kappa.post <- lambda2.post <-  numeric(0)
smpls <- mcmc$n_posterior / mcmc$n_chains #number of samples to take from each chain
niter <- mcmc$n_iter - mcmc$burnin  #number of iterations to take into account 
jump <- round(niter/smpls)
for(ii in 1:mcmc$n_chains){
  #need smpls number of samples from each chain out of niter samples
  lambda.post <- c(lambda.post, output$chains[[ii]][seq(1,mcmc$n_iter-mcmc$burnin,jump),1])
  kappa.post <- c(kappa.post, output$chains[[ii]][seq(1,mcmc$n_iter-mcmc$burnin,jump),2])
  lambda2.post <- c(lambda2.post, output$chains[[ii]][seq(1,mcmc$n_iter-mcmc$burnin,jump),3])
}


#save mcmc results
save(output, lambda.post, kappa.post, lambda2.post, file=target_mcmc)

#5: Run simulations
set.seed(seed_batch+2023)
sim_reinf <- function(ii){
  tmp <- list(lambda = lambda.post[ii], kappa = kappa.post[ii], lambda2 = lambda2.post[ii])
  answer <- expected_l2(parms = tmp, data = ts_adjusted, delta = cutoff)
  ex2 <- Reduce("+", answer)
  ex2 <- c(rep(0,90),ex2)
  return(rnbinom(length(ex2), size=1/kappa.post[ii], mu =c(0, diff(ex2))))
}



write('start sims',file="third_infections.txt",append=TRUE) #comment to confirm that theres not a zombie node

sims <- sapply(rep(1:mcmc$n_posterior, n_sims_per_param), sim_reinf)

write('end sims',file="third_infections.txt",append=TRUE) #comment to confirm that theres not a zombie node

#6: analysis
sri <- data.table(date = ts_adjusted$date, sims)
sri_long <- melt(sri, id.vars = 'date')
sri_long[, ma_val := frollmean(value, 7), variable]

eri_ma <- sri_long[, .(exp_reinf = median(ma_val, na.rm = TRUE)
                       , low_reinf = quantile(ma_val, 0.025, na.rm = TRUE)
                       , upp_reinf = quantile(ma_val, 0.975, na.rm = TRUE)), keyby = date]


eri_ma <- eri_ma[date > fit_through]


number_of_days <- nrow(eri_ma)

days_diff <- ts[date > fit_through]$ma_reinf - eri_ma$upp_reinf
days_diff[days_diff<0] <- 0
days_diff[days_diff>0] <- 1

conseq_diff <- frollsum(days_diff, 5, fill =0)
proportion <- length(days_diff[days_diff==1])/number_of_days
date_first <- which(conseq_diff==5)[1]

# Diagnostics 
gd <- gelman.diag(output$chains)
gd$psrf <- gd$psrf[ -4,]

lambda_convergence <- gd$psrf[1]
kappa_convergence <- gd$psrf[2]
lambda2_convergence <- gd$psrf[3]

# not calculating after wave split because omicron_date/wave_split < fit_through :) 

results <- list(pscale = parameters.r$pscale[i]
                , lambda_con = lambda_convergence
                , kappa_con = kappa_convergence
                , lambda_2_con = lambda2_convergence
                , proportion = proportion
                , date_first = which(conseq_diff==5)[1]
                , seed = seed_batch
)
saveRDS(results, file=paste0("sbv/raw_output/ml2third/results_", i,"_seed_", seed_batch, ".RDS"))
