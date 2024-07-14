# Pipeline for method adapted from
# Pulliam, JRC, C van Schalkwyk, N Govender, A von Gottberg, C 
# Cohen, MJ Groome, J Dushoff, K Mlisana, and H Moultrie. (2022) Increased
# risk of SARS-CoV-2 reinfection associated with emergence of Omicron in
# South Africa. _Science_ <https://www.science.org/doi/10.1126/science.abn4947>
# 
# Repository: <https://github.com/jrcpulliam/reinfections>
# 


args = commandArgs(trailingOnly=TRUE)

results <- list()

#set i, the index in the parameter set
i<-strtoi(args[1])

method <- 1
write(paste0("set number", i),file="arrayjob_m1.txt",append=TRUE)

dir.create(paste0('sbv/raw_output'))
dir.create(paste0('sbv/raw_output/m', method))

load(file=paste0("sbv/method_", method, "_analysis/parameters.RData"))
data_source <- 'data/inf_for_sbv.RDS'
configpth <- paste0('sbv/method_',method,'_analysis/m',method,'_config_general.json')
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

lapply(required_files, load, envir = .GlobalEnv)

parameters.r <- save_params

attach(jsonlite::read_json(configpth))

results <- list()

write('running',file="arrayjob_m1.txt",append=TRUE) #comment to confirm that theres not a zombie node


ts <- generate_data(1, data_source, seed = seed_batch)
ts_adjusted <- ts[, c("date", "observed", "ma_tot", "cases" )]

#Run MCMC
output <- do.mcmc(mcmc$n_chains, ts_adjusted)


write('done mcmc',file="arrayjob_m1.txt",append=TRUE) #comment to confirm that theres not a zombie node

#Save posterior
lambda.post <- kappa.post <- numeric(0)
smpls <- mcmc$n_posterior / mcmc$n_chains #number of samples to take from each chain
niter <- mcmc$n_iter - mcmc$burnin  #number of iterations to take into account 
jump <- round(niter/smpls)
for(ii in 1:mcmc$n_chains){
  #need smpls number of samples from each chain out of niter samples
  lambda.post <- c(lambda.post, output$chains[[ii]][seq(1,mcmc$n_iter-mcmc$burnin,jump),1])
  kappa.post <- c(kappa.post, output$chains[[ii]][seq(1,mcmc$n_iter-mcmc$burnin,jump),2])
}

#5: Run simulations
set.seed(seed_batch+2023)
sim_reinf <- function(ii){
  tmp <- list(lambda = lambda.post[ii], kappa = kappa.post[ii])
  answer <- expected(parms = tmp, data = ts_adjusted, delta = cutoff)
  ex2 <- Reduce("+", answer)
  ex2 <- c(rep(0,90),ex2)
  return(rnbinom(length(ex2), size=1/kappa.post[ii], mu =c(0, diff(ex2))))
}

write('start sims',file="arrayjob_m1.txt",append=TRUE) #comment to confirm that theres not a zombie node

sims <- sapply(rep(1:mcmc$n_posterior, n_sims_per_param), sim_reinf)

write('end sims',file="arrayjob_m1.txt",append=TRUE) #comment to confirm that theres not a zombie node

#6: analysis
sri <- data.table(date = ts_adjusted$date, sims)
sri_long <- melt(sri, id.vars = 'date')
sri_long[, ma_val := frollmean(value, 7), variable]

eri_ma <- sri_long[, .(exp_reinf = median(ma_val, na.rm = TRUE)
                       , low_reinf = quantile(ma_val, 0.025, na.rm = TRUE)
                       , upp_reinf = quantile(ma_val, 0.975, na.rm = TRUE)), keyby = date]

print(paste0("Upper reinfections ", str(eri_ma$upp_reinf)))

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
gd$psrf <- gd$psrf[ -3,]

lambda_convergence <- gd$psrf[1]
kappa_convergence <- gd$psrf[2]

## calculate diagnostics for after wave split: 
eri_ma <- eri_ma[date > wave_split]
number_of_days_aw <- nrow(eri_ma)
days_diff <- ts[date > wave_split]$ma_reinf - eri_ma$upp_reinf

print(paste0("ts reinfections ", str(ts[date > wave_split]$ma_reinf)))

days_diff[days_diff<0] <- 0
days_diff[days_diff>0] <- 1
conseq_diff_aw <- frollsum(days_diff, 5, fill =0)
proportion_aw <- length(days_diff[days_diff==1])/number_of_days_aw
date_first_aw <- which(conseq_diff_aw==5)[1]

results <- list(pscale = parameters.r$pscale[i]
                , lambda_con = lambda_convergence
                , kappa_con = kappa_convergence
                , proportion = proportion
                , date_first = which(conseq_diff==5)[1]
                , proportion_after_wavesplit = proportion_aw
                , date_first_after_wavesplit = which(conseq_diff_aw==5)[1]
                , seed = seed_batch
)
saveRDS(results, file=paste0("sbv/raw_output/m",method,"/results_", i,".RDS"))
