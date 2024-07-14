# jonashaslbeck@gmail.com; May 2020

# Get iter from command line ...
iter <- commandArgs(trailingOnly=TRUE)
iter <- as.numeric(iter)

# -----------------------------------------------------------------------------
# ---------- Load Packages ----------------------------------------------------
# -----------------------------------------------------------------------------

# Parallelize
library(foreach)
library(doParallel)

# Data Generation
# library(mlVAR) # we use a function in aux_functions.R instead

source("aux_functions_sim.R")

 
# -----------------------------------------------------------------------------
# ---------- Load Models ------------------------------------------------------
# -----------------------------------------------------------------------------

# l_CM <- readRDS(file = "files/Models_74cells.RDS") # local
l_CM <- readRDS(file = "Models_74cells.RDS")

# Fixed things
p <- 6
means <- rep(0, p) # mean vector
E <- diag(p) # error covariance matrix

# n variation
n_seq <- 8:500

# -----------------------------------------------------------------------------
# ---------- Loop -------------------------------------------------------------
# -----------------------------------------------------------------------------

nClust <- 12 # We run the n-variations in parallel

# Fire up foreach
cl <- makeCluster(nClust)
registerDoParallel(cl)

# start clustered loops
outlist <- foreach(cell=1:74,
                   # .packages=c("mlVAR"),
                   .verbose=FALSE,
                   .export=c("l_CM", "fSimulate", "fEstimate", "means", "E", "n_seq")) %dopar% {

                     # ----- Create Output List -----
                     
                     seed <- iter * 1000 + cell # unique seed from iter and cell
                     set.seed(seed)
                     
                     outlist <- fSimulate(Ar = l_CM[[cell]][[iter]],
                                          means = means,
                                          E = E,
                                          nIter = 100, # we repeat the simulation for EACH model in EACH cell 100 times
                                          verbatim = FALSE,
                                          n_seq = n_seq)
                     
                     # ----- Return -----
                     
                     return(outlist)
                     
                   } # end: foreach (n)


# Close down foreach
stopCluster(cl)


# --------- Save Results ---------

saveRDS(outlist, file = paste0('ARVAR_2020_Iter_', iter, '.RDS'))



