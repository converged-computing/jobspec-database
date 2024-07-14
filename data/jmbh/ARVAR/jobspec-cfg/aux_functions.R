# jonashaslbeck@gmail.com; August 2018
# dablander.fabian@gmail.com; June 2020


# -----------------------------------------------------------------------------------
# --------- Aux Functions 1 -----------------------------------------------------------
# -----------------------------------------------------------------------------------

# --------- Simulate VAR ----------

simVAR <- function(pars, 
                   means = 0, 
                   lags = 1, Nt = 100, init, residuals = 0.1, 
                   burnin)
{
  
  # print(pars)
  
  if (is.matrix(pars)) 
    pars <- list(pars)
  if (any(sapply(pars, function(x) length(unique(dim(x))) > 
                 1))) 
    stop("non-square graph detected.")
  if (missing(burnin)) {
    burnin <- min(round(Nt/2), 100)
  }
  # Ni <- ncol(pars[[1]])
  
  residuals <- 1
  Ni <- 6
  
  
  if (length(means) == 1) 
    means <- rep(means, Ni)
  maxLag <- max(lags)
  if (length(residuals) == 1) {
    residuals <- diag(residuals, Ni)
  }
  else if (length(residuals) == Ni) {
    residuals <- diag(residuals)
  }
  if (!is.matrix(residuals) && ncol(residuals) != Ni && nrow(residuals) != 
      Ni) {
    stop("'residuals' is not a square matrix")
  }
  totTime <- Nt + burnin
  if (missing(init)) {
    init <- matrix(0, maxLag, Ni)
  }
  Res <- matrix(, totTime, Ni)
  Res[1:maxLag, ] <- init
  for (t in (maxLag + 1):(totTime)) {
    Res[t, ] <- means + rowSums(do.call(cbind, lapply(seq_along(lags), 
                                                      function(i) pars[[i]] %*% (Res[t - lags[i], ] - means)))) + 
      mvtnorm::rmvnorm(1, rep(0, Ni), residuals)
  }
  return(as.data.frame(Res[-(1:burnin), ]))
}



# --------- Evaluate characteristics of VAR matrix ----------

f_eval <- function(A) {
  
  # Stationary?
  eig <- eigen(A)
  eig_val <- abs(eig$values)
  stable <- ifelse(all(eig_val < 1), 1, 0)
  
  # Ratio
  R <- mean(abs(diag(A))) / mean(abs(A[upper.tri(A) | lower.tri(A)]))
  
  # Exp
  D <- mean(eig_val)
  
  return(c(stable, R, D))
  
} # eoF


f_eval2 <- function(A) {
  
  # Ratio
  eig_val <- eigen(A)$values
  R <- mean(abs(diag(A))) / mean(abs(A[upper.tri(A) | lower.tri(A)]))
  
  # Dim
  D <- mean(eig_val)
  
  # Var eigen
  VarEig <- var(eig_val)
  
  # Trace abs value
  AbsTr <- sum(abs(diag(A)))
  
  out <- c(R, D, VarEig, AbsTr)
  names(out) <- c("R", "D", "VarEig", "AbsTr")
  
  return(out)
  
} # eoF



# --------- Compute the Gap ----------

get_gap_diff <- function(pe_diff, ee_diff, n_seq = 8:500) {
  n_pe <- which(pe_diff > 0)[1]
  n_ee <- which(ee_diff > 0)[1]
  
  gap <- n_seq[n_ee] - n_seq[n_pe]
  gap
}

get_crossing <- function(pe_diff, n_seq = 8:500) {
  n_pe <- which(pe_diff > 0)[1]
  n_seq[n_pe]
}

# --------- ? ----------

compute_average <- function(error_mat) {
  avg <- matrix(NA, ncol = 2, nrow = n_len)
  
  for (i in seq(2)) {
    for (j in seq(n_len)) {
      sum_j <- 0
      
      for (k in seq(n_rep)) {
        sum_j <- sum_j + error_mat[[k]][j, i]
      }
      
      avg[j, i] <- sum_j / n_rep
    }
  }
  
  avg
}


# --------- Computes average Gap for all Cells ----------

#' takes simulation output, returns list of 600.000 estimation error differences
get_all_ee_curves <- function(out_var, nr_models = 100, nr_iter = 100, cells = seq(74), cols = seq(493)) {
  nsim <- length(cells) * nr_models * nr_iter
  ee_diff <- matrix(NA, nrow = nsim, ncol = length(cols))
  
  index <- 1
  
  for (cell in cells) {
    for (model in seq(nr_models)) {
      for (iter in seq(nr_iter)) {
        m <- out_var[[cell]][[model]]
        
        out <- m$EE_all[, , iter]
        ee_diff[index, ] <- out[, 1] - out[, 2]
        
        index <- index + 1
      }
    }
  }
  
  ee_diff
}


# Function to compute cell freq
get_cell <- function(row, ODbound = 0.03571429/2, Dbound = 0.01428571/2, p = 6) {
  is_within <- function(x, vec) x > vec[1] && x < vec[2]
  
  OD <- row[1]
  D <- row[2]
  
  cell_pos <- apply(m_cell_fill, 1, function(x) { 
    cell <- as.numeric(x)
    ODbounds <- c(cell[1] - ODbound, cell[1] + ODbound)
    Dbounds <- c(cell[2] - Dbound, cell[2] + Dbound)
    is_within(OD, ODbounds) && is_within(D, Dbounds)
  })
  
  stopifnot(sum(cell_pos) == 1)
  which(cell_pos)
}


# Compute ngap for each cell
compute_ngap_cells <- function(out_var, n_cells = 74) {
  
  ngap_cell <- numeric(n_cells)
  
  for (cell in seq(n_cells)) {
    pe <- lapply(out_var[[cell]], function(x) x$PE_all[, , 1:100])
    ee <- lapply(out_var[[cell]], function(x) x$EE_all[, , 1:100])
   
    # compute the differences across all ns
    arvar_diff <- sapply(seq(100), function(i) {
      pe_diff <- pe[[i]][, 1, i] - pe[[i]][, 2, i] # AR - VAR (PE)
      ee_diff <- ee[[i]][, 1, i] - ee[[i]][, 2, i] # AR - VAR (EE)
      
      gap <- get_gap_diff(pe_diff, ee_diff)
      gap
    })
    
    # compute the mean across differences
    ngap_cell[cell] <- round(mean(arvar_diff, na.rm = TRUE), 2)
  }
  
  ngap_cell
}


#' Weights the estimation error difference curves by resampling them according to their probability (based on cells)
get_cell_weighted_samples <- function(cell_weights, diffs, n_cells = 60, seed = 1) {
  nvar <- 10000
  draws <- round(cell_weights * nvar)
  
  # if the following is uncommented, renormalizes to sample only 100 per cell
  # x <- cell_weights * nvar
  # draws <- round((x / max(x)) * 100)
  samples <- c()
  
  set.seed(seed)
  for (i in seq(n_cells)) {
    d <- draws[i]
    
    if (d != 0) {
      lines <- diffs[[i]][sample(seq(100), size = d, replace = TRUE), ]
      samples <- rbind(samples, lines)
    }
    
  }
  
  samples
}


#' Weights the estimation error difference curves by resampling them according to their probability (based on the mixed model)
get_weighted_samples <- function(diffs_mat, model_weights, seed = 1) {
  set.seed(seed)
  nr_models <- nrow(diffs_mat)
  index <- sample(nr_models, replace = TRUE, size = nr_models, prob = model_weights)
  diffs_mat[index, ]
}


# -------- Compute Weight of each Model --------
get_model_weights <- function(var_mats, mind_maastricht = 'files/MM_MindMaastricht.RDS') {
  library('mvtnorm')
  
  mm <- readRDS(file = mind_maastricht)
  len <- length(var_mats)
  
  M <- as.numeric(mm[, , 1])
  S <- diag(as.numeric(mm[, , 2]))
  
  counter <- 1
  weights <- rep(NA, len * 100)
  
  for (d in seq(len)) {
    for (i in 1:100) {
      x <- as.numeric(var_mats[[d]][[i]])
      weights[counter] <- dmvnorm(x, mean = M, sigma = S, log = FALSE)
      counter <- counter + 1
    }
  }
  
  weights / sum(weights)
}


#' Computes the n at which the estimation error is better for VAR
#' either using resampling or not; offset is due to the fact that we do not sample the first
#' 7 ns because the errors have high variance with such low n
compute_nswitch <- function(diffs_mat, model_weights = NULL, seed = 1, offset = 7) {
  if (!is.null(model_weights)) {
    diffs_mat <- get_weighted_samples(diffs_mat, model_weights, seed = seed)
  }
  
  nswitch <- apply(diffs_mat, 1, function(diffs) which(diffs > 0)[1] + offset)
  nswitch
}


get_ee_comp <- function(out_var, n_cells = 74, n_rep = 100, n_models = 100, n_seq = 8:500) {
  ee_comp <- array(NA, dim = c(n_cells, n_models, 2, 493)) # 74 cells, 100 models, 100 replications, 2 rules, 493 ns
  
  for (cell in seq(n_cells)) {
    for (m in seq(n_models)) {
      
      pe <- out_var[[cell]][[m]]$PE_all
      ee <- out_var[[cell]][[m]]$EE_all
      
      # Average prediction errors
      pe_ar <- apply(pe[, 1, ], 1, mean)
      pe_var <- apply(pe[, 2, ], 1, mean)
      
      # Standard error of VAR prediction error
      pe_se_var <- apply(pe[, 2, ], 1, sd) / sqrt(n_seq)
      
      # Average estimation errors
      ees <- cbind(
        apply(ee[, 1, ], 1, mean), # AR
        apply(ee[, 2, ], 1, mean)  # VAR
      )
      
      # Choose model with lowest prediction error
      sel_rule1 <- (pe_var < pe_ar) + 1
      
      # 1 Standard Error Rule
      sel_rule2 <- ((pe_var + pe_se_var) < pe_ar) + 1
      
      best <- (ee_var < ee_ar) + 1
      
      ee_sel <- sapply(seq(493), function(i) {
        c('best' = ees[i, best[i]], 'rule1' = ees[i, sel_rule1[i]], 'rule2' = ees[i, sel_rule2[i]])
      })
      
      ee_comp[cell, m, 1, ] <- ee_sel[1, ] - ee_sel[2, ]
      ee_comp[cell, m, 2, ] <- ee_sel[1, ] - ee_sel[3, ]
      
      # # Average over repetitions
      # for (i in seq(n_rep)) {
      #   # Rule 1: Choose model with lowest prediction error
      #   pred <- pe[, , i]
      #   sel_rule1 <- apply(pred, 1, which.min)
      #   
      #   # Rule 2: 1SER
      #   sel_rule2 <- sapply(seq(493), function(i) {
      #     (pred[i, 1] > (pred[i, 2] + pe_se_var[i])) + 1
      #   })
      #   
      #   est <- ee[, , i]
      #   best <- apply(est, 1, which.min)
      #   
      #   ee_sel <- sapply(seq(493), function(i) {
      #     c('best' = est[i, best[i]], 'rule1' = est[i, sel_rule1[i]], 'rule2' = est[i, sel_rule2[i]])
      #   })
      # }
      
    } # end for: i
    
    print(cell)
    
  } # end for: d
  
  ee_comp
}

# -----------------------------------------------------------------------------------
# --------- Aux Functions 2 -----------------------------------------------------------
# -----------------------------------------------------------------------------------

# --------- Shading for SEs ----------

library(scales)

shadeSE <- function(x, y_up, y_low, color, alpha) {
  
  col <- alpha(colour = color, alpha = alpha)
  n <- length(x)
  
  n <- length(x)
  p_x <- c(x, x[n:1])
  p_y <- c(y_up, y_low[n:1])
  polygon(x = p_x, y = p_y, col=col, border = NA)
  
} # eoF


# --------- Estimation Function ----------

fEstimate <- function(train, test, model = "AR", trueVAR) {
  
  # Aux variables
  p <- ncol(train)
  n_train <- nrow(train)
  n_test <- nrow(test)
  
  # Lag data
  train_dep <- as.matrix(train[-1, ])
  train_ind <- as.matrix(train[-n_train, ])
  test_dep <- as.matrix(test[-1, ])
  test_ind <- as.matrix(test[-n_test, ])
  
  # Storage
  m_predictions <- matrix(NA, nrow = n_test-1, ncol = p)
  m_VAR <- matrix(0, p, p)
  
  # Fit model
  if(model == "AR") {
    for(i in 1:p) {
      model_i <- lm(train_dep[, i] ~ train_ind[, i]) # fit
      m_VAR[i, i] <- model_i$coefficients[2] # save in parameter matrix
      m_predictions[, i] <- model_i$coefficients[1] + model_i$coefficients[2]*test_ind[, i] # save predictions
    }
  }
  
  if(model == "VAR") {
    for(i in 1:p) {
      model_i <- lm(train_dep[, i] ~ train_ind) # fit
      m_VAR[i, ] <- model_i$coefficients[-1] # save in parameter matrix
      m_predictions[, i] <- model_i$coefficients[1] + test_ind %*% matrix(model_i$coefficients[-1], nrow=p)  # save predictions
    }
  }
  
  if(model == "cAR") {
    for(i in 1:p) {
      model_i <- lm(train_dep[, i] ~ train_ind) # fit
      m_VAR[i, ] <- model_i$coefficients[-1] # save in parameter matrix
      m_VAR[i, -i] <- 0
      model_i$coefficients[-1][-i] <- 0
      m_predictions[, i] <- model_i$coefficients[1] + test_ind %*% matrix(model_i$coefficients[-1], nrow=p)  # save predictions
    }
  }
  
  ## Compute errors
  # Estimation Error
  MSE_EE <- mean((m_VAR - trueVAR)^2)
  MSE_EE_false <- mean((diag(m_VAR) - diag(trueVAR))^2)
  
  # Prediction Error: MSE
  v_MSE_PE <- rep(NA, p)
  for(i in 1:p) v_MSE_PE[i] <- mean((test_dep[, i] - m_predictions[, i])^2)
  MSE_PE <- mean(v_MSE_PE)
  
  # Prediction Error: R2
  v_R2_PE <- rep(NA, p)
  for(i in 1:p) v_R2_PE[i] <- 1 - var(m_predictions[, i]-test_dep[, i]) / var(test_dep[, i])
  
  
  # Output
  outlist <- list("estVAR" = m_VAR, 
                  "predsTest" = m_predictions, 
                  "trueTest" = test_dep, 
                  "MSE_PE" = MSE_PE,
                  "MSE_EE" = MSE_EE, 
                  "MSE_EE_false" = MSE_EE_false, 
                  "R2_PE" = v_R2_PE)
  
} # eoF




# # Testing
# train <- data[1:1500, ]
# test <- data[1501:2000, ]
# 

# out <- fEstimate(train = train, 
#                  test = test, 
#                  model = "AR", 
#                  trueVAR = Ar)


# --------- Simulation Function ----------


fSimulate <- function(Ar, 
                      means, 
                      E, 
                      nIter, 
                      verbatim = TRUE, 
                      n_seq) 
  
{
  
  l_data <- list()
  
  # Generate Testset
  testset <- simVAR(pars = Ar, 
                    means = means, 
                    lags = 1, 
                    Nt = 2000,
                    residuals = E)
  
  # Storage
  n_var <- length(n_seq)
  a_PE <- a_EE <- a_EEf <- array(NA, dim = c(n_var, 2, nIter))
  
  
  for(i in 1:nIter) {
    
    # Generate Data
    data <- simVAR(pars = Ar, 
                   means = means, 
                   lags = 1, 
                   Nt = n_seq[length(n_seq)],
                   residuals = E)
    
    l_data[[i]] <- data
    
    for(n in 1:n_var) {
      
      # ----- Fit AR model & make predictions -----
      out_AR <- fEstimate(train = data[1:n_seq[n], ], 
                          test = testset, 
                          model = "AR", 
                          trueVAR = Ar)
      
      a_PE[n, 1, i] <- out_AR$MSE_PE
      a_EE[n, 1, i] <- out_AR$MSE_EE
      a_EEf[n, 1, i] <- out_AR$MSE_EE_false
      
      
      # ----- Fit VAR model & make predictions -----
      out_VAR <- fEstimate(train = data[1:n_seq[n], ], 
                           test = testset, 
                           model = "VAR", 
                           trueVAR = Ar)
      
      a_PE[n, 2, i] <- out_VAR$MSE_PE
      a_EE[n, 2, i] <- out_VAR$MSE_EE
      
      # ## REMOVED BY JONAS MAY 27
      # # ----- Fit VAR / cAR model & make predictions -----
      # out_cAR <- fEstimate(train = data[1:n_seq[n], ], 
      #                      test = testset, 
      #                      model = "cAR", 
      #                      trueVAR = Ar)
      # 
      # a_PE[n, 3, i] <- out_cAR$MSE_PE
      # a_EE[n, 3, i] <- out_cAR$MSE_EE
      
      
      if(verbatim) cat(paste0("Iteration = ", i, " n = ", n, "\n"))
    } # for: loop n
    
  } # for: iterations
  
  # --------- Aggregate over Iteration ----------
  
  # ## Removed by Jonas May 27th to save memory (computing these summaries can be done later)
  #
  # # Means
  # m_PE <- apply(a_PE, 1:2, median) # prediction errors
  # m_EE <- apply(a_EE, 1:2, median) # estimation errors
  # 
  # # Upper / Lower 25% quantile
  # PE_qu <- apply(a_PE, 1:2, function(x) quantile(x, probs = c(.25, .75)))
  # EE_qu <- apply(a_EE, 1:2, function(x) quantile(x, probs = c(.25, .75)))
  
  # --------- Reducing Memory ----------
  
  a_PE <- round(a_PE, 5)
  a_EE <- round(a_EE, 5)
  a_EEf <- round(a_EEf, 5)
  
  # --------- Return Results ----------
  
  outlist <- list( #"PE" = m_PE, 
    #"PE_qu" = PE_qu,
    #"EE" = m_EE, 
    #"EE_qu" = EE_qu, 
    "PE_all" = a_PE, 
    "EE_all" = a_EE, 
    "EEf_all" = a_EEf)
  
} # end of Simulation Func


# removes unneeded stuff that costs too much memory to store
prune_simoutput <- function(ARVAR_iter, len = 300) {
  x <- ARVAR_iter
  
  for (i in seq(60)) {
    x[[i]]$EE_qu <- NULL
    x[[i]]$PE_qu <- NULL
    x[[i]]$EEf_all <- NULL
  }
  
  x
}

get_weighted_sd <- function(x, weights) {
  n <- length(x)
  
  weighted_mean <- sum(x * weights, na.rm = TRUE)
  num <- sum(weights * (x - weighted_mean)^2, na.rm = TRUE)
  denom <- (n - 1)/n * sum(weights, na.rm = TRUE)
  
  sqrt(num / denom)
}

plot_weights <- function(cell = 1) {
  lo <- (cell - 1) * 100
  hi <- cell * 100
  
  w <- model_weights[seq(lo, hi)]
  hist(w, breaks = 20)
  abline(v = mean(w), lwd = 2, col = 'green')
  abline(v = cell_weights[cell] / 100, lwd = 2)
}

# diff <- numeric(60)
# 
# for (d in seq(60)) {
#   lo <- (d - 1) * 100
#   hi <- d * 100
#   
#   w <- mean(model_weights[seq(lo, hi)])
#   cw <- cell_weights[d] / 100
#   
#   diff[d] <- w - cw
# }