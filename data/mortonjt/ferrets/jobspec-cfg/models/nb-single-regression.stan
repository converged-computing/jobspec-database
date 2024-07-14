data {
  int<lower=0> N;    // number of samples
  int<lower=0> R;    // number of runs (batches)
  int<lower=0> J;    // number of subjects
  int<lower=0> p;    // number of covariates
  real depth[N];     // sequencing depths of microbes
  matrix[N, p] x;    // covariate matrix
  int y[N];       // observed microbe abundances
  int<lower=1, upper=J> subj_ids[N];   // subject ids
  int<lower=1, upper=R> batch_ids[N];  // batch ids
  real gamma[R];                     // batch differnetials
}

parameters {
  // parameters required for linear regression on the species means
  vector[p] beta;                 // covariates
  vector[J] alpha;                // subject differentials
  real<lower=0.001> disp;
}

transformed parameters {
  vector[N] lam;
  vector[N] lam_clr;
  vector[N] z;

  z = to_vector(rep_array(0, N));
  lam = x * beta;
  // add batch effects
  for (n in 1:N){
    lam[n] += gamma[batch_ids[n]];
  }
  // add in subject specific effects
  for (n in 1:N){
    lam[n] += alpha[subj_ids[n]];
  }
}

model {
  // setting priors ...
  disp ~ inv_gamma(1., 1.);

  for (j in 1:p){
    beta[j] ~ normal(0., 5.); // uninformed prior
  }

  // generating counts
  for (n in 1:N){
    target += neg_binomial_2_log_lpmf(y[n] | depth[n] + lam[n], disp);
  }
}