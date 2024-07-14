###############################################################################L
#
#             Connectivity of (i.e., movement between) refuges in western 
#               Tennessee and the LMAV by mallards
#
#
#
# This code will perform the analysis for the multistate model to estimate 
# movement between refuges for mallards in western Tennessee. The project is 
# part of a mallard project at Tennessee Tech University in collaboration with 
# Tennessee Wildlife Resources Agency and the U.S. Fish and Wildlife Service. 

# Dependencies: data -> "all_dat_oct2019_march2023.rds" & "spatial_sanctuary.shp"
#               R scripts -> "DataPrep_MultistateModel.R", 
#                            "ModelSpecification_multistate.R", 
#                            "Functions_MultistateModel.R" 
#
#
#
# Allison C Keever & Nick Masto
# Tennessee Tech University
# akeever@tntech.edu & nmasto1214@gmail.com
# GitHub: akeever2 & nmasto
# Created: 5/5/21
# Last updated: 8/23/23

# Updated to add spatial refuge data, additional return data, 
# and other specifications 8/22/2023

###############################################################################L



# Prep work space ---------------------------------------------------------------

# Load necessary packages
library(tidyverse)
library(jagsUI)
library(tidybayes)
library(mcmcplots)
library(superdiag)
library(R2jags)
library(jagshelper)

# Run script to load and prep data
source("scripts/DataPrep_MultistateModel.R")

# Run script to create model text file
source("scripts/ModelSpecification_multistate.R")

# Run script to create needed functions
source("scripts/Functions_MultistateModel.R")



# Naive analysis --------------------------------------------------------------

# mov_mat <- temp <-  matrix(0, nrow = 14, ncol = 14,
#                   dimnames = list(row.names = c("BB", "LINWR", "BS", "HI", "M", 
#                                                 "WL", "CNWR", "BLNWR", "LL", 
#                                                 "RLNWR", "HB", "TNNWR", "WRNWR", 
#                                                 "Dead_lost"), 
#                                   col.names = c("BB", "LINWR", "BS", "HI", "M", 
#                                                 "WL", "CNWR", "BLNWR", "LL", 
#                                                 "RLNWR", "HB", "TNNWR", "WRNWR", 
#                                                 "Dead_lost")))
# enc_hist_only <- as.matrix(enc_hist %>% select(starts_with("occ_")))
# for(i in 1:nrow(enc_hist_only)){
#   for(j in f[i]:(ncol(enc_hist_only) - 1)){
#     temp <-  matrix(0, nrow = 14, ncol = 14)
#     temp[enc_hist_only[i,j], enc_hist_only[i,j+1]] <- 1
#     mov_mat <- mov_mat + temp
#   }
# }
# 
# partial_move_mat <- t(apply(prop.table(mov_mat, 1)[1:13, 1:13], 1, 
#                           function(x) round(x / sum(x, na.rm = TRUE), 5)))


# Run analyses for null model --------------------------------------------------

# Bundle all data for JAGS
jags_data <- list(y = as.matrix(enc_hist %>% select(contains("occ"))),   # y = occasion
                  f = f,                                                 # f specifies date/hr first cap
                  Size1 = Refs$Area_sqkm,                                # Area of "from" refuge
                  Size2 = apply(matrix(1:13, nrow = 13), 1,              # Area of "to" refuge
                                FUN = function(x) cbind(Refs$Area_sqkm[-x])),
                  Sex = ifelse(enc_hist$sex == "F", 0, 1),               # F is the dummy
                  Age = ifelse(enc_hist$age == "juv", 0, 1), # J is the dummy
                  Distance = dist_datum,                                 # specify distance df
                  noccs = ncol(enc_hist %>% select(contains("occ"))),    # no of occasions
                  ninds = nrow(enc_hist %>% select(contains("occ"))),    # no individs
                  init_state = first_state(enc_hist %>% select(contains("occ")), f), # first state function
                  z = known_state(enc_hist %>% select(contains("occ")), 14)) # known state (refuge or unk)
                                                                             # optimizes searching param space

# Initial values to help with convergence
inits <- function(){list(phi = runif(1, 0, 1), p = runif(1, 0, 1), 
                         z = init_z(enc_hist %>% select(contains("occ")), f))} # run the function

# Parameters to monitor
params <- c("phi", "p", 
            "psi.BB", "psi.LL", "psi.LINWR", "psi.WL", "psi.BS", "psi.HI", 
            "psi.M",  "psi.CNWR", "psi.BLNWR", "psi.RLNWR_N",
            "psi.HB", "psi.RLNWR_S", "psi.P", 
            "b.dist", "b.size1", "b.size2", "b.sex", "b.age", "b0") 
# no longer need intercepts for all refuges - b0 applies to all

# This will not be the case. Will need to run in parallel:
# 10 chains/100 iterations/50 burn-ins/3 thins on each of 10 cores
# Continue simulation up to 500 max at intervals of 100

# MCMC test settings
ni <- 100
nb <- 50    
nt <- 3
nc <- 10
nmax <- 1000

# Run the models
system.time(out <- autojags(data = jags_data, inits = inits, parameters.to.save = params,
                            model.file = "fullconstant_MallardMSM.txt", n.chains = nc, 
                            iter.increment = ni, n.thin = nt, max.iter = nmax, 
                            n.burnin = nb, parallel = TRUE, codaOnly = params))


# Save the output
save(out, file = "Results/FullMultStatMod_v500i.RData")


# Results processing------------------------------------------------------------

load("Results/FullMultStatMod_v500i.RData") # this is actually 1000 iterations

# Survival and capture probability
out %>% gather_draws(phi, p) %>%
  mean_qi(.width = 0.9)

# Traceplots 
out %>% 
  gather_draws(b0, b.dist, b.size1, b.size2, b.sex, b.age, phi, p, deviance) %>%
  rename(Variable = .variable, 
         Chain = .chain, 
         Draw = .draw, 
         Iteration = .iteration, 
         Value = .value) %>% 
  ggplot() +
  facet_wrap(~Variable, scales = "free") +
  geom_line(aes(x = Iteration, y = Value, color = as.factor(Chain)))+
  theme_classic() +
  theme(text = element_text(size = 13),
        legend.position = "right") +
  labs(color = "Chains")

ggsave("Figures/Chains.png", dpi = 600)

# Beta coefficients for probability of moving among sites  

betas <- out %>%
  gather_draws(b0, b.dist, b.size1, b.size2, b.sex, b.age) %>%
  mean_qi(.width = c(.68, 0.9, 0.95)) %>%
  select(1:5) %>%
  rename(Variable = .variable, Mean = .value, Lower = .lower, Upper = .upper) 

write.csv(betas, "Results/betas_90.csv")

# Calculating transition probabilities - step 1 create prediction data frame based on our Refuges
calc_refuge_probs(to.ref = "BB", dist_datum = dist_datum, Refs = Refs)    # BB as example
calc_refuge_probs(from.ref = "BB", dist_datum = dist_datum, Refs = Refs)

probs <- lapply(c("BB", "LINWR","BS","HI","M","WL","CNWR",
                  "BLNWR","LL","RLNWR_N","HB", "RLNWR_S", "P"),
                calc_refuge_probs, dist_datum = dist_datum, Refs = Refs, to.ref = NULL)

probs <- bind_rows(probs)

probs2 <- probs %>% filter(psi < 0.8,
                           sex == 1,
                           age == 1) %>% slice_max(psi, n = 23)

probs2 %>% slice_max(psi, n = 10)

write.csv(probs, "Results/ref_probs.csv")

# Plotting---------------------------------------------------------------------

# Plot posteriors for phi and p
out %>% spread_draws(phi, p) %>%
  rename("post.phi" = phi, "post.p" = p) %>%
  mutate(prior.phi = runif(n = n(), 0, 1),
         prior.p = runif(n = n(), 0, 1)) %>%
  pivot_longer(cols = starts_with("p"),
               names_to = c("type", ".variable"),
               names_sep = "[.]",
               values_to = ".value") %>%
  ggplot(aes(x = .value, y = .variable, fill = type))+
  stat_slab(alpha = 0.5) +
  scale_x_continuous(name = "Probability", breaks = seq(0, 1, 0.1)) +
  scale_y_discrete(name = NULL)+
  scale_fill_brewer(palette = "Dark2", name = "Type", # My favorite palette
                    labels = c("Posterior", "Prior")) +
  theme_classic() +
  theme(text = element_text(size = 13),
        legend.position = "top")


# Plot posteriors for betas
out %>% spread_draws(b.dist, b.size1, b.size2, b.sex, b.age) %>%
  rename("post_b.dist" = b.dist, 
         "post_b.size1" = b.size1, 
         "post_b.size2" = b.size2,
         "post_b.sex" = b.sex, "post_b.age" = b.age) %>%
  mutate(prior_b.dist = rnorm(n = n(), 0, .5),
         prior_b.size1 = rnorm(n = n(), 0, .5),
         prior_b.size2 = rnorm(n = n(), 0, .5),
         prior_b.sex = rnorm(n = n(), 0, .5),
         prior_b.age = rnorm(n = n(), 0, .5)) %>%
  pivot_longer(cols = starts_with("p"),
               names_to = c("type", ".variable"),
               names_sep = "[_]",
               values_to = ".value") %>%
  ggplot(aes(x = .value, y = type))+
  stat_slab(alpha = 0.5) +
  facet_wrap(~ .variable, scales = "free_x") +
  scale_x_continuous(name = "Coefficient value") +
  scale_y_discrete(name = NULL)+
  scale_fill_brewer(palette = "Dark2", name = "Type",
                    labels = c("Posterior", "Prior")) +
  theme_classic() +
  theme(text = element_text(size = 13),
        legend.position = "top")


# Plot relationships with covariates
pred_datum <- expand_grid(age = c(0, 1),
                          sex = c(0, 1),
                          size1 = 9.7, # Mean = 9.7; range = 1.32:45.7 sqkm
                          size2 = 9.7,
                          dist = 46.9            # Mean = 46.9; range = 1.3:120 km
)


sex.label <- c('0' = "Female", '1' = "Male")
age.label <- c('0' = "Juvenile", '1' = "Adult")

# As the Bayesian inference returns a distribution of possible effect values (the posterior),
# the credible interval is just the range containing a particular percentage of probable values.
# For instance, the 95% credible interval is simply the central portion of the posterior distribution
# that contains 95% of the values. Using 89% is another popular choice, and used to be the default
# for a long time (read story https://github.com/easystats/bayestestR/discussions/250). How did it start?

# Naturally, when it came about choosing the CI level to report by default, people started using 95%,
# the arbitrary convention used in the frequentist world. However, some authors suggested that 95%
# might not be the most appropriate for Bayesian posterior distributions, potentially lacking
# stability if not enough posterior samples are drawn (Kruschke, 2014).

# The proposition was to use 90% instead of 95%. However, recently, McElreath (2014, 2018)
# suggested that if we were to use arbitrary thresholds in the first place, why not use 89%?
# Moreover, 89 is the highest prime number that does not exceed the already unstable 95% threshold.
# What does it have to do with anything? Nothing, but it reminds us of the total arbitrariness
# of these conventions (McElreath, 2018).


p.size1 <- out %>% 
  spread_draws(b.dist, b.size1, b.size2, b.sex, b.age, b0) %>%
  expand_grid(pred_datum) %>%
  mutate(lp = b0 + b.dist * dist + b.size1 * size1 + b.size2 * size2 +
           b.sex * sex + b.age * age,
         psi = plogis(lp)) %>%
  filter(age == 1, sex == 1) %>%
  ggplot(aes(x = size1, y = psi))+
  stat_lineribbon(.width = c(0.68, 0.90, 0.95)) +
  # facet_wrap(age ~ sex, labeller = labeller(sex = sex.label, age = age.label)) +
  scale_x_continuous(name = "Size of emigration sanctuary (sq.km)") +
  scale_y_continuous(name = "Daily probability of sanctuary transition",
                     breaks = seq(0, 0.002, 0.0005), limits = c(0, 0.002))+ # may need to adjust
  scale_fill_brewer(palette = "Greys") +
  theme_classic() +
  theme(text = element_text(size = 13),
        legend.position = "none")

p.size1
ggsave("Figures/AdultMale_size1_50km2.png", dpi = 600)


pred_datum <- expand_grid(age = c(0, 1),
                          sex = c(0, 1),
                          size1 = 9.7,  # Mean = 9.7; range = 2:50 sqkm
                          size2 = seq(1, 50, 1),
                          dist = 46.9 # Mean = 46.9; range = 1:120 km
)


p.size2 <- out %>% spread_draws(b.dist, b.size1, b.size2, b.sex, b.age, b0) %>%
  expand_grid(pred_datum) %>%
  mutate(lp = b0 + b.dist * dist + b.size1 * size1 + b.size2 * size2 +
           b.sex * sex + b.age * age,
         psi = plogis(lp)) %>%
  filter(age == 1, sex == 1) %>%
  ggplot(aes(x = size2, y = psi))+
  stat_lineribbon(.width = c(0.68, 0.90, 0.95)) +
  #facet_wrap(age ~ sex, labeller = labeller(sex = sex.label, age = age.label)) +
  scale_x_continuous(name = "Size of immigration sanctuary (sq.km)") +
  scale_y_continuous(name = "Daily probability of sanctuary transition",
                     breaks = seq(0, 0.0025, 0.0005), limits = c(0, 0.0025))+ # may need to adjust
  scale_fill_brewer(palette = "Greys") +
  theme_classic() +
  theme(text = element_text(size = 13),
        legend.position = "none")
p.size2
ggsave("Figures/AdultMale_size2.png", dpi = 600)


pred_datum <- expand_grid(age = c(0, 1),
                          sex = c(0, 1),
                          size1 = 9.7,         # Mean = 9.7; range = 1:50 sqkm
                          size2 = 9.7,
                          dist = seq(1, 115, 1) # Mean = 46.9; range = 1:120 km
)


p.dist <- out %>% spread_draws(b.dist, b.size1, b.size2, b.sex, b.age, b0) %>%
  expand_grid(pred_datum) %>%
  mutate(lp = b0 + b.dist * dist + b.size1 * size1 + b.size2 * size2 +
           b.sex * sex + b.age * age,
         psi = plogis(lp)) %>%
  filter(age == 1, sex == 1) %>%
  ggplot(aes(x = dist, y = psi))+
  stat_lineribbon(.width = c(0.68, 0.90, 0.95)) +
  # facet_wrap(age ~ sex, labeller = labeller(sex = sex.label, age = age.label)) +
  scale_x_continuous(name = "Distance between sanctuaries (km)") +
  scale_y_continuous(name = "Daily probability of sanctuary transition")+#,
  #                   breaks = seq(0, 0.0002, 0.15), limits = c(0, 0.15))+ # may need to adjust
  scale_fill_brewer(palette = "Greys") +
  theme_classic() +
  theme(text = element_text(size = 13),
        legend.position = "none")

p.dist
ggsave("Figures/AdultMale_dist.png", dpi = 600)


pred_datum <- expand_grid(age = c(0, 1),
                          sex = c(0, 1),
                          size1 = 9.7,  # Mean = 9.7; range = 1:50 sqkm
                          size2 = 9.7,
                          dist = 46.9    # Mean = 46.9; range = 1:120 km
)


p.ind <- out %>% spread_draws(b.dist, b.size1, b.size2, b.sex, b.age, b0) %>%
  expand_grid(pred_datum) %>%
  mutate(lp = b0 + b.dist * dist + b.size1 * size1 + b.size2 * size2 +
           b.sex * sex + b.age * age,
         psi = plogis(lp)) %>%
  ggplot(aes(x = factor(age), y = psi, fill = factor(sex)))+
  scale_fill_brewer(palette = "Dark2", labels = c("Female", "Male"), name = "Sex") +
  scale_color_brewer(palette = "Dark2", labels = c("Female", "Male"), name = "Sex") +
  stat_pointinterval(aes(color = factor(sex)), 
             position = position_dodge(), .width = c(0.68, 0.90), pch = 21)+
  scale_x_discrete(name = "Age", labels = c("Juvenile", "Adult")) +
  scale_y_continuous(name = "Daily probability of sanctuary transition",
                     breaks = seq(0, 0.00045, 0.00015), limits = c(0, 0.00045))+
  
  theme_classic() +
  theme(text = element_text(size = 13))

p.ind
ggsave("Figures/AgeSex.png", dpi = 600)


library(patchwork)
patch1 <- (p.dist + p.ind) / (p.size1 + p.size2) + #plot_layout(guides = "collect") +  
  plot_annotation(tag_levels = 'a')
ggsave("Figures/combo1.png", width = 10, height = 8, dpi = 600)

patch2 <- p.size1 + p.size2 + plot_annotation(tag_levels = 'a')
ggsave("Figures/combo2.png", width = 7, height = 5, units = "in",dpi = 600)

# Predict a mean psi (and 1 SDs)-----------------------------

pred_datum1 <- expand_grid(age = c(0, 1), # Normal adult male
                          sex = c(0, 1),
                          size1 = 9.7,   # Mean = 17.4; range = 1:50 sqkm
                          size2 = 9.7,
                          dist = 46.9    # Mean = 46.9; range = 1:120 km
)


pred_datum2 <- expand_grid(age = c(0, 1), # Normal adult male
                           sex = c(0, 1),
                           size1 = 9.7,   # Mean = 17.4; range = 1:50 sqkm
                           size2 = 9.7,
                           dist = 46.9 + 25.8    # Mean = 46.9; SD = 25.8 range = 1:120 km
)

pred_datum3 <- expand_grid(age = c(0, 1), # Normal adult male
                           sex = c(0, 1),
                           size1 = 9.7,   # Mean = 17.4; range = 1:50 sqkm
                           size2 = 9.7,
                           dist = 46.9 - 25.8    # Mean = 46.9; range = 1:120 km
)


p1 <- out %>% spread_draws(b.dist, b.size1, b.size2, b.sex, b.age, b0) %>%
  expand_grid(pred_datum1) %>%
  mutate(lp = b0 + b.dist * dist + b.size1 * size1 + b.size2 * size2 +
           b.sex * sex + b.age * age,
         psi = plogis(lp)) %>%
  filter(age == 1, sex == 1) %>%
  select(size1, dist, psi) %>%
  group_by(size1, dist) %>%
  mean_qi(.width = c(0.68, 0.9, 0.95)) 

p2 <- out %>% spread_draws(b.dist, b.size1, b.size2, b.sex, b.age, b0) %>%
  expand_grid(pred_datum2) %>%
  mutate(lp = b0 + b.dist * dist + b.size1 * size1 + b.size2 * size2 +
           b.sex * sex + b.age * age,
         psi = plogis(lp)) %>%
  filter(age == 1, sex == 1) %>%
  select(size1, dist, psi) %>%
  group_by(size1, dist) %>%
  mean_qi(.width = c(0.68, 0.9, 0.95)) 

p3 <- out %>% spread_draws(b.dist, b.size1, b.size2, b.sex, b.age, b0) %>%
  expand_grid(pred_datum3) %>%
  mutate(lp = b0 + b.dist * dist + b.size1 * size1 + b.size2 * size2 +
           b.sex * sex + b.age * age,
         psi = plogis(lp)) %>%
  filter(age == 1, sex == 1) %>%
  select(size1, dist, psi) %>%
  group_by(size1, dist) %>%
  mean_qi(.width = c(0.68, 0.9, 0.95)) 

psi <- bind_rows(p1,p2,p3) %>% mutate(psi_month = 1-(1-psi)^120,
                                      psi_month_low = 1-(1-.lower)^120,
                                      psi_month_up = 1-(1-.upper)^120)

write.csv(psi, "Results/psi_120d.csv")
  
?gather_draws
prob <- as.matrix(out %>% spread_draws(b.dist, b.size1, b.size2, b.sex, b.age, b0) %>%
                    expand_grid(pred_datum) %>%
                    mutate(lp = b0 + b.dist * dist + b.size1 * size1 + b.size2 * size2 +
                             b.sex * sex + b.age * age,
                           psi = plogis(lp)) %>%
                    filter(age == 1, sex == 1) %>%
                    select(size1, dist, psi) %>%
                    group_by(size1, dist) %>%
                    mean_qi() %>%
                    select(size1, dist, psi) %>%
                    pivot_wider(names_from = dist, names_prefix = "dist", values_from = psi) %>%
                    select(-size1))



size  = seq(2, 50, 1)     # need to find the size seq based on new refuges
dist  = seq(1, 120, 1)    # and the dist sequence
plotly::plot_ly(x = ~ dist, y = ~ size,
                z = ~ prob) %>%
  plotly::add_surface()



