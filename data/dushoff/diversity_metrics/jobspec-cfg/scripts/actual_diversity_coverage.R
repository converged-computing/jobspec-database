#check statistical coverage for CI for coverage-standardized samples

# CI for coverage-based rarefaction test plan. Goal is to test statistical coverage of these CI, but not necessarily to "checkplot" them, in that nuance beyond whether they are shifted high/low vs. simply to narrow might not be clear from this and that's ok.
# 




#0 Load some libraries and functions
library(furrr)
source("scripts/helper_funs/uniroot_gamma_and_lnorm.R")
# source("scripts/helper_funs/estimation_funs.R")
R_FUTURE_FORK_ENABLE=T
library(iNEXT)
library(tictoc)
library(data.table)

# 1: simulate 2 SADS. More is unweildy. Have same Richness (100-200 ish), Evenness (realistic) but one lognormal and the other gamma shaped.

# simulate full a few full communities. Using a nice eveness like 0.3 leads to a weird simpson of 60.7. Fine


richness<-200
even<-0.3
simpson<-even*(richness-1)+1
simpson<-50
even<-(simpson-1)/(richness-1)
gamma_comm<-fit_SAD(rich=richness, simpson=simpson, dstr="gamma")
lnorm_comm<-fit_SAD(rich=richness, simpson=simpson, dstr="lnorm")

# 2) from each SAD, take 1e4 samples at 10 sample sizes from say 1e1 to 1e4.
# 
nc<-24 #24 on amarel
plan(strategy = multiprocess, workers = nc)

reps<-1e5
# SS<-c(10^c(1:5), 5*10^c(1:5))
clist<-list("gamma_comm"=gamma_comm, 
        "lnorm_comm"=lnorm_comm)

save(clist, file="my_comms.RData")
tic()
baseline_samples<-future_map_dfr(1:reps
                                  # , .options = future_options(globals(structure=T, add=c("reps", "SS", "gamma_comm", "lnorm_comm", "sample_infinite"
                                 # , "dfun", "compute_cov"))
                                 , function(rep){
                                 indis<-runif(1, 5e1, 5e3)
                                     map_dfr(1:length(clist), function(SAD){
                                         mydst=clist[[SAD]][[3]]
                                         myabs=sample_infinite(mydst, indis)
                                         # 3) For each sample, compute true coverage, Hill diversity with ell={-1,0,1}
                                         # 
                                         rich=sum(myabs>0)
                                         shan=dfun(myabs,0)
                                         simp=dfun(myabs,-1)
                                         cov=sum((myabs>0)*mydst)
                                         
                                         return(data.frame(t(myabs), SS=indis, comm=names(clist)[SAD]
                                                           , rich=rich, shan=shan, simp=simp, tc=cov))
                                 })
                             })
toc()




#write to disk.
fwrite(baseline_samples
       , "data/comm_samp_short.csv" )


bs_short<-baseline_samples[,-c(1:200)] %>%
    gather(dtype, div, rich, shan, simp)

