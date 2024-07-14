# rxx03 series manipulates the variance of the prime distribution (a=0,b=1,c=2,d=3)

# Note that options can be passed, but if the guideFl option is supplied, any
# options saved in that file override all options passed here. The only
# exception is the group argument, which specifies which 

# Libraries ====

# add this to the search path for summit computing.
#.libPaths(c(.libPaths(), '/projects/chme2908/R_libs'))

#library(CMUtils)
pkgLst <- c(
  'car',
  'optparse',
  #'psych',
  'reshape',
  'pbdMPI'
  #'filelock'
  #'gtheory',
  #'parallel'
  #'pryr'
)
for (p in pkgLst) {
  library(p, character.only = T, quietly = T)
}

# Date String ====

suppressWarnings({
  dateStr <- format(Sys.time(), format='%Y-%m-%d_%H.%M.%S')
  comm.print(paste('Run start time:', format(Sys.time(), format='%Y-%m-%d_%H:%M:%S')))
  startTime <<- proc.time()["elapsed"]
})

# Working Directory ====
cmDir <- '~chrismellinger/GoogleDrive/ImplicitMeasureReliability/'
corcDir <- '/scratch/summit/chme2908/ImplicitMeasureReliability/'
if (dir.exists(cmDir)) {
  setwd(cmDir)
} else if (dir.exists(corcDir)) {
  setwd(corcDir)
}
comm.print(paste0('Working directory: ', getwd()))

# Get arguments ====

opts = 
  optionList = list(
    make_option(c("--nprim"), type="integer", help="number of primes"),
    make_option(c("--n.iter"), type="integer", 
                help="
                Number of iterations (the iteration number to end on). Note that
                this is multiplied by the number of variances in the variance
                list for the final number of iterations.
                "),
    make_option(c("--iter.start"), type="integer", 
                help="
                Iteration number to begin at. This is with reference to the
                total number of iterations n.iter * length(varianceLst).
                "),
    make_option(c("--ncores"), type="integer", help="number of cores to use"),
    make_option(c("--ntarg"), type="integer", help="number of targets"),
    make_option(c("--nsubj"), type="integer", help="number of subjects"),
    make_option(c("--nreps"), type="integer", help="number of subjects"),
    #make_option(c("--pvarHi"), type="integer", help="variance of primes in 'high' condition"),
    #make_option(c("--pvarLo"), type="integer", help="variance of primes in 'low' condition")
    make_option(c("--varianceLst"), type="character", help="comma separated values"),
    make_option(c("--dateStr"), type="character", help="string representing the date"),
    make_option(c("--guideFl"), type="character", 
                help="string of file name to use as guide matrix"),
    make_option(c("--group"), type="integer",
                help="the group number that should be processed"),
    make_option(c("--estFlNm"), type="character", 
                help="string of file name to store results"),
    make_option(c("--dataDir"), type="character", 
                help="Directory where the generated data is.")
  )
optParser = OptionParser(option_list = optionList)
args = parse_args(optParser)

# Check the one required argument
if (is.null(args$n.iter)) {
  stop(
    paste("Must specify n.iter!")
  )
}
n.iter <- args$n.iter

# Default simulation parameters ====

# For the parrallelization
#ncores <<- 2 * n.iter - 1 # This should be one less than is actually 
# available. Reserve one core for the main process.

# These are only needed if generating the data in files ahead of time.

dataDir <- 'GeneratedData'
timingFile <- 'Timing.txt'; timingFileLock <- paste0(timingFile, '.lock')
timingDir <- 'TimingInfo'
# scratchDirHi <- './scratch/HighVarData'
#if (!dir.exists(resultDir)) {dir.create(resultDir)}
#if (!dir.exists(dataDir)) {dir.create(dataDir)}
if (!file.exists(timingFile)) {file.create(timingFile)}
if (!file.exists(timingFileLock)) {file.create(timingFileLock)}
#if (!dir.exists(timingDir)) {dir.create(timingDir)}

nsubj <<- 15
nprim <<- 2
npcat <<- 2
ntarg <<- 2
ntcat <<- 2
nreps <<- 2 # should be even number

svar <<- 1
# pvarLo <<- 1
# pvarHi <<- pvarLo * 2
tvar <<- 1
evar <<- 1
varianceLst <<- "1"
iter.start <<- 1

# If the guide file is not provided, the program will assume that it should 
# generate data on its own
guideFl <- NULL

# Substitute Provided Arguments for Default Values ====

# Overwrite the defaults when they are provided.
for (var in names(args)) {
  assign(var, args[var][[1]])
}

# process the variance list parameter
varianceLst <<- as.numeric(unlist(strsplit(varianceLst, ",")))

# Substitute all args for the ones in guideFl ====

if (!is.null(guideFl)) {
  load(guideFl)
}

# Print Important Parameters for the Logs ====
varLst <- c(
  #"ncores",
  "n.iter",
  "nsubj",
  "nprim",
  "npcat",
  "ntarg",
  "ntcat",
  "nreps",
  "svar",
  #"pvarLo",
  #"pvarHi",
  "tvar",
  "evar",
  "varianceLst",
  "iter.start",
  "dateStr",
  "guideFl",
  "group",
  "estFlNm",
  "dataDir"
)
for (var in varLst) {
  if (exists(var)) {
    comm.print(paste0(var, ": ", get(var)))
  }
}

# Random Seed & Data Directory ====

# Random number considerations. This will make the result reproducible 
# and also ensure that each iteration is reasonably independent.
# RNGkind("L'Ecuyer-CMRG")
comm.set.seed(593065038)
#if (!dir.exists(dateStr)) {dir.create(dateStr)}

# profileFl <- paste0(dateStr, '_profile.txt')

# Build functions for the job ====

genData = function(
                  nsubj,
                  nprim,
                  npcat,
                  ntarg,
                  ntcat,
                  nreps,# should be even number

                  svar,
                  pvar,
                  tvar,
                  evar
                  ) {
  # subject differences ====
  
  snum <- 1:nsubj
  pnum <- 1:nprim
  tnum <- 1:ntarg
  rnum <- 1:nreps
  
  prej <- rnorm(snum,0,svar)
  basert <- rnorm(snum,0,1)
  subj <- data.frame(snum,prej,basert)
  
  
  # prime differences ====
  pprot <- rep(rnorm(nprim,0,pvar),npcat)
  pcat <- rep(rnorm(npcat,0,1),each=nprim)
  prime <- data.frame(pnum,pcat,pprot)
  
  
  # target differences ====
  tvaln <- rep(rnorm(ntarg,0,tvar),ntcat)
  tcat <- rep(rnorm(ntcat,0,1),each=ntarg)
  target <- data.frame(tnum,tcat,tvaln)
  
  
  # build integrated data file ====
  d <- expand.grid(rnum=rnum,tnum=tnum,pnum=pnum,snum=snum)
  d <- merge(d,target)
  d <- merge(d,prime,by.x="pnum",by.y="pnum")
  d <- merge(d,subj,by.x="snum",by.y="snum")
  
  d$error <- rnorm(nrow(d),0,evar)
  # d$rt <- 600 + 1*(d$pcat*d$tcat) + 1*(d$pcat*d$tcat*d$prej) + 
  #   1*(d$pcat*d$tcat*d$prej*d$tvaln) + 1*(d$pcat*d$tcat*d$prej*d$pprot) + 
  #   1*(d$pcat*d$tcat*d$prej*d$pprot*d$tvaln) + 5*d$error
  
  # This formula was the first attempt at using real-world values.
  d$rt <- 6.4 + 0.07*(d$prej) + 0.00*(d$pcat) + 0.02*(d$tcat) + 0.02*(d$pprot) +
    0.02*(d$tvaln) + 0.00*(d$prej*d$pcat) + 0.00*(d$prej*d$tcat) +
    0.04*(d$pcat*d$tcat) + 0.04*(d$prej*d$pprot) + 0.02*(d$pprot*d$tcat) +
    0.00*(d$prej*d$tvaln) + 0.00*(d$pcat*d$tvaln) + 0.00*(d$pprot*d$tvaln) +
    0.08*(d$prej*d$pcat*d$tcat) + 0.00*(d$prej*d$pcat*d$tcat*d$tvaln) +
    0.06*(d$prej*d$pcat*d$tcat*d$pprot) +
    0.05*(d$pcat*d$tcat*d$prej*d$pprot*d$tvaln) + 0.25*d$error
  
  # These are the real world values based on the heirarchical ordering approach.
   d <- within(d, {
  #   rt <- 6.4 + 
  #     .09449*(prej) + .09449*(pcat) + .09449*(tcat) + 
  #     .09449*(pprot) + .09449*(tvaln) + 
  #     # two ways
  #     .08452*(prej*pcat) + 
  #     .08452*(prej*tcat) + .08452*(prej*pprot) + .08452*(prej*tvaln) + 
  #     .08452*(pcat*tcat) + .08452*(pcat*pprot) + .08452*(pcat*tvaln) + 
  #     .08452*(tcat*pprot) + .08452*(tcat*tvaln) + 
  #     .08452*(pprot*tvaln) + 
  #     # three ways
  #     0.07319*(prej*pcat*tcat) + 
  #     0.07319*(prej*pcat*pprot) + 0.07319*(prej*pcat*tvaln) + 
  #     0.07319*(prej*tcat*pprot) + 0.07319*(prej*tcat*tvaln) + 
  #     0.07319*(prej*pprot*tvaln) + 0.07319*(pcat*tcat*pprot) + 
  #     0.07319*(pcat*tcat*tvaln) + 0.07319*(pcat*pprot*tvaln) + 
  #     0.07319*(tcat*pprot*tvaln) + 
  #     # four ways
  #     0.05976*(prej*pcat*tcat*pprot) + 
  #     0.05976*(prej*pcat*tcat*tvaln) + 
  #     0.05976*(prej*pcat*pprot*tvaln) + 
  #     0.05976*(prej*tcat*pprot*tvaln) + 
  #     0.05976*(pcat*tcat*pprot*tvaln) + 
  #     # five way
  #     .04226*(pcat*tcat*prej*pprot*tvaln) + 
  #     0.10351*error
    
    # variance decomposition ====
    snum <- as.factor(snum)
    pcat <- as.factor(pcat)
    pnum <- as.factor(pnum)
    tcat <- as.factor(tcat)
    tnum <- as.factor(tnum)
  })
  
  contrasts(d$snum) <- contr.poly
  contrasts(d$pcat) <- contr.poly
  contrasts(d$pnum) <- contr.poly
  contrasts(d$tcat) <- contr.poly
  contrasts(d$tnum) <- contr.poly
  
  return(d)
}

finishCalcs <- function (est) {
  est <- within(est, {
    var.resid <- ms.resid
    var.snpctc <- 
      (ms.snpctc - ms.snpctn - ms.sntcpn + ms.snpntn) / 
      (nprim*ntarg*nreps)
    var.snpctn <- (ms.snpctn - ms.snpntn) / (nprim*nreps)
    var.sntcpn <- (ms.sntcpn - ms.snpntn) / (ntarg*nreps)
    var.snpntn <- (ms.snpntn - ms.resid) / (nreps)
    
    # recode negative variances to zero (this will bias the estimates but is 
    # necessary to avoid negative reliabilities)
    var.snpctc[var.snpctc<0] <- 0
    var.snpctn[var.snpctn<0] <- 0
    var.sntcpn[var.sntcpn<0] <- 0
    var.snpntn[var.snpntn<0] <- 0
    
    rxxmse <- (ms.snpctc-ms.resid) / ms.snpctc
    
    rxxvar <- 
      (var.snpctc + (var.snpctn / ntarg) + (var.sntcpn/ nprim) + 
         (var.snpntn / (nprim*ntarg))
      ) / 
      (var.snpctc + (var.snpctn / ntarg) + (var.sntcpn / nprim) + 
         (var.snpntn / (nprim*ntarg)) + var.resid / (nprim*ntarg*nreps)
      )
    
    rxxvar.prand <- 
      var.snpctc / 
      (var.snpctc + (var.snpctn / ntarg) + (var.sntcpn / nprim) + 
         (var.snpntn / (nprim*ntarg)) + var.resid / (nprim*ntarg*nreps)
      )
  })
  
  return(est)
}

renameEstCols <- function (est) {
  colnames(est) <- c(
    "ms.int",
    "ms.sn",
    "ms.pc",
    "ms.tc",
    "ms.pn",
    "ms.tn",
    
    "ms.snpc",
    "ms.sntc",
    "ms.pctc",
    "ms.snpn",
    "ms.pcpn",
    "ms.tcpn",
    "ms.sntn",
    "ms.pctn",
    "ms.tctn",
    "ms.pntn",
    
    "ms.snpctc",
    "ms.snpcpn",
    "ms.sntcpn",
    "ms.pctcpn",
    "ms.snpctn",
    "ms.sntctn",
    "ms.pctctn",
    "ms.snpntn",
    "ms.pcpntn",
    "ms.tcpntn",
    
    "ms.snpctcpn",
    "ms.snpctctn",
    "ms.snpcpntn",
    "ms.sntcpntn",
    "ms.pctcpntn",
    
    "ms.snpctcpntn",
    "ms.resid",
    "r_sh","r_pf", "nprim", "ntarg", "nreps", "var", "runID"
  )
  return(est)
}

modelFn <- function (d, i=-1) 
{
  initTime <- proc.time()[3]
  m1 <- lm(rt ~ snum * pcat * tcat * pnum * tnum, data=d)
  lmTm <- round((proc.time()[3] - initTime) / 60, 2)
  #print(paste0('Iteration ', i, ' lm() call finished: ', tm, ' minutes'))
  
  initTime <- proc.time()[3]
  my.anova1 <- Anova(m1, type="III", singular.ok = TRUE)
  anovaTm <- round((proc.time()[3] - initTime) / 60, 2)
  #print(paste0('Iteration ', i, ' Anova() call finished: ', tm, ' minutes'))
  numPar <- nrow(my.anova1)
  est <- my.anova1[1:numPar,'Sum Sq'] / my.anova1[1:numPar,'Df']
  names(est) <- rownames(my.anova1)
  
  # split-half reliability
  
  # Sample half of the trials for each participant by replication cell.
  d$half <- 1
  for (s in unique(d$snum)) {
    curRows <- row.names(d[d$snum==s,])
    sampledRows <- sample(curRows, size=length(curRows)/2, replace=F)
    d[sampledRows, 'half'] <- 2
  }
  
  d1 <- data.frame(snum=unique(d$snum))

  rts <- tapply(d$rt, INDEX=list(d$snum, d$half, d$pcat, d$tcat), mean, na.rm=T)
  #d1 <- cast(d, snum ~ half + pcat + tcat, mean, value="rt")
  d1 <- data.frame(
    snum = dimnames(rts)[[1]]
  )
  d1 <- within(d1, {
    h1p1t1 = rts[snum,1,1,1]
    h1p1t2 = rts[snum,1,1,2]
    h1p2t1 = rts[snum,1,2,1]
    h1p2t2 = rts[snum,1,2,2]
    
    h2p1t1 = rts[snum,2,1,1]
    h2p1t2 = rts[snum,2,1,2]
    h2p2t1 = rts[snum,2,2,1]
    h2p2t2 = rts[snum,2,2,2]
    
    half1 <- (h1p1t1 - h1p1t2) - (h1p2t1 - h1p2t2)
    half2 <- (h2p1t1 - h2p1t2) - (h2p2t1 - h2p2t2)
  })
  rSh <- cor(d1$half1,d1$half2)
  est['r_sh'] <- rSh

  # parallel forms reliability, must recode replications to only 
  # 2 values (odd and even)
  d$rnumx <- as.numeric(d$rnum) %% 2
  d$rnumx[d$rnumx==0] <- 2
  d1 <- cast(d, snum ~ rnumx + pcat + tcat, mean, value="rt")
  d1$rnum1 <- (d1[,2] - d1[,3]) - (d1[,4] - d1[,5])
  d1$rnum2 <- (d1[,6] - d1[,7]) - (d1[,8] - d1[,9])
  
  # Build the return vector
  est['r_pf'] <- cor(d1$rnum1,d1$rnum2)
  
  rm(m1, my.anova1, numPar, d, d1)
  
  # Also return the timing info
  tming <- list(lmTm=lmTm, anovaTm=anovaTm)
  return(list(est=est, tming=tming))
}

iterFn <- function (i, curPvar, d=NULL) {
  initTm <- proc.time()[3]
  if (is.null(d)) {
    d <- genData(
      nsubj = nsubj,
      nprim = nprim,
      npcat = npcat,
      ntarg = ntarg,
      ntcat = ntcat,
      nreps = nreps, # should be even number
      
      svar = svar,
      pvar = curPvar,
      tvar = tvar,
      evar = evar
    )
  }
  #print(paste0('Iteration ', i, ' data gen time: ', tm))
  
  #initTm <- proc.time()[3]
  curEst <- modelFn(d, i=i)$est
  curEst['nprim'] <- nprim
  curEst['ntarg'] <- ntarg
  curEst['nreps'] <- nreps
  
  # Save the timing info in a separate file
  
  
  rm(d); gc();
  return(curEst)
}

 
# Parallelize! the modeling part... ====

init()

# Make a guiding list of variances and iteration numbers.

if (is.null(guideFl)) {
  iterVec <- iter.start:(n.iter*length(varianceLst))
  guideMat <- data.frame(
    i=iterVec,
    variance=rep(varianceLst, times=n.iter)[iterVec]
  )
}

# Subset to the groups if we need it.

if (exists("group")) {
  if (!("group" %in% colnames(guideMat))) {
    comm.print("guideMat does not contain a group variable. Processing all rows")
  } else {
    guideMat <- guideMat[guideMat$group==group,]
  }
}

time.proc <- system.time({
  id <- get.jid(nrow(guideMat))
  estLst <- lapply(id, 
                     function (i) {
                       curVar <- guideMat$variance[i]
                       iterNum <- guideMat$i[i]
                       if (is.null(guideFl)) {
                         return(
                           c(iterFn(iterNum, curPvar=curVar), 
                             var=curVar, runID=i)
                         )
                       } else {
                         load(guideMat$flNm[i])
                         return(
                           c(iterFn(iterNum, curPvar=curVar, d=d), 
                             var=curVar, runID=i)
                         )
                       }
                     }
  )
  estLst <- unlist(allgather(estLst), recursive=F)
})
comm.print(time.proc)
#file.remove(timingFileLock)

endTime <- suppressWarnings(format(Sys.time(), format='%Y-%m-%d_%H.%M.%S'))
comm.print(paste("Run finished", endTime))

# Finalize The Dataframe ====

estMat <- Reduce(rbind, estLst)
row.names(estMat) <- 1:nrow(estMat)
est <- as.data.frame(estMat)

est <- renameEstCols(est)
est <- finishCalcs(est)

if (exists("estFlNm")) {
  flName <- estFlNm
} else {
  flName <- paste0('est_', dateStr, '_.RData')
}

if (!dir.exists(dirname(flName))) {
  dir.create(dirname(flName), recursive = T)
}

comm.print(paste0('File: ', flName))
save(est, file = flName)

finalize()
