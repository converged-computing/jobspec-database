#!/usr/bin/env Rscript

# This is a helper script to run the pipeline.
# Choose how to execute the pipeline below.
# See https://books.ropensci.org/targets/hpc.html
# to learn about your options.


## load docopt and remotes (or devtools) from CRAN
suppressMessages({
  library(docopt)               # we need docopt (>= 0.3) as on CRAN
  library(targets)
  library(clustermq)
  library(RcppParallel)
})

## configuration for docopt
doc <- "Usage: run.R [-h] [-x] [-n NCORES] [-s SCHEDULER]
-n --ncores NCORES          Number of threads to use 
                            [default: RcppParallel::defaultNumThreads()]
-s --scheduler SCHEDULER    scheduler to use 
                            [default: targets::use_targets_scheduler()]
-h --help                   show this help text
-x --usage                  show help and short example usage"

opt <- docopt(doc)			# docopt parsing

if (opt$usage) {
  cat(doc, "\n\n")
  cat("where NCORES is the number of workers sent to tar_make_clustermq().
  and scheduler can be one of multicore/multiprocess
Basic usage:
  run.R -n 12 -s multicore
\n")
  q("no")
}


if (length(opt$ncores) == 1) {
  opt$ncores <- as.numeric(opt$ncores)
} else {
  opt$ncores <- RcppParallel::defaultNumThreads()
}

if (length(opt$scheduler) == 1 && as.character(opt$scheduler) %in%  c("multicore","multiprocess")) {
  ## as littler can now read ~/.littler.r and/or /etc/littler.r we can preset elsewhere
  opt$scheduler <- as.character(opt$scheduler)
}else if(is.null(opt$scheduler)){
  opt$scheduler <- targets::use_targets_scheduler()
}else{
  warning("choose a valid option for scheduler, see run.R -x for info")
  opt$scheduler <- targets::use_targets_scheduler()
}

options(clustermq.scheduler = opt$scheduler   )
targets::tar_make_clustermq(workers = as.numeric(opt$ncores))


# targets::tar_make_clustermq(workers = as.numeric(ncores)) # nolint
# targets::tar_make_future(workers = 2) # nolint


