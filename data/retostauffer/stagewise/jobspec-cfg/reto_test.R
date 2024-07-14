
library("argparse")

parser <- ArgumentParser("Allows console args for testing")
parser$add_argument("-b", "--batchsize", type = "integer", default = 100,
                    help = "Size of the batch; if < 100 it takes nobs / batchsize!")
parser$add_argument("-m", "--maxit", type = "integer", default = 10,
                    help = "Number of iterations, defaults to 10")
parser$add_argument("-n", "--nobs", type = "integer", default = 1e5,
                    help = "Number of observations, defaults to 1e5")
parser$add_argument("-p", type = "integer", default = 1e2,
                    help = "Number of parameters (covariates), defaults to 1e2")
parser$add_argument("--ff", action = "store_true",
                    help = "Should ff be used (ffdf instead of data.frame)?")
parser$add_argument("--bigmem", action = "store_true",
                    help = "Should bigmem be used (big.matrix instead of data.frame)?")
parser$add_argument("--noquick", action = "store_false", dest = "quick_ffdf",
                    help = "Should quick_ffdf NOT be used (data set is model frame)")
parser$add_argument("-o", "--original", action = "store_true",
                    help = "Should the original sdr package be used?")

args <- parser$parse_args()
print(str(args))
if (is.null(args$batchsize)) {
    args$print_help()
    stop("-b/--batchsize missing")
} else if (args$batchsize <= 0) {
    stop("batchsize must be positive")
} else if (args$batchsize < 100) {
    args$batchsize <- args$nobs / args$batchsize
}
if (args$maxit < 5) stop("-m/--maxit must be 5 or larger")


# Re-loading the package (for development purposes)
cat("Loading sdr package via devtools\n")
library("devtools")
if (args$original) {
    load_all("sdrorig")
} else {
    load_all("sdr")
}

cat("Sourcing testfunctions (to load test data)\n")
source("testfunctions.R")
t <- Sys.time()
cat("Loading testdata with nobs = ", args$nobs, " p = ", args$p,
    " ff = ", args$ff, " bigmem = ", args$bigmem, "\n")
xxx <- get_testdata(args$nobs, args$p, args$ff, args$bigmem)
t <- as.numeric(Sys.time() - t, units = "mins")
cat("\n\nJust reading the test data took ", round(t, 2), " minutes\n")

print(head(xxx$data[, 1:5]))

## Variable selection with correlation filtering
## and best subset updating.
t <- Sys.time()
set.seed(666)
library("peakRAM")
cat("Calling sdr now ...\n")
pk <- peakRAM({
    b <- sdr(formula   = xxx$formula,
             data      = xxx$data,
             CF        = TRUE,
             updating  = "bs",
             family    = NO,
             batch_ids = args$batchsize,
             maxit     = args$maxit,
             quick_ffdf = args$quick_ffdf)
})
t <- as.numeric(Sys.time() - t, units = "mins")
cat("\n\nFull estimation took me ", round(t, 2), " minutes in total\n")

cat("\n\n")
print(pk)



