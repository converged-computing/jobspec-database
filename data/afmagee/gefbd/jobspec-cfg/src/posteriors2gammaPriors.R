# This is designed to be called with Rscript with several arguments
# arg1: path to and prefix name of logfiles (e.g. ~/folder/output/analysis_with_x_dataset)
# arg2: path to ouput file

# Call this script from phyload/simulation_study

# This prints several

# Get arguments
args = commandArgs(trailingOnly=TRUE)

if (!length(args) == 2) {
  stop("This script requires 2 arguments")
}

gammaMoments2Params <- function(m,variance) {
  b <- m/variance
  a <- m * b

  res <- c(a,b)
  names(res) <- c("alpha","beta")

  return(res)
}

getConcatenatedLogFile <- function(rb.file.prefix) {
  # Get all log files matching prefix (accounts for replicate runs)
  # This will not work on windows
  folder <- dirname(rb.file.prefix)
  rb_logs <- list.files(folder,full.names=TRUE)
  rb_logs <- rb_logs[grepl(basename(rb.file.prefix),rb_logs)]
  rb_logs <- rb_logs[grepl(".log",rb_logs)]
  rb <- do.call(rbind,lapply(rb_logs,read.table,stringsAsFactors=FALSE,header=TRUE))
  return(rb)
}

getRevGammaPrior <- function(rb.posterior,parameter.names,variance.inflation.factor=2.0) {
  outstring <- ""

  for (i in 1:length(parameter.names)) {
    key <- paste0("^",parameter.names[i],"$")
    x <- rb.posterior[,grepl(key,names(rb.posterior))]

    m <- mean(x)
    v <- var(x) * variance.inflation.factor

    alphabeta <- gammaMoments2Params(m,v)

    outstring <- c(outstring,paste0(parameter.names[i],"_hyperprior_alpha <- ",alphabeta[1]))
    outstring <- c(outstring,paste0(parameter.names[i],"_hyperprior_beta <- ",alphabeta[2]))
  }

  return(paste0(outstring,sep="\n"))

}

rb.log <- getConcatenatedLogFile(args[1])
priors.string <- getRevGammaPrior(rb.log,c("speciation_rate","extinction_rate","fossilization_rate"))
cat(priors.string,file=args[2])
