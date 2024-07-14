args <- commandArgs(trailingOnly = TRUE)
runAnalysisDir <- args[1]
replicateID <- args[2]
if (length(args) != 2)
{
  cat ("Rscript R/mannwhitney.R runAnalysisDir 1\n")
  quit("yes")
}
rimapFile <- paste(runAnalysisDir, "/rimap-", replicateID, "-gene.txt", sep="")
sigOutFile <- paste(runAnalysisDir, "/significant-", replicateID, ".txt", sep="")
mwResultFile <- paste(runAnalysisDir, "/mannwhitney-results-", replicateID, ".txt", sep="")
######################

gocats <- read.table("test/melissa/SpyMGAS315_go_bacteria.txt", header=FALSE, col.names=c("gene", "go", "pval"))
gocats <- gocats[gocats$pval <= 1.0e-5,]
descrips <- read.table("test/melissa/SpyMGAS315_go_category_names.txt", sep="\t", header=FALSE, col.names=c("go", "count", "descrip"), quote="")
gocats$go <- as.character(gocats$go)
descrips$go <- as.character(descrips$go)

run.mw <- function(x, gocats, gocats.col="go", descrips=NULL, noisy=FALSE) {
  results <- data.frame() 
  for (gocat in unique(gocats[,gocats.col])) {
    genes <- gocats[gocats[,gocats.col]==gocat, "gene"]
    if (length(genes) >= 10) {
      f <- is.element(x$gene, genes)
      notf <- !f
      if (sum(f) >= 10) {
        wt <- wilcox.test(x[f,"score"], x[!f,"score"], alternative="greater")
        if (noisy) cat(gocat, sum(f), wt$p.value, sep="\t")
        if (!is.null(descrips)) {
          w <- which(descrips[,gocats.col]==gocat)
          if (length(w) != 1L) stop("couldn't find description for ", gocat)
          if (noisy) cat(descrips[w,"descrip"])
          results <- rbind(results, data.frame(gocat=gocat, count=sum(f), p.value=wt$p.value, go.description=descrips[w,"descrip"]))
        } else results <- rbind(results, data.frame(gocat=gocat, count=sum(f), p.value=wt$p.value))
        if (noisy) cat("\n")
      }
    }
  }
  results
}

get.significant <- function(results, p.val) {
  adjust <- p.adjust(results[,p.val], method="BH")
  results$q.val <- adjust
  if (sum(adjust < 0.05) == 0) {
    cat("no significant results\n")
    return(NULL)
  }
  # temp <- results[adjust < 0.05,c(p.val, "count", "go.description")]
  temp <- results[adjust < 0.05,c(p.val, "q.val", "count", "gocat", "go.description")]
  format.data.frame(temp[order(temp[,p.val]),], digits=1, scientific=TRUE)
  # v.return <- data.frame(temp[order(temp[,p.val]),])
  # v.return$p.val <- NULL
  # format.data.frame(v.return, digits=4)
}



  
# x <- read.table("in.gene", header=FALSE)
# num.unique.mw <- run.mw(data.frame(gene=x[,1], score=x[,12]), gocats, descips=descrips)
# x <- read.table("ri1-refgenome4-map.gene", header=FALSE)
# sde.spy.mw <- run.mw(data.frame(gene=x[,1], score=rowSums(x[,10:13])), gocats, descrips=descrips)
# spy.sde.mw <- run.mw(data.frame(gene=x[,1], score=rowSums(x[,14:17])), gocats, descrips=descrips)
# all.mw <- run.mw(data.frame(gene=x[,1], score=x[,18]), gocats)

# 1     2       3               4               5       6       7           8
# gene  all     topology        notopology      sde2spy spy2sde mattsde2spy mattspy2sde
x <- read.table(rimapFile, header=TRUE)
all.mw        <- run.mw(data.frame(gene=x[,1], score=x[,2]), gocats, descrips=descrips)
topology.mw   <- run.mw(data.frame(gene=x[,1], score=x[,3]), gocats, descrips=descrips)
notopology.mw <- run.mw(data.frame(gene=x[,1], score=x[,4]), gocats, descrips=descrips)
sde.spy.mw    <- run.mw(data.frame(gene=x[,1], score=x[,5]), gocats, descrips=descrips)
spy.sde.mw    <- run.mw(data.frame(gene=x[,1], score=x[,6]), gocats, descrips=descrips)
mt.sde.spy.mw <- run.mw(data.frame(gene=x[,1], score=x[,7]), gocats, descrips=descrips)
mt.spy.sde.mw <- run.mw(data.frame(gene=x[,1], score=x[,8]), gocats, descrips=descrips)

# these should all be sorted in the same way but make sure
# num.unique.mw <- num.unique.mw[order(num.unique.mw$gocat),]
all.mw <- all.mw[order(all.mw$gocat),]
topology.mw <- topology.mw[order(topology.mw$gocat),]
notopology.mw <- notopology.mw[order(notopology.mw$gocat),]
mt.sde.spy.mw <- mt.sde.spy.mw[order(mt.sde.spy.mw$gocat),]
mt.spy.sde.mw <- mt.spy.sde.mw[order(mt.spy.sde.mw$gocat),]
sde.spy.mw <- sde.spy.mw[order(sde.spy.mw$gocat),]
spy.sde.mw <- spy.sde.mw[order(spy.sde.mw$gocat),]

numrow <- nrow(all.mw)
if (numrow != nrow(spy.sde.mw) || numrow != nrow(sde.spy.mw) || # numrow != nrow(num.unique.mw) ||
    numrow != nrow(topology.mw) || 
    numrow != nrow(notopology.mw) || 
    numrow != nrow(mt.sde.spy.mw) || 
    numrow != nrow(mt.spy.sde.mw) || 
    sum(all.mw$gocat==topology.mw$gocat) != numrow ||
    sum(all.mw$gocat==notopology.mw$gocat) != numrow ||
    sum(all.mw$gocat==mt.sde.spy.mw$gocat) != numrow ||
    sum(all.mw$gocat==mt.spy.sde.mw$gocat) != numrow ||
    sum(all.mw$gocat==spy.sde.mw$gocat) != numrow ||
    sum(all.mw$gocat==sde.spy.mw$gocat) != numrow)#  ||
    # sum(all.mw$gocat==num.unique.mw$gocat) != numrow)
  stop("results don't have all the same elements in the same order")

# print(length(all.mw$gocat))
# print(length(all.mw$count))
# print(length(all.mw$p.value))
# print(length(sde.spy.mw$p.value))
# print(length(spy.sde.mw$p.value))
# print(length(all.mw$go.description))

all.results <- data.frame(gocat=all.mw$gocat, count=all.mw$count,
                          # p.num.unique=num.unique.mw$p.value,
                          p.all=all.mw$p.value,
                          p.topology=topology.mw$p.value,
                          p.notopology=notopology.mw$p.value,
                          p.mt.sde.spy=mt.sde.spy.mw$p.value,
                          p.mt.spy.sde=mt.spy.sde.mw$p.value,
                          p.sde.spy=sde.spy.mw$p.value,
                          p.spy.sde=spy.sde.mw$p.value,
                          go.description=all.mw$go.description)

# outfile <- "output/cornellf/3/run-analysis/significant-1.txt"
cat("q-value", "count", "go-term", "description", file=sigOutFile, sep="\t")
cat("\n", file=sigOutFile, append=TRUE)
# for (stat in c("p.num.unique", "p.all", "p.sde.spy", "p.spy.sde")) {
for (stat in c("p.all", "p.topology", "p.notopology", "p.sde.spy", "p.spy.sde", "p.mt.sde.spy", "p.mt.spy.sde")) {
  cat("\n", file=sigOutFile, append=TRUE)
  cat("#", stat, "\n", file=sigOutFile, append=TRUE)
  write.table(get.significant(all.results, stat), sigOutFile, row.names=FALSE, quote=FALSE, sep=" & ",
              append=TRUE, col.names=FALSE,eol="\\\\\n")
}

write.table(format.data.frame(all.results, digits=4), file=mwResultFile, quote=FALSE, row.names=FALSE, sep="\t")


# Now check virulence genes
# x <- read.table("in.gene", header=FALSE)
gocats <- read.table("test/melissa/virulent_genes.txt", header=TRUE)
gocats$gene <- as.character(gocats$gene)
gocats <- gocats[is.element(gocats$gene, as.character(x[,1])),]
# x <- x[is.element(as.character(x[,1]), gocats$gene),]
# run.mw(data.frame(gene=x[,1], score=x[,12]), gocats, gocats.col="virulent")

# x <- read.table("ri1-refgenome4-map.gene", header=FALSE)
x <- read.table(rimapFile, header=TRUE)
gocats <- read.table("test/melissa/virulent_genes.txt", header=TRUE)
gocats$gene <- as.character(gocats$gene)
gocats <- gocats[is.element(gocats$gene, as.character(x[,1])),]
x <- x[is.element(as.character(x[,1]), gocats$gene),]
# run.mw(data.frame(gene=x[,1], score=rowSums(x[,10:13])), gocats, gocats.col="virulent")
# run.mw(data.frame(gene=x[,1], score=rowSums(x[,14:17])), gocats, gocats.col="virulent")
# run.mw(data.frame(gene=x[,1], score=x[,18]), gocats, gocats.col="virulent")
run.mw(data.frame(gene=x[,1], score=x[,5]), gocats, gocats.col="virulent")
run.mw(data.frame(gene=x[,1], score=x[,6]), gocats, gocats.col="virulent")
run.mw(data.frame(gene=x[,1], score=x[,3]), gocats, gocats.col="virulent")
