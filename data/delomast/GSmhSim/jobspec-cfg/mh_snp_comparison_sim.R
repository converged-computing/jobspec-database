# evaluating SNP and microhap panels for imputation for GS
# 
# 

.libPaths(c("/project/oyster_gs_sim/R_packages/4.3/", .libPaths()))
Rlibrarylocation <- "/project/oyster_gs_sim/R_packages/4.3/"
cmdArgs <- commandArgs(trailingOnly=TRUE)

#' # Script parameters given on the command line
#' #' @param randSeed random seed to set R's random number generator
#' #' @param iterationNumber used to set unique file output names
#' #' @param localTempDir directory to write temp files to
#' #' @param inputVCFpath path to VCF with data to seed simulation (define founder population)
randSeed <- cmdArgs[1]
iterationNumber <- cmdArgs[2]
localTempDir <- cmdArgs[3]
inputVCFpath <- cmdArgs[4]


# parameters for interactive testing
# Rlibrarylocation <- NULL
# randSeed <- 7
# iterationNumber <- 1
# localTempDir <- "./"
# inputVCFpath <- "./ngulf.vcf"

library(AlphaSimR, lib.loc=Rlibrarylocation)
library(tidyverse, lib.loc=Rlibrarylocation)
library(optiSel, lib.loc=Rlibrarylocation)
library(AllocateMate, lib.loc=Rlibrarylocation)

source("utils.R")

# set random seed
set.seed(as.numeric(randSeed))

# window for defining microhaplotypes
windSize <- 125

# define chr sizes and numbers of loci
if(grepl("mbpLowDepth.vcf$", inputVCFpath)){
	num <- data.frame(chr = c("NC_047559.1",
							  "NC_047560.1",
							  "NC_047561.1",
							  "NC_047562.1",
							  "NC_047563.1",
							  "NC_047564.1",
							  "NC_047565.1",
							  "NC_047566.1",
							  "NC_047567.1",
							  "NC_047568.1"),
					  len = c(55785328,
					  		73222313,
					  		58319100,
					  		53127865,
					  		73550375,
					  		60151564,
					  		62107823,
					  		58462999,
					  		37089910,
					  		57541580)
	)
	HDpanelSize <- 35000
	num_choose_qtl <- 1000
	prefix <- "MBPld"
	# LD panel sizes to test
	panelSizes <- seq(50, 450, 100)
	# this will be the maximum number of broodstock used AFTER the founder generation
	# the founders are just those input with the vcf
	nBrood <- 100
} else if(grepl("mbp.vcf$", inputVCFpath)){
	num <- data.frame(chr = c("NC_047559.1",
							  "NC_047560.1",
							  "NC_047561.1",
							  "NC_047562.1",
							  "NC_047563.1",
							  "NC_047564.1",
							  "NC_047565.1",
							  "NC_047566.1",
							  "NC_047567.1",
							  "NC_047568.1"),
					  len = c(55785328,
					  		73222313,
					  		58319100,
					  		53127865,
					  		73550375,
					  		60151564,
					  		62107823,
					  		58462999,
					  		37089910,
					  		57541580)
	)
	HDpanelSize <- 25000
	num_choose_qtl <- 1000
	prefix <- "MBPrad"
	# LD panel sizes to test
	# this will be the maximum number of broodstock used AFTER the founder generation
	# the founders are just those input with the vcf
	nBrood <- 200 
	panelSizes <- seq(50, 450, 100)
} else if(grepl("ngulf.vcf$", inputVCFpath)) {
	num <- data.frame(chr = c("NC_035789.1",
							  "NC_035780.1",
							  "NC_035781.1",
							  "NC_035782.1",
							  "NC_035783.1",
							  "NC_035784.1",
							  "NC_035785.1",
							  "NC_035786.1",
							  "NC_035787.1",
							  "NC_035788.1"),
					  len = c(32650045, 
					  		65668440,
					  		61752955, 
					  		77061148, 
					  		59691872, 
					  		98698416, 
					  		51258098, 
					  		57830854, 
					  		75944018, 
					  		104168038)
	)
	HDpanelSize <- 3000
	num_choose_qtl <- 100
	prefix <- "Ngulf"
	# LD panel sizes to test
	panelSizes <- seq(50, 450, 100)
	# this will be the maximum number of broodstock used AFTER the founder generation
	# the founders are just those input with the vcf
	nBrood <- 200 
} else if(grepl("allPhased_eobc.vcf$", inputVCFpath)) {
	num <- data.frame(chr = c("NC_035789.1",
							  "NC_035780.1",
							  "NC_035781.1",
							  "NC_035782.1",
							  "NC_035783.1",
							  "NC_035784.1",
							  "NC_035785.1",
							  "NC_035786.1",
							  "NC_035787.1",
							  "NC_035788.1"),
					  len = c(32650045, 
					  		65668440,
					  		61752955, 
					  		77061148, 
					  		59691872, 
					  		98698416, 
					  		51258098, 
					  		57830854, 
					  		75944018, 
					  		104168038)
	)
	HDpanelSize <- 40000
	num_choose_qtl <- 1000
	prefix <- "EOBC_all"
	# LD panel sizes to test
	panelSizes <- seq(50, 450, 100)
	# this will be the maximum number of broodstock used AFTER the founder generation
	# the founders are just those input with the vcf
	nBrood <- 200 
} else if(grepl("allPhased_eobc_subset.vcf$", inputVCFpath)) {
	num <- data.frame(chr = c("NC_035789.1",
							  "NC_035780.1",
							  "NC_035781.1",
							  "NC_035782.1",
							  "NC_035783.1",
							  "NC_035784.1",
							  "NC_035785.1",
							  "NC_035786.1",
							  "NC_035787.1",
							  "NC_035788.1"),
					  len = c(32650045, 
					  		65668440,
					  		61752955, 
					  		77061148, 
					  		59691872, 
					  		98698416, 
					  		51258098, 
					  		57830854, 
					  		75944018, 
					  		104168038)
	)
	HDpanelSize <- 40000
	num_choose_qtl <- 1000
	prefix <- "EOBC_subset"
	# LD panel sizes to test
	panelSizes <- seq(50, 450, 100)
	# this will be the maximum number of broodstock used AFTER the founder generation
	# the founders are just those input with the vcf
	nBrood <- 200 
} else if (grepl("allPhased_oysterChina.vcf$", inputVCFpath)) {
	num <- data.frame(chr = c("NC_047559.1",
							  "NC_047560.1",
							  "NC_047561.1",
							  "NC_047562.1",
							  "NC_047563.1",
							  "NC_047564.1",
							  "NC_047565.1",
							  "NC_047566.1",
							  "NC_047567.1",
							  "NC_047568.1"),
					  len = c(55785328,
					  		73222313,
					  		58319100,
					  		53127865,
					  		73550375,
					  		60151564,
					  		62107823,
					  		58462999,
					  		37089910,
					  		57541580)
	)
	HDpanelSize <- 40000
	num_choose_qtl <- 1000
	prefix <- "CHI"
	# LD panel sizes to test
	panelSizes <- seq(50, 450, 100)
	# this will be the maximum number of broodstock used AFTER the founder generation
	# the founders are just those input with the vcf
	nBrood <- 100 
} else if (grepl("allPhased_atlSalm.vcf$", inputVCFpath)) {
	num <- data.frame(chr = c("CM037938.1",
							  "CM037939.1",
							  "CM037940.1",
							  "CM037941.1",
							  "CM037942.1",
							  "CM037943.1",
							  "CM037944.1",
							  "CM037945.1",
							  "CM037946.1",
							  "CM037947.1",
							  "CM037948.1",
							  "CM037949.1",
							  "CM037950.1",
							  "CM037951.1",
							  "CM037952.1",
							  "CM037953.1",
							  "CM037954.1",
							  "CM037955.1",
							  "CM037956.1",
							  "CM037957.1",
							  "CM037958.1",
							  "CM037959.1",
							  "CM037960.1",
							  "CM037961.1",
							  "CM037962.1",
							  "CM037963.1",
							  "CM037964.1"),
					  len = c(121099132,
					  		93940316,
					  		105647486,
					  		89649482,
					  		93193300,
					  		93615478,
					  		71324303,
					  		75551214,
					  		158363262,
					  		127569415,
					  		106954443,
					  		102706041,
					  		116099125,
					  		108040617,
					  		112367654,
					  		99331370,
					  		75857577,
					  		94174473,
					  		91428890,
					  		99150133,
					  		62426268,
					  		66053996,
					  		103339795,
					  		50520442,
					  		55990407,
					  		104044219,
					  		49143029)
	)
	HDpanelSize <- 60000
	num_choose_qtl <- 1500
	prefix <- "ATL"
	# LD panel sizes to test
	panelSizes <- seq(50, 750, 100)
	# this will be the maximum number of broodstock used AFTER the founder generation
	# the founders are just those input with the vcf
	nBrood <- 100 
	# this expansion pop was used for testing
	# kept code in in case of revisiting
	# expand pop
	# print("expanding pop")
	# expandPop(inputVCFpath, numInds = 200, numGens = 5,
	# 		  num = num, 
	# 		  vcfOut = paste0(localTempDir, "/", "temp", iterationNumber, "/founderVCF.txt"),
	# 		  numFinal = 200)
	# # redirect to expanded VCF
	# inputVCFpath <- paste0(localTempDir, "/", "temp", iterationNumber, "/founderVCF.txt")
} else {
	stop("not set up for input VCF")
}

nOffspringPerCross <- 50
nGenerations <- 3

print(Sys.time())
print("begin loading genotypes")

# evaluate VCF: num SNPs, HE of snps and microhaps
system2("python3", args = c("./eval_vcf.py", inputVCFpath, paste0(localTempDir, "/", "temp", iterationNumber, "/"),
							windSize, randSeed, num_choose_qtl))
snpEval <- read_tsv(paste0(localTempDir, "/", "temp", iterationNumber, "/HeSNP.txt"),
					col_names = c("chr", "pos", "lineNum", "He", "qtl"), col_types = "cdddl")
mhEval <- read_tsv(paste0(localTempDir, "/", "temp", iterationNumber, "/HeMH.txt"),
				   col_names = c("chr", "pos", "He", "lineNum", "numSNPs", "aRich"), col_types = "ccdcdd")

# testing with removal of highly variable microhaplotypes
mhEval <- mhEval %>% filter(aRich <= numSNPs + 1) %>%
	select(-numSNPs, -aRich)

# QTL
qtl <- snpEval %>% filter(qtl) 
# choose HD panel - random SNPs
HDpanel <- snpEval %>% filter(!qtl) %>% slice_sample(n = HDpanelSize)

## choose largest panel first, then choose smaller panels by
##   selecting subsets of loci (in order of being selected) 
##   in each chromosome
## This is equivalent to running the algorithm multiple times for the 
##    different panel sizes but reduces computation time

# allocate number of loci for each chromosome
# proportional to chr length
dfOut <- num %>% mutate(num = (len * max(panelSizes)) / sum(len), num = round(num)) 
# account for rounding error
dfOut$num[which.max(dfOut$num)] <- dfOut$num[which.max(dfOut$num)] + max(panelSizes) - sum(dfOut$num)
# Define largest panels
largestMH <- greedyChooseLoci(num = dfOut, locusEval = mhEval)
largestSNP <- greedyChooseLoci(num = dfOut, locusEval = snpEval %>% filter(!qtl))
# store all panels as a list
LDpanels <- list(mh = list(), snp = list(), randSNP = list())
for(s in panelSizes){
	dfOut <- num %>% mutate(num = (len * s) / sum(len), num = round(num)) # proportional to chr length
	dfOut$num[which.max(dfOut$num)] <- dfOut$num[which.max(dfOut$num)] + s - sum(dfOut$num)
	LDpanels$mh[[length(LDpanels$mh) + 1]] <- data.frame()
	LDpanels$snp[[length(LDpanels$snp) + 1]] <- data.frame()
	LDpanels$randSNP[[length(LDpanels$randSNP) + 1]] <- data.frame()
	#for each chromosome
	for(i in 1:nrow(dfOut)){
		if(dfOut$num[i] < 1) next
		# for each LD mh panel, choose MH
		LDpanels$mh[[length(LDpanels$mh)]] <- 
			rbind(LDpanels$mh[[length(LDpanels$mh)]], 
				  largestMH %>% filter(chr == dfOut$chr[i]) %>%
				  	arrange(selOrder) %>% slice(1:dfOut$num[i]))
		# for each LD SNP panel, choose SNPs
		LDpanels$snp[[length(LDpanels$snp)]] <- 
			rbind(LDpanels$snp[[length(LDpanels$snp)]], 
				  largestSNP %>% filter(chr == dfOut$chr[i]) %>%
				  	arrange(selOrder) %>% slice(1:dfOut$num[i]))
		# for each rand SNP panel, choose SNPs
		LDpanels$randSNP[[length(LDpanels$randSNP)]] <- 
			rbind(LDpanels$randSNP[[length(LDpanels$randSNP)]], 
				  HDpanel %>% filter(chr == dfOut$chr[i]) %>%
				  	slice_sample(n = dfOut$num[i]))
	}
	LDpanels$mh[[length(LDpanels$mh)]] <- LDpanels$mh[[length(LDpanels$mh)]] %>%
		arrange(chr, wStart)
	LDpanels$snp[[length(LDpanels$snp)]] <- LDpanels$snp[[length(LDpanels$snp)]] %>%
		arrange(chr, wStart)
	LDpanels$randSNP[[length(LDpanels$randSNP)]] <- LDpanels$randSNP[[length(LDpanels$randSNP)]] %>%
		arrange(chr, pos)
}

# read in chosen loci in from VCF
inputGenos <- vcf_readLoci(vcfPath = inputVCFpath, 
						   lineNumbers = sort(unique(c(qtl$lineNum, HDpanel$lineNum, 
						   							as.numeric(unlist(str_split(largestMH$lineNum, ","))), 
						   							largestSNP$lineNum))),
						   numLines = 20000)

# input to alphasimR
haplo_list <- list()
genMap <- list()
qtlPerChr <- c()
qtlPos <- list() # position of preselected QTLs
snpPos <- list() # position of preselected SNP chip loci, HD and LD panels
snpChipPositions <- data.frame() # correspondence between vcf chr/pos and alphasimr locus name
for(i in 1:nrow(num)){
	# record number of qtl for inputing into AlphaSimR
	qtlPerChr <- c(qtlPerChr, sum(qtl$chr == num$chr[i]))
	
	# create haplotype and map inputs
	tempBool <- inputGenos[[2]]$chr == num$chr[i]
	haplo_list[[i]] <- inputGenos[[1]][,tempBool]
	genMap[[i]] <- inputGenos[[2]]$pos[tempBool]
	genMap[[i]] <- genMap[[i]] / num$len[i] # normalize to 1M
	qtlPos[[i]] <- which(inputGenos[[2]]$pos[tempBool] %in% qtl$pos[qtl$chr == num$chr[i]])
	snpPos[[i]] <- which(!(inputGenos[[2]]$pos[tempBool] %in% qtl$pos[qtl$chr == num$chr[i]]))
	snpChipPositions <- rbind(snpChipPositions,
							  inputGenos[[2]][tempBool,][snpPos[[i]],] %>% 
		mutate(intPos = snpPos[[i]], chrN = i, name = paste0(chrN, "_", intPos)))
}

# now make lists of alphaSimR locus names for each panel for selecting later on
HDpanel <- HDpanel %>% left_join(snpChipPositions, by = c("chr", "pos"))
LDselect <- list(mh = list(), snp = list(), randSNP = list())
for(i in 1:length(LDpanels$mh)){
	# splitting microhaps into lines for each snp
	LDselect$mh[[i]] <- data.frame()
	for(ch in unique(LDpanels$mh[[i]]$chr)){
		LDselect$mh[[i]] <- rbind(LDselect$mh[[i]], 
					  data.frame(chr = ch, 
					  		   pos = LDpanels$mh[[i]] %>% filter(chr == ch) %>% 
					  		   	pull(pos) %>% str_split(",") %>% unlist() %>% 
					  		   	as.numeric()))
	}
	# make correspondence table between vcf and alphasimr chr/pos
	LDselect$mh[[i]] <- LDselect$mh[[i]] %>% left_join(snpChipPositions, by = c("chr", "pos"))
	LDselect$snp[[i]] <- LDpanels$snp[[i]] %>% select(chr, pos) %>%
		left_join(snpChipPositions, by = c("chr", "pos"))
	LDselect$randSNP[[i]] <- LDpanels$randSNP[[i]] %>% select(chr, pos) %>%
		left_join(snpChipPositions, by = c("chr", "pos"))
}


print(Sys.time())
print("end loading")

founderPop <- newMapPop(genMap=genMap, haplotypes=haplo_list)
SP <- SimParam$new(founderPop)
SP$setTrackPed(isTrackPed = TRUE) # have AlphaSimR maintain pedigree records

# forcing allocation of preselcted qtl and snpChip loci
# list with chr (in order) of integer positions of loci in genMap[[i]] that are invalid (in order)
# e.g. list(c(1,2,4), c(89,92,105)))
SP$invalidQtl <- snpPos
SP$invalidSnp <- qtlPos

SP$addTraitA(nQtlPerChr = qtlPerChr)
SP$setVarE(h2 = 0.3) # in the range of heritability for growth, meat yield, survival, etc
SP$setSexes("yes_sys") # at the time of breeding, all individuals will only be one sex
SP$addSnpChip(nSnpPerChr = sapply(snpPos, length)) # all non-QTL SNPs saved from simulation

pop <- list()
# pull founders from simulated pop while avoiding full and half sibs
pop[[1]] <- newPop(founderPop)


# calculate LD panel summary statistics in founders
hapMat <- pullSnpHaplo(pop[[1]])
sumStatLD <- tibble()
for(ldType in names(LDpanels)){
	for(i in 1:length(LDpanels[[ldType]])){
		pointLD <- LDpanels[[ldType]][[i]] # pointer to current panel to increase readability
		tempStats <- tibble(loc = 1:nrow(pointLD),
							aRich = NA,
							nSNP = NA,
							He = NA)
		for(j in 1:nrow(pointLD)){
			# position of SNPs in locus
			tempPos <- as.numeric(str_split(pointLD$pos[j], ",")[[1]])
			# names of SNPs in alphaSimR
			tempName <- LDselect[[ldType]][[i]] %>% filter(pos %in% tempPos) %>% pull(name)
			tempHapMat <- hapMat[,tempName,drop = FALSE]
			# concat to form microhap alleles
			a <- tempHapMat[,1]
			if(ncol(tempHapMat) > 1) for(k in 2:ncol(tempHapMat)) a <- paste0(a, tempHapMat[,k])
			tempStats$aRich[j] <- n_distinct(a)
			tempStats$nSNP[j] <- length(tempName)
			tempStats$He <- 1 - sum((table(a) / length(a))^2)
		}
		
		sumStatLD <- sumStatLD %>% bind_rows(
			tibble(
				type = ldType,
				panel = i,
				nLoci = nrow(pointLD),
				aRich_mu = mean(tempStats$aRich),
				aRich_sd = sd(tempStats$aRich),
				nSNP_mu = mean(tempStats$nSNP),
				nSNP_min = min(tempStats$nSNP),
				nSNP_max = max(tempStats$nSNP),
				He_mu = mean(tempStats$He)
			)
		)
	}
}
save(sumStatLD,  file = paste0("rda/ldPanelSumStats_", prefix, "_", iterationNumber, ".rda"))

# calculate allele freqs in base generation
founderAlleleFreqs <- apply(pullSnpGeno(pop[[1]]), 2, function(x) sum(x) / (2 * length(x)))

# write out parameter file for renumf90
# note that blupf90 needs to be run from the temporary directory as 
# the paths in the parameter file are relative
# using full paths can be too many characters for blupf90 to handle
# and you get an error from having incorrect file paths (b/c they are truncated)
cat("DATAFILE
f90dat.txt
TRAITS
3
FIELDS_PASSED TO OUTPUT

WEIGHT(S)

RESIDUAL_VARIANCE
2.0
EFFECT          # first fixed effect, overall mean
2 cross numer
EFFECT           # first random effect (animal)
1 cross alpha
RANDOM           ## additive effect without pedigree
animal
SNP_FILE         ## SNP marker file
f90snp.txt
(CO)VARIANCES    ## its variance component
1.0
OPTION use_yams
OPTION AlphaBeta 0.99 0.01
OPTION tunedG 0
OPTION whichG 1 # vanRaden 2008
OPTION whichfreq 0 # use freqs from file
OPTION FreqFile baseFreqs.txt # file with frequencies (in same order as genotypes)
OPTION whichfreqScale 0 # use freqs from file
OPTION minfreq 0.0 # turning off all filters and checks
OPTION monomorphic 0
OPTION verify_parentage 0
OPTION no_quality_control
OPTION num_threads_pregs 2 # number of threads
OPTION threshold_duplicate_samples 100 # effectively ignore
OPTION high_threshold_diagonal_g 2 # effectively ignore
OPTION low_threshold_diagonal_g 0.5 # effectively ignore
", file=paste0(localTempDir, "/", "temp", iterationNumber, "/", "renum.txt"), sep = "")

# initial spawning
pop[[2]] <- randCross(pop[[1]], nCrosses = nBrood/2, nProgeny = nOffspringPerCross, balance = TRUE)

if(!dir.exists(paste0(localTempDir, "/", "temp", iterationNumber))) dir.create(paste0(localTempDir, "/", "temp", iterationNumber))
trainPhenos <- data.frame()
gebvRes <- data.frame()
imputeRes <- data.frame()
for(gen in 1:nGenerations){
	print(Sys.time())
	print(paste("begin gen: ", gen))
	# phenotype training pop (sibs) of current generation adn add to phenotype data set
	trainPhenos <- rbind(trainPhenos, sibTestEqual(fam = pop[[gen + 1]], propTest = 0.6)) # phenotype 30, select from 20
	
	for(i in 1:length(LDselect$mh)){
		print(Sys.time())
		print(paste("begin panel: ", i))
		# loop through mh and snp
		for(locType in names(LDselect)){
			print(Sys.time())
			print(locType)
			# make inputs
			ped <- SP$pedigree[,1:2] # full pedigree
			# only get inds starting with founder pop
			# this chunk written to also work for cases where some "pre-simulation" breeding was
			# performed - usually to generate LD between chromosomes
			allInds <- unlist(lapply(pop, function(x) x@id))
			ped <- ped[allInds,]
			# pretend you don't know parents of founders
			ped[pop[[1]]@id,1:2] <- 0
			write.table(ped, file = paste0(localTempDir, "/", "temp", iterationNumber, "/ped.txt"),
						sep = " ", quote = FALSE, col.names = FALSE, row.names = TRUE)
			
			# get all genotypes for HD and current LD panel
			g <- all_pullSnpGenos(pop, loci = unique(c(HDpanel$name, LDselect[[locType]][[i]]$name)))
			trueGenos <- g[as.character(pop[[gen + 1]]@id),]
			# get id's for inds with high-density genotypes (parents)
			highDensInds <- unique(c(ped[,1], ped[,2]))
			highDensInds <- highDensInds[highDensInds > 0]
			# low-density offspring missing genotypes for HD panel 
			g[!rownames(g) %in% highDensInds,!colnames(g) %in% LDselect[[locType]][[i]]$name] <- 9
			
			# impute
			imputeDose <- data.frame(id = rownames(g))
			# for each chromosome
			for(j in 1:nrow(num)){
				tempCols <- colnames(g)[grepl(paste0("^", j, "_"), colnames(g))] # loci in chromosome j
				write.table(g[,tempCols],
							file = paste0(localTempDir, "/", "temp", iterationNumber, "/apGeno.txt"), 
							sep = " ", quote = FALSE, col.names = FALSE, 
							row.names = TRUE)
				
				# run AlphaImpute2
				# 1 output file prefix
				# 2 genotype input
				# 3 pedigree input
				# 4 random seed
				# 5 max thread for imputation
				system2("bash", args = c("runAlphaImpute2.sh",
										 paste0(localTempDir, "/", "temp", iterationNumber, "/imputeOut"),
										 paste0(localTempDir, "/", "temp", iterationNumber, "/apGeno.txt"),
										 paste0(localTempDir, "/", "temp", iterationNumber, "/ped.txt"),
										 "7",
										 "2")) # using two threads b/c ceres has hyperthreading on all cores
				
				# load results
				tempImputeDose <- read.table(paste0(localTempDir, "/", "temp", iterationNumber, "/imputeOut.genotypes"))
				colnames(tempImputeDose) <- c("id", tempCols)
				imputeDose <- imputeDose %>% 
					left_join(tempImputeDose %>% mutate(id = as.character(id)), by = "id")
			}
			
			# calc imputation accuracy and save
			# locus wise and individual wise (Calus et al 2014 doi:10.1017/S1751731114001803)
			# get only loci/individuals in the _current_ generation that were imputed
			imputeCalls <- imputeDose[imputeDose$id %in% rownames(trueGenos), !colnames(imputeDose) %in% LDselect[[locType]][[i]]$name]
			rownames(imputeCalls) <- imputeCalls$id
			imputeCalls <- as.matrix(imputeCalls[,-1])
			trueGenos <- trueGenos[rownames(imputeCalls), colnames(imputeCalls)]
			
			# filter out nonvariable loci
			# loci can become fixed during the simulation
			# and occasionally imputation will return the same genotype for all individuals
			# correlation is undefined (0/0) when either variable is constant
			temp <- (apply(trueGenos, 2, n_distinct) > 1) & (apply(imputeCalls, 2, n_distinct) > 1)
			trueGenos <- trueGenos[,temp]
			imputeCalls <- imputeCalls[,temp]
			rm(temp)
			# mean across loci
			perLocusAcc <- mean(sapply(1:ncol(trueGenos), function(x) cor(trueGenos[,x], imputeCalls[,x])))
			# now center and scale per locus, then calculate per individual accuracy (correlation) Calus et al 2014
			trueGenos <- scale(trueGenos, center = TRUE, scale = TRUE)
			imputeCalls <- scale(imputeCalls, center = TRUE, scale = TRUE)
			perIndAcc <- mean(sapply(1:nrow(trueGenos), function(x) cor(trueGenos[x,], imputeCalls[x,])))
			# save for results
			imputeRes <- imputeRes %>% 
				rbind(data.frame(genNum = gen,
								 panelNum = i,
								 locusType = locType,
								 numLoci = panelSizes[i],
								 locusImputeAcc = perLocusAcc,
								 indImputeAcc = perIndAcc,
								 numLociVar = ncol(trueGenos)))
			rm(imputeCalls) # save some memory
			rm(trueGenos)
			
			# calculate GEBVs
			rownames(imputeDose) <- imputeDose$id
			imputeDose <- as.matrix(imputeDose[,-1])
			# only HD panel loci
			g <- imputeDose[rownames(g), colnames(imputeDose) %in% HDpanel$name]
			rm(imputeDose)
			
			sol <- calcGEBVs_blupf90(g = g, founderAlleleFreqs = founderAlleleFreqs, 
									 localTempDir = localTempDir, iterationNumber = iterationNumber, 
									 trainPhenos = trainPhenos, SP_pedigree = SP$pedigree, 
									 curGenIDs = pop[[gen + 1]]@id)
			# NOTE: only using _current_ generation to calculate accuracy of gebvs
			comp <- data.frame(id = pop[[gen + 1]]@id, gv = as.vector(gv(pop[[gen + 1]]))) %>% 
				left_join(data.frame(id = as.character(sol$levelNew), gebv = sol$V4), by = "id") %>%
				left_join(trainPhenos %>% select(id, Trait_1) %>% rename(pheno = Trait_1), by = "id")
			
			# calc accuracy of prediction and save
			gebvRes <- gebvRes %>% 
				rbind(data.frame(genNum = gen,
								panelNum = i,
								locusType = locType,
								numLoci = panelSizes[i], 
								acc = cor(comp$gv[is.na(comp$pheno)], comp$gebv[is.na(comp$pheno)])))
		}
	}
	
	print(Sys.time())
	print("calculating GEBVs wtih full panel")
	g <- all_pullSnpGenos(pop, loci = HDpanel$name)
	sol <- calcGEBVs_blupf90(g = g, 
							 founderAlleleFreqs = founderAlleleFreqs, 
							 localTempDir = localTempDir, iterationNumber = iterationNumber, 
							 trainPhenos = trainPhenos, SP_pedigree = SP$pedigree, 
							 curGenIDs = pop[[gen + 1]]@id)
	# saving accuracy to serve as "control"
	# NOTE: only using _current_ generation to calculate accuracy of gebvs
	comp <- data.frame(id = pop[[gen + 1]]@id, gv = as.vector(gv(pop[[gen + 1]]))) %>% 
		left_join(data.frame(id = as.character(sol$levelNew), gebv = sol$V4), by = "id") %>%
		left_join(trainPhenos %>% select(id, Trait_1) %>% rename(pheno = Trait_1), by = "id")
	
	# calc accuracy of prediction and save
	gebvRes <- gebvRes %>% 
		rbind(data.frame(genNum = gen,
						 panelNum = length(panelSizes) + 1,
						 locusType = "HD",
						 numLoci = nrow(HDpanel),
						 acc = cor(comp$gv[is.na(comp$pheno)], comp$gebv[is.na(comp$pheno)])))
		
	# make next generation based on GEBVs calculated with full HD panel for all individuals
	if(gen < nGenerations){
		print(Sys.time())
		print("begin ocs")
		# OCS with lagrangian
		selCands <- comp %>% filter(is.na(pheno)) %>% pull(id)
		ocsData <- data.frame(Indiv = pop[[gen + 1]]@id, Sex = if_else(pop[[gen + 1]]@sex == "M", "male", "female")) %>%
			left_join(data.frame(Indiv = as.character(sol$levelNew), gebv = sol$V4), by = "Indiv") %>%
			filter(Indiv %in% selCands)
		# create G for OCS routine
		Amat <- createG(g = g[ocsData$Indiv,],
						af = founderAlleleFreqs[colnames(g)]) # G with first method of VanRaden (2008)
		matingPlan <- runOCS(ocsData = ocsData, Gmat = Amat[ocsData$Indiv,ocsData$Indiv], 
							 N = nBrood / 2, Ne = 50)
		rm(Amat) # save memory
		print(Sys.time())
		print("end ocs")
		# create next generation
		pop[[gen + 2]] <- makeCross(pop[[gen + 1]], 
									crossPlan = as.matrix(matingPlan[,2:1]), # female in first col, male in second
									nProgeny = nOffspringPerCross)
	}
}
# initial testing, save everything
# save.image(paste0("multGen_", iterationNumber, ".rda"))
if(!dir.exists("rda")) dir.create("rda")
# for low memory use, only save needed outputs
save(imputeRes, gebvRes, file = paste0("rda/multGen_empir_MH_small_", prefix, "_", iterationNumber, ".rda"))
