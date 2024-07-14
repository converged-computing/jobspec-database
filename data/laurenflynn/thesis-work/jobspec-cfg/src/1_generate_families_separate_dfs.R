set.seed(777)

# Load Libraries ----

setwd("../PedUtils/R")
files.sources = list.files()
sapply(files.sources, source) #loads in all the PedUtils functions
setwd("~/git_thesis_work")
source("src/family_filters.R")
library(plyr) #need to load plyr before dplyr
library(truncnorm)
library(PanelPRO)
library(tidyverse)
library(stringr)


mlh1Freq <- 0.05
numberFamilies <- 10000


# Set MLH1 Frequency to user provided val ----
load("~/PanelPRO/data/PanelPRODatabase.rda")
PanelPRODatabase$AlleleFrequency[["MLH1_anyPV",1]] <- mlh1Freq
PanelPRODatabase$AlleleFrequency[["MLH1_anyPV",2]] <- mlh1Freq
PanelPRODatabase$AlleleFrequency[["MLH1_anyPV",3]] <- mlh1Freq
save(PanelPRODatabase, file = "~/PanelPRO/data/PanelPRODatabase.rda")


# Generate Families ----
families = list()
probandIDS = c()
probandMLH1Status = c()
probandAffectionStatus = c()


for(i in 1:numberFamilies){
  # Cancers
  cancers = "Colorectal"
  # Genes
  genes = "MLH1"
  #family members
  # Paternal aunts, paternal uncles
  nSibsPatern =floor(rtruncnorm(n=2, mean=3, 3))
  # Maternal aunts, maternal uncles
  nSibsMatern = floor(rtruncnorm(n=2, mean=3, 3))
  # Sisters and brothers
  nSibs = floor(rtruncnorm(n=2, mean=3, 3))
  # We make the assumption that the number of sons and daughters for the
  # proband and all siblings, is the same. Nieces and nephews of the proband
  # are not sampled separately.
  nGrandchild = floor(rtruncnorm(n=2, mean=6, 2))
  nChild = floor(rtruncnorm(n=2, mean=3, 2))
  
  # Simulate family using `PedUtils` code
  fam = sim.runSimFam(nSibsPatern, nSibsMatern, nSibs, nChild,
                      PanelPRODatabase, genes, cancers,
                      includeGeno = TRUE, includeBiomarkers = TRUE, censoring = FALSE)
  famDF = as.data.frame(fam)
  for(j in 1:nrow(famDF)){
    if(famDF[j,]$isAffAny == 0){
      famDF[j,]$AgeAny = NA    
    }
    if(famDF[j,]$isAffCOL == 0){
      famDF[j,]$AgeCOL = NA    
    }
  }
  proband = famDF %>% filter(isProband==1)
  probandIDS = c(probandIDS, proband$ID)
  probandMLH1Status = c(probandMLH1Status, proband$MLH1)
  probandAffectionStatus = c(probandAffectionStatus, proband$isAffAny)
  families[[i]] = famDF
}


mlh1StatusFamilies <- families
for(i in 1:numberFamilies){
  families[[i]] = removeProbandStatus(families[[i]])
}


firstDegreeAffected <- list()
firstDegree <- list()
for(i in 1:numberFamilies){
  firstDegree[[i]] = firstDegreeFamilyMembers(families[[i]])
  affectedFirstDegreeRels = firstDegree[[i]] %>% filter(isProband == 0) %>% filter(isAffAny ==1)
  if(nrow(affectedFirstDegreeRels)  > 0){
    firstDegreeAffected[[i]] = 1
  }
  else{
    firstDegreeAffected[[i]] = 0
  }
}

maskedFamilies <- list()
for(i in 1:numberFamilies){
  maskedFamilies[[i]] = maskedInfoAmbry(families[[i]])
}

print("Full Families from PedUtils with MLH1 Status for Probands")
describeFamilies(mlh1StatusFamilies)
print("*******")
print("Full Families from PedUtils with Unaffected Information Masked")
describeFamilies(maskedFamilies)
print("*******")
print("First Degree Families")
describeFamilies(firstDegree)



numAffectedRelatives <- c()
numMLH1Relatives <- c()
numMLH1UnaffectedRelatives <- c()


for(i in 1:length(families)){
  affected <- families[[i]] %>% filter(isAffAny == 1)
  numAffectedRelatives[[i]] = nrow(affected)
}

for(i in 1:length(families)){
  mlh1Rels <- families[[i]] %>% filter(MLH1 ==1)
  numMLH1Relatives [[i]] = nrow(mlh1Rels)
}

for(i in 1:length(families)){
  unaffectedMLH1 <- families[[i]] %>% filter(isAffAny==0) %>% filter(MLH1==1)
  numMLH1UnaffectedRelatives[[i]] = nrow(unaffectedMLH1)
}



#automatically adjusts to the number of families specified
# file_name = str_interp("RObjects/familyinfo${numberFamilies}Families_no_for_each.Rdata")
# save(families, maskedFamilies, mlh1StatusFamilies, firstDegree, probandMLH1Status, probandIDS, numAffectedRelatives,numMLH1Relatives,numMLH1UnaffectedRelatives,numberFamilies, file = file_name)

# save.image(file = "no_censoring.RData", version = NULL, ascii = FALSE,
#            compress = !ascii, safe = TRUE)
save(families, maskedFamilies,firstDegree,probandMLH1Status,probandIDS,numAffectedRelatives,numMLH1Relatives, numMLH1UnaffectedRelatives,probandAffectionStatus,firstDegreeAffected, file="no_censoring.Rdata")
for(i in 1:length(families)){
  file_name = str_interp("RObjects/family_dfs/individualDataFrames/no_censoring/family_info_${numberFamilies}_families_no_for_each_${i}.Rdata")
  family = families[[i]]
  maskedFamily = maskedFamilies[[i]]
  firstDeg = firstDegree[[i]]
  proMLH1 = probandMLH1Status[[i]]
  proID = as.integer(probandIDS[[i]])
  numAffRels = numAffectedRelatives[[i]]
  numMLH1Rels = numMLH1Relatives[[i]]
  numMLH1UnaffRels = numMLH1UnaffectedRelatives[[i]]
  proAffStatus = probandAffectionStatus[[i]]
  firstDegAff = firstDegreeAffected[[i]]
  save(family, maskedFamily, firstDeg, proMLH1, proID, numAffRels,numMLH1Rels,numMLH1UnaffRels,numberFamilies, mlh1Freq, proAffStatus, firstDegAff, file = file_name)
  
}


