#### Set Up ####
start_time <- Sys.time()
set.seed(777)
# load libraries
library(plyr) #need to load plyr before dplyr
library(dplyr)
library(PanelPRO)
library(tidyverse)

sizeOfArray = 400
numberFamilies <- 10000
mlh1Freq <- 0.05

iterations <- numberFamilies / sizeOfArray
#commandArgs picks up the variables you pass from the command line
args <- commandArgs(trailingOnly = TRUE)
args <- as.integer(args)
print(args)
lowerbound = args * iterations - (iterations -1) #assumes we are breaking into job array of 50
upperbound = args * iterations
print(lowerbound)
print(upperbound)


# Set MLH1 Frequency to provided val ----
load("~/PanelPRO/data/PanelPRODatabase.rda")
PanelPRODatabase$AlleleFrequency[["MLH1_anyPV",1]] <- mlh1Freq
PanelPRODatabase$AlleleFrequency[["MLH1_anyPV",2]] <- mlh1Freq
PanelPRODatabase$AlleleFrequency[["MLH1_anyPV",3]] <- mlh1Freq
updatedPanelPRODatabase <- PanelPRODatabase
print(updatedPanelPRODatabase$AlleleFrequency["MLH1_anyPV",])

families <- list()
firstDegree <- list()
maskedFamilies <- list()
probandIDS <- list()
numAffectedRelatives <- list()
numMLH1Relatives <- list()
numMLH1UnaffectedRelatives <- list()
probandAffectionStatus <- list()
firstDegreeAffected <- list()
probandMLH1Status <- list()


for(i in lowerbound:upperbound){
  load(str_interp("RObjects/family_dfs/individualDataFrames/no_censoring/family_info_${numberFamilies}_families_no_for_each_${i}.Rdata"))
  families[[i]] = family
  firstDegree[[i]] = firstDeg
  maskedFamilies[[i]] = maskedFamily
  probandIDS[[i]] = family[family$isProband ==1,]$ID
  numAffectedRelatives[[i]] = numAffRels
  numMLH1Relatives[[i]] = numMLH1Rels
  numMLH1UnaffectedRelatives[[i]] = numMLH1UnaffRels
  probandAffectionStatus[[i]] = proAffStatus
  firstDegreeAffected[[i]] = firstDegAff
  probandMLH1Status[[i]] = proMLH1
}




families <- families[lowerbound:upperbound]
firstDegree <- firstDegree[lowerbound:upperbound]
maskedFamilies <- maskedFamilies[lowerbound:upperbound]
probandIDS <- probandIDS[lowerbound:upperbound]
numAffectedRelatives <- numAffectedRelatives[lowerbound:upperbound]
numMLH1Relatives <- numMLH1Relatives[lowerbound:upperbound]
numMLH1UnaffectedRelatives <-  numMLH1UnaffectedRelatives[lowerbound:upperbound]
probandAffectionStatus <- probandAffectionStatus[lowerbound:upperbound]
firstDegreeAffected <- firstDegreeAffected[lowerbound:upperbound]
probandMLH1Status <- probandMLH1Status[lowerbound:upperbound]

print("Read in all data.")

#### Full Families ####

fullCarrierRisk <- list()

dataframe_index <- args
dataframe_index <- as.integer(dataframe_index)


#get proband carrier risk for full families
for(i in 1:iterations){
  families[[i]]$MLH1 <- NA
  out <- PanelPRO::PanelPRO(families[[i]], genes="MLH1", cancers="Colorectal", database = updatedPanelPRODatabase)
  id <- as.character(probandIDS[i])
  fullCarrierRisk[[i]] = out$posterior.prob[[id]]$estimate[2]
}

print("Ran PanelPRO on Full Families")


#### Full Families with Masked Information for Unaffected Individuals ####

# get outputs of PanelPRO for families with masked info
maskedCarrierRisk <- list()
#get proband carrier risk for full families with masked information about unaffected relatives
for(i in 1:iterations){
  maskedFamilies[[i]]$MLH1 <- NA
  out <- PanelPRO::PanelPRO(maskedFamilies[[i]], genes="MLH1", cancers="Colorectal", database = updatedPanelPRODatabase)
  id <- as.character(probandIDS[i])
  maskedCarrierRisk[[i]] = out$posterior.prob[[id]]$estimate[2]
}

print("Ran PanelPRO with unaffected family members info masked")


#### First Degree Families ####
firstDegreeCarrierRisk <- list()

#get proband carrier risk for first degree families nothing masked
for(i in 1:iterations) {
  firstDegree[[i]]$MLH1 <- NA
  out <- PanelPRO::PanelPRO(firstDegree[[i]], genes="MLH1", cancers="Colorectal", database = updatedPanelPRODatabase)
  id <- as.character(probandIDS[i])
  firstDegreeCarrierRisk[[i]] = out$posterior.prob[[id]]$estimate[2]
  
}

print("Ran PanelPRO on first degree families")
print(firstDegreeCarrierRisk)

#### Combine Results Into a Summary Table ####
summaryTableMLH1 <- cbind(format(unlist(fullCarrierRisk), scientific=FALSE), unlist(maskedCarrierRisk), unlist(firstDegreeCarrierRisk), unlist(probandMLH1Status))
summaryTableMLH1 <- as.data.frame(summaryTableMLH1)
summaryTable <- cbind(lowerbound:upperbound, summaryTableMLH1)
names(summaryTable) <- c("famID","fullCarrierRisk", "carrierRiskUnaffectedInfoMasked", "firstDegreeCarrierRisk", "probandMLH1Status")
print(head(summaryTable))
summaryTable$fullCarrierRisk <- as.numeric(summaryTable$fullCarrierRisk)
summaryTable$carrierRiskUnaffectedInfoMasked <- as.numeric(summaryTable$carrierRiskUnaffectedInfoMasked)
summaryTable$firstDegreeCarrierRisk <- as.numeric(summaryTable$firstDegreeCarrierRisk)


summaryTable <- cbind(summaryTable, unlist(numAffectedRelatives))
summaryTable <- cbind(summaryTable, unlist(numMLH1Relatives))
summaryTable <- cbind(summaryTable, unlist(numMLH1UnaffectedRelatives))
summaryTable <- cbind(summaryTable, unlist(probandAffectionStatus))
summaryTable <- cbind(summaryTable, unlist(firstDegreeAffected))


names(summaryTable) <- c("famID","fullCarrierRisk", "carrierRiskUnaffectedInfoMasked", "firstDegreeCarrierRisk", "probandMLH1Status", "numAffectedRelatives", "numMLH1Relatives","numMLH1UnaffectedRelatives","probandAffectionStatus","firstDegreeAffectedFamilyMembersBinary")

print(nrow(summaryTable))
file_name = str_interp("RObjects/summary_tables/no_censoring/panelPROSummaryTable${numberFamilies}FamiliesNoCensoring${lowerbound}_${upperbound}.Rdata")

save(summaryTable, file = file_name)

end_time <- Sys.time()
print(str_interp("Time elapsed: ${end_time - start_time}"))
