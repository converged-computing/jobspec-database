# PACKAGES ####
library(sp)
library(raster)
library(data.table)
library(ctmm)
library(GISTools)

# ANALYSIS ####
# Read job number from command line
args = commandArgs(trailingOnly=TRUE)
ind_file <- args[1]
print(ind_file)

# Load data and create telemetry object
df <- fread(ind_file)
aid <- df$individual.local.identifier[1]

l <- as.telemetry(df)

# Load habitat raster (trees) and crop it to save some time and RAM
r1 <- raster("data/treecover2010_cerrado.tif")
e <- extent(min(l$longitude) - 0.2, max(l$longitude) + 0.2, min(l$latitude) - 0.2, max(l$latitude) + 0.2)
r2 <- crop(r1, e)

# Record start time to monitor how long replicates take to compute
sTime <- Sys.time()
print(sTime)

# Create training and test sets ###
tl <- length(l$timestamp)
fh <- tl/2
sh <- tl/2+1
train <- l[1:fh,]
test <- l[sh:tl,]

# ANALYSIS ####
# Fit the movement model to the test data
svf <- variogram(train)
guess <- ctmm.guess(train, variogram=svf, interactive=FALSE)
fit <- ctmm.select(train, guess, trace=2)
summary(fit) 
print("Fitted movement model")
  
# Calculate the habitat-naive AKDE UD ###
ud <- akde(train, fit, weights=TRUE)
ess <- summary(ud)$DOF["area"]
ud10_area <- summary(ud, level.UD=0.10, units=FALSE)$CI[1,2]
ud20_area <- summary(ud, level.UD=0.20, units=FALSE)$CI[1,2]
ud30_area <- summary(ud, level.UD=0.30, units=FALSE)$CI[1,2]
ud40_area <- summary(ud, level.UD=0.40, units=FALSE)$CI[1,2]
ud50_area <- summary(ud, level.UD=0.50, units=FALSE)$CI[1,2]
ud60_area <- summary(ud, level.UD=0.60, units=FALSE)$CI[1,2]
ud70_area <- summary(ud, level.UD=0.70, units=FALSE)$CI[1,2]
ud80_area <- summary(ud, level.UD=0.80, units=FALSE)$CI[1,2]
ud90_area <- summary(ud, level.UD=0.90, units=FALSE)$CI[1,2]
ud99_area <- summary(ud, level.UD=0.99, units=FALSE)$CI[1,2]
print("UD created")

# Fit the RSF ###
rsf <- ctmm:::rsf.fit(train, UD=ud, R=list(trees=r2), debias=TRUE, error=0.01)
summary(rsf)
print("Fitted RSF")

# Calculate the RSF-informed AKDE
ud_rsf <- akde(train, rsf, R=list(trees=r2))
ud10_rsf_area <- summary(ud_rsf, level.UD=0.10, units=FALSE)$CI[1,2]
ud20_rsf_area <- summary(ud_rsf, level.UD=0.20, units=FALSE)$CI[1,2]
ud30_rsf_area <- summary(ud_rsf, level.UD=0.30, units=FALSE)$CI[1,2]
ud40_rsf_area <- summary(ud_rsf, level.UD=0.40, units=FALSE)$CI[1,2]
ud50_rsf_area <- summary(ud_rsf, level.UD=0.50, units=FALSE)$CI[1,2]
ud60_rsf_area <- summary(ud_rsf, level.UD=0.60, units=FALSE)$CI[1,2]
ud70_rsf_area <- summary(ud_rsf, level.UD=0.70, units=FALSE)$CI[1,2]
ud80_rsf_area <- summary(ud_rsf, level.UD=0.80, units=FALSE)$CI[1,2]
ud90_rsf_area <- summary(ud_rsf, level.UD=0.90, units=FALSE)$CI[1,2]
ud99_rsf_area <- summary(ud_rsf, level.UD=0.99, units=FALSE)$CI[1,2]

# Cross-validate the UDs ###

# Export the UDs as sp objects
ud10 <- SpatialPolygonsDataFrame.UD(ud, level.UD = 0.10)
ud20 <- SpatialPolygonsDataFrame.UD(ud, level.UD = 0.20)
ud30 <- SpatialPolygonsDataFrame.UD(ud, level.UD = 0.30)
ud40 <- SpatialPolygonsDataFrame.UD(ud, level.UD = 0.40)
ud50 <- SpatialPolygonsDataFrame.UD(ud, level.UD = 0.50)
ud60 <- SpatialPolygonsDataFrame.UD(ud, level.UD = 0.60)
ud70 <- SpatialPolygonsDataFrame.UD(ud, level.UD = 0.70)
ud80 <- SpatialPolygonsDataFrame.UD(ud, level.UD = 0.80)
ud90 <- SpatialPolygonsDataFrame.UD(ud, level.UD = 0.90)
ud99 <- SpatialPolygonsDataFrame.UD(ud, level.UD = 0.99)

ud10_rsf <- SpatialPolygonsDataFrame.UD(ud_rsf, level.UD = 0.10)
ud20_rsf <- SpatialPolygonsDataFrame.UD(ud_rsf, level.UD = 0.20)
ud30_rsf <- SpatialPolygonsDataFrame.UD(ud_rsf, level.UD = 0.30)
ud40_rsf <- SpatialPolygonsDataFrame.UD(ud_rsf, level.UD = 0.40)
ud50_rsf <- SpatialPolygonsDataFrame.UD(ud_rsf, level.UD = 0.50)
ud60_rsf <- SpatialPolygonsDataFrame.UD(ud_rsf, level.UD = 0.60)
ud70_rsf <- SpatialPolygonsDataFrame.UD(ud_rsf, level.UD = 0.70)
ud80_rsf <- SpatialPolygonsDataFrame.UD(ud_rsf, level.UD = 0.80)
ud90_rsf <- SpatialPolygonsDataFrame.UD(ud_rsf, level.UD = 0.90)
ud99_rsf <- SpatialPolygonsDataFrame.UD(ud_rsf, level.UD = 0.99)

# Export the test data as an sp object
test_sp <- SpatialPoints.telemetry(test)

# Determine which points are inside the 95% OD
pct10 <- round(unname(GISTools:::poly.counts(test_sp, ud10)[2])/nrow(test)*100, 2)
pct20 <- round(unname(GISTools:::poly.counts(test_sp, ud20)[2])/nrow(test)*100, 2)
pct30 <- round(unname(GISTools:::poly.counts(test_sp, ud30)[2])/nrow(test)*100, 2)
pct40 <- round(unname(GISTools:::poly.counts(test_sp, ud40)[2])/nrow(test)*100, 2)
pct50 <- round(unname(GISTools:::poly.counts(test_sp, ud50)[2])/nrow(test)*100, 2)
pct60 <- round(unname(GISTools:::poly.counts(test_sp, ud60)[2])/nrow(test)*100, 2)
pct70 <- round(unname(GISTools:::poly.counts(test_sp, ud70)[2])/nrow(test)*100, 2)
pct80 <- round(unname(GISTools:::poly.counts(test_sp, ud80)[2])/nrow(test)*100, 2)
pct90 <- round(unname(GISTools:::poly.counts(test_sp, ud90)[2])/nrow(test)*100, 2)
pct99 <- round(unname(GISTools:::poly.counts(test_sp, ud99)[2])/nrow(test)*100, 2)

pct10_rsf <- round(unname(GISTools:::poly.counts(test_sp, ud10_rsf)[2])/nrow(test)*100, 2)
pct20_rsf <- round(unname(GISTools:::poly.counts(test_sp, ud20_rsf)[2])/nrow(test)*100, 2)
pct30_rsf <- round(unname(GISTools:::poly.counts(test_sp, ud30_rsf)[2])/nrow(test)*100, 2)
pct40_rsf <- round(unname(GISTools:::poly.counts(test_sp, ud40_rsf)[2])/nrow(test)*100, 2)
pct50_rsf <- round(unname(GISTools:::poly.counts(test_sp, ud50_rsf)[2])/nrow(test)*100, 2)
pct60_rsf <- round(unname(GISTools:::poly.counts(test_sp, ud60_rsf)[2])/nrow(test)*100, 2)
pct70_rsf <- round(unname(GISTools:::poly.counts(test_sp, ud70_rsf)[2])/nrow(test)*100, 2)
pct80_rsf <- round(unname(GISTools:::poly.counts(test_sp, ud80_rsf)[2])/nrow(test)*100, 2)
pct90_rsf <- round(unname(GISTools:::poly.counts(test_sp, ud90_rsf)[2])/nrow(test)*100, 2)
pct99_rsf <- round(unname(GISTools:::poly.counts(test_sp, ud99_rsf)[2])/nrow(test)*100, 2)

# Cross-validated log-likelihood
ud_r <- raster(ud, DF = "PDF")
ud_rsf_r <- raster(ud_rsf, DF = "PDF")

cv_ll_u <- raster::extract(ud_r, test_sp)
cv_ll_ud <- sum(cv_ll_u)

cv_ll_r <- raster::extract(ud_rsf_r, test_sp)
cv_ll_rsf <- sum(cv_ll_r)

eTime <- Sys.time()
  
# Create vector of results
results <- data.frame(aid, ind_file, ess, cv_ll_ud, cv_ll_rsf, 
		      ud10_area, ud20_area, ud30_area, ud40_area, ud50_area, ud60_area, ud70_area, ud80_area, ud90_area, ud99_area, 
		      ud10_rsf_area, ud20_rsf_area, ud30_rsf_area, ud40_rsf_area, ud50_rsf_area, ud60_rsf_area, ud70_rsf_area, ud80_rsf_area, ud90_rsf_area, ud99_rsf_area,
		      pct10, pct20, pct30, pct40, pct50, pct60, pct70, pct80, pct90, pct99,
		      pct10_rsf, pct20_rsf, pct30_rsf, pct40_rsf, pct50_rsf, pct60_rsf, pct70_rsf, pct80_rsf, pct90_rsf, pct99_rsf)
  
# Store results in data.frame
write.table(results, 'results/cv_summary_ce_tapir.csv', append=TRUE, row.names=FALSE, col.names=FALSE, sep=',') 

# Save important objects to an Rda file
save(df,svf,fit,ud,rsf,ud_rsf,file=paste0("results/",aid,".Rda"))
  
# Print indicators of progress
print(aid)
print(eTime)
print(eTime - sTime)
