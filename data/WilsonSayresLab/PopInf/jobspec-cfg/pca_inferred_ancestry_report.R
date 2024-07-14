#!/usr/bin/Rscript
# Authors: Angela Taravella and Anagha Deshpande
#usage: pca_inferred_ancestry_report.R evec_file eval_file reference_panel_info.txt unkpop_info.txt output_fileSetemName output_inferredPopReport_fileStemName


### LOAD PACKAGES ###
library("car")
library("plotrix")
library("viridis")


### DEFINE FUNCTIONS ###
# Function 1: circle_function #
# circle_function plots centroid of each population coordinate along with plotting 1, and 3 standard deviation circles around the cluster mean.
circle_function <- function(continent) {
  continent_of_choice <- continent
  continent_of_choice_data <- evec_file_full[evec_file_full$V3.x %in% continent_of_choice,]
  mean_x <- mean(continent_of_choice_data$V2.y)
  mean_y <- mean(continent_of_choice_data$V3.y)
  points(mean_x, mean_y, pch = 3, cex = 1)
  test <- sqrt( ((mean_x - (continent_of_choice_data$V2.y))^2) + ((mean_y - (continent_of_choice_data$V3.y))^2) )
  radius <- sqrt(sum((test-0)^2/(length(test)-1))) # this looks good
  draw.circle(mean_x, mean_y, radius)
  draw.circle(mean_x, mean_y, radius*3)
}

# Function 2: Get centroid coordinates and radii for each population cluster  #
circle_function_noplot <- function(continent) {
  continent_of_choice <- continent
  continent_of_choice_data <- evec_file_full[evec_file_full$V3.x %in% continent_of_choice,]
  mean_x <- mean(continent_of_choice_data$V2.y)
  mean_y <- mean(continent_of_choice_data$V3.y)
  test <- sqrt( ((mean_x - (continent_of_choice_data$V2.y))^2) + ((mean_y - (continent_of_choice_data$V3.y))^2) )
  radius <- sqrt(sum((test-0)^2/(length(test)-1))) # this looks good
  my_list <- list("radius" = radius, "mean_x" = mean_x, "mean_y" = mean_y)
  return(my_list)
}


### SET UP COMMAND INPUTS ###
args = commandArgs(trailingOnly=TRUE)
evec = args[1]
eval = args[2]
pop = args[3]
unkpop = args[4]
out = args[5]
out_inferred_report = args[6]


### SET UP OUTPUT PDF ###
pfdfilename <- paste(out,".pdf",sep="")
pdf(pfdfilename, 8,8) #save as pdf, with size 8x8



# THIS IS FOR TESTING PURPOSES ONLY AND SHOULD BE DELETED WHEN NO LONGER NEEDED!!!
#evec_file <- read.table("merge_all_chr_reference_panel_unknown_set_SNPs_no_missing_plink_LDprune_Fix2.evec", header = FALSE)
#eval_file <- read.table("merge_all_chr_reference_panel_unknown_set_SNPs_no_missing_plink_LDprune.eval", header = FALSE)
#known_population_file <- read.table("ThousandGenomesSamples_AdmxRm.txt", header = FALSE, sep = "\t")
#unkownpop_file <- read.table("GTExSamples.txt", header = FALSE, sep = "\t")


### READ IN DATA ###
evec_file <- read.table(evec, header = FALSE)
eval_file <- read.table(eval, header = FALSE)
known_population_file <- read.table(pop, header = FALSE, sep = "\t")
unkownpop_file <- read.table(unkpop, header = FALSE, sep = "\t")

print("All data has been imported into the program")


### SET UP INITIAL DATA FRAMES ###
known_sample_info_merge <- merge(known_population_file, evec_file,  by.x = "V1", by.y = "V1")
UnkownPop_data_WithInfo <- merge(unkownpop_file, evec_file, by.x = "V1", by.y = "V1")
evec_file_full <- rbind(known_sample_info_merge, UnkownPop_data_WithInfo)


### CALCULATE PERCENT VARIANCE EXPLAINED FOR EACH PC ###
eval_file$Perc_Var<- prop.table(eval_file$V1)*100

print("Plotting has started")

### PLOTTING ###
# Set up colors
n_colors <- length(levels(evec_file_full$V3.x))
#plot_colors <- rainbow(n_colors)
plot_colors <- viridis(n_colors)

# Generate the legend
plot(1, type="n", axes=FALSE, xlab="", ylab="", main = "Key")
legend('topleft', bty="n", legend = c(levels(evec_file_full$V3.x)), fill = c(plot_colors),
       cex = 1, title=expression(bold("Population Clusters")))
legend('top', bty="n", legend = c(levels(evec_file_full$V2.x)),
       cex = 1, pch = c(0, 1)[evec_file_full$V2.x], title=expression(bold("Genetic Sex")))



# Plot the graphs of the first 10 PCs
i <- 1
a <- 0
pc1 <- 0
pc2 <- 0
while (i<10)
{
  j <- 4 + a
  z <- 5 + a
  pc1 <- eval_file[i,2]
  pc1 <- round(pc1, digits = 2)
  axis_pc1 <- paste("PC",i," (",pc1,"%)",sep="")
  pc2 <- eval_file[j,2]
  pc2 <- round(pc2, digits = 2) # round so number doesnt have a lot of decimal places
  axis_pc2 <- paste("PC",i+1," (",pc2,"%)",sep="")
  plot(evec_file_full[,j],evec_file_full[,z], xlab = axis_pc1, ylab = axis_pc2,
       col=c(plot_colors)[evec_file_full$V3.x], pch =c(20, 15)[evec_file_full$V2.x], asp = 1)
  i <- i+2
  a <- a + 1

}

### SET UP OUTPUT INFERRED POPULATION REPORT ###
inferredpopreport <- paste(out_inferred_report,".txt",sep="")
#inferredpopreport <- paste("out_inferred_report_new",".txt",sep="")


### INFERRED ANCESTRY REPORT ###
print("Inferred population report has started")

# Generate the legend for the plot with cluster and midpoint info
plot(1, type="n", axes=FALSE, xlab="", ylab="", main = "Key")
legend('topleft', bty="n", legend = c(levels(evec_file_full$V3.x)), fill = c(plot_colors),
       cex = 1, title=expression(bold("Population Clusters")))
legend('top', bty="n", legend = c(levels(evec_file_full$V2.x)),
       cex = 1, pch = c(0, 1)[evec_file_full$V2.x], title=expression(bold("Genetic Sex")))
legend('bottomleft', bty="n", legend = c("Cluster Centroid",
                                   "1 and 3 Standard Deviations from Cluster Mean",
                                   "Pair-wise Cluster Mid Points"),
       cex = 1, pch = c(3, 1, 4))

# Plot
plot(evec_file_full[,4],evec_file_full[,5], xlab = "PC1", ylab = "PC2",
     col=c(plot_colors)[evec_file_full$V3.x], pch =c(20, 15)[evec_file_full$V2.x], asp = 1)

# Print all clusters onto the scatter plot
test_vector <- levels(known_sample_info_merge$V3.x)
for (i in 1:length(test_vector)) {
  circle_function(test_vector[i])
}


### Store relevant information in matricies ###
# Stores values of radius, mean X, and mean Y in a martrix called "mat_meanx_meany_rad"
n <- length(test_vector)
mat_meanx_meany_rad <- matrix(ncol=4, nrow=n)
test_vector <- levels(known_sample_info_merge$V3.x)
for (i in 1:length(test_vector)) {
  my_list <- circle_function_noplot(test_vector[i])
  mat_meanx_meany_rad[i,] <- c(test_vector[i],my_list$mean_x,my_list$mean_y,my_list$radius)
}

# Find the mid point (X and Y coordinates) for each pairwise comparison and store in a matrix called "mat_meanx_meany_compares"
n_com <- factorial(length(test_vector))/((factorial((length(test_vector)-2)))*factorial(2))
mat_meanx_meany_compares <- matrix(ncol=4, nrow=n_com)
combination_num <- 0

for (i in 1:length(test_vector)) {
  num <- i + 1
  for (num in i:length(test_vector)) {
    if (num > length(test_vector)) {} # we will skip this index because it is past the length of our vector and uninformative (no data)
    else {
      if (test_vector[i] == test_vector[num]) {} #skip this comparison. This is a self-self comparison
      else {
        comparison <- c(test_vector[i], test_vector[num])
        # so now we have to get our values we need to find midpoints
        my_list <- circle_function_noplot(test_vector[i])
        test_mean_x_first <- my_list$mean_x
        test_mean_y_first <- my_list$mean_y

        my_list <- circle_function_noplot(test_vector[num])
        test_mean_x_second <- my_list$mean_x
        test_mean_y_second <- my_list$mean_y

        x_mid <- (test_mean_x_first+test_mean_x_second)/2
        y_mid <- (test_mean_y_first+test_mean_y_second)/2

        points(x_mid,y_mid, col = "black", pch = 4)
        comparison_name <- paste(test_vector[i], test_vector[num], sep = "-")
        combination_num <- combination_num + 1
        mat_meanx_meany_compares[combination_num,] <- c(comparison_name,x_mid,y_mid,0)  # need a fourth column for merging with other matrix. I will add 0 to the 4th column
      }
    }
  }
}

### Generate inferred population report ###
# Below makes the empty data frame with the correct number of rows and columns 
# with informative information.
row_num <- length(UnkownPop_data_WithInfo[,1])
col_num <- length(test_vector) + 5
df <- data.frame(matrix(ncol = col_num, nrow = row_num))
vector_col_names <- c("Sample",
                      "Confident Inferred Pop: 1SD",
                      "Confident Inferred Pop: 2SD",
                      "Confident Inferred Pop: 3SD",
                      "Less Restricted Inferred Pop")
for (i in 1:length(test_vector)) {
  col_name_cluster_i <- paste("Distance to cluster", test_vector[i], sep = " ")
  vector_col_names <- c(vector_col_names, col_name_cluster_i)
}
colnames(df) <- vector_col_names


# This is looping through each individual in UnkownPop_data_WithInfo (so each 
# unknown sample) and see if they are within 3 standard deviations to a known
# population. If not, the program will see how close that individual is to
# a mid point between 2 populations and if that is closer than any population's 
# 3rd standard deviations, it will be called as the two populations of that 
# midpoint. If not, then that individual will be called to the population's 3rd
# SD it is closest to.

# Begin with a 0 iterator so that we can loop through all individuals
iterator <- 0
for (i in UnkownPop_data_WithInfo[,1]) {

  iterator <- iterator + 1
  sample_i <- paste(UnkownPop_data_WithInfo[iterator,1])
  pc_x <- UnkownPop_data_WithInfo[iterator,4]
  pc_y <-  UnkownPop_data_WithInfo[iterator,5]

  vector_to_add_df <- c(sample_i)

  inferred_pop_1SD <- c()
  inferred_pop_2SD <- c()
  inferred_pop_3SD <- c()

  test_all_for_ans_1SD <- FALSE
  test_all_for_ans_2SD <- FALSE
  test_all_for_ans_3SD <- FALSE

  for (cluster in 1:length(mat_meanx_meany_rad[,1])) {
    x <- as.numeric(mat_meanx_meany_rad[cluster,2])
    y <- as.numeric(mat_meanx_meany_rad[cluster,3])
    dist_i_to_cluster_centriod <- sqrt( ((x - pc_x)^2) + ((y - pc_y)^2) )

    sd_1 <- as.numeric(mat_meanx_meany_rad[cluster,4])
    if (dist_i_to_cluster_centriod <= sd_1) {
      test_all_for_ans_1SD <- TRUE
      test_all_for_ans_1SD <- FALSE
      inferred_pop_1SD <- paste(inferred_pop_1SD, mat_meanx_meany_rad[cluster,1], sep = "")
    }

    sd_2 <- as.numeric(mat_meanx_meany_rad[cluster,4]) * 2
    if (dist_i_to_cluster_centriod <= sd_2) {
      test_all_for_ans_2SD <- TRUE
      #inferred_pop_2SD <- c()
      if (is.null(inferred_pop_2SD)) {
        inferred_pop_2SD <- paste(inferred_pop_2SD, mat_meanx_meany_rad[cluster,1], sep = "")
      } else
      inferred_pop_2SD <- paste(inferred_pop_2SD, mat_meanx_meany_rad[cluster,1], sep = "-")
    }

    sd_3 <- as.numeric(mat_meanx_meany_rad[cluster,4]) * 3
    if (dist_i_to_cluster_centriod <= sd_3) {
      test_all_for_ans_3SD <- TRUE
      #inferred_pop_3SD <- c()
      if (is.null(inferred_pop_3SD)) {
        inferred_pop_3SD <- paste(inferred_pop_3SD, mat_meanx_meany_rad[cluster,1], sep = "")
      } else
      inferred_pop_3SD <- paste(inferred_pop_3SD, mat_meanx_meany_rad[cluster,1], sep = "-")
    }

  }

  if (!test_all_for_ans_1SD) {
    inferred_pop_1SD <- "-"
  }
  if (!test_all_for_ans_2SD) {
    inferred_pop_2SD <- "-"
  }
  if (!test_all_for_ans_3SD) {
    inferred_pop_3SD <- "-"
  }

  vector_to_add_df <- c(vector_to_add_df, inferred_pop_1SD, inferred_pop_2SD, inferred_pop_3SD)
# If the individual does not fall within 3SD of any cluster, the script will 
# compare the distance from 3SD of all clusters with distance from midpoints 
# of all pairwise clusters. 
  less_rest_inf_pop <- c()
  cat_points <- rbind(mat_meanx_meany_rad, mat_meanx_meany_compares)
  n_all <- length(cat_points[,1])
  com_all_pnts <- matrix(ncol=2, nrow=n_all)
  if (inferred_pop_3SD == "-"){
    for (test_all_pts in 1:length(cat_points[,1])) {
      if (cat_points[test_all_pts,4] != 0){
        # so below put the calculation for comparing i or test_all_points to 3SD not the centroid
        x <- as.numeric(cat_points[test_all_pts,2])
        y <- as.numeric(cat_points[test_all_pts,3])
        dist_i_to_centroid <- sqrt( ((x - pc_x)^2) + ((y - pc_y)^2) )
        dist_i_to_pnt <- abs(dist_i_to_centroid - (3*as.numeric(mat_meanx_meany_rad[test_all_pts,4])))
      } else {
        x <- as.numeric(cat_points[test_all_pts,2])
        y <- as.numeric(cat_points[test_all_pts,3])
        dist_i_to_pnt <- sqrt( ((x - pc_x)^2) + ((y - pc_y)^2) )
        # have to figure out what to do with lines 285-286 and possibly 284
        #com_all_pnts[test_all_pts,] <- c(cat_points[test_all_pts,1],dist_i_to_pnt)
        #index_num <- which(com_all_pnts == min(com_all_pnts)) - n_all
        #less_rest_inf_pop <- com_all_pnts[index_num]
      }
      # then here put finding the lowest value and getting the less_rest_inf_pop from com_all_pnts[index_num]. Below may have to change
      #com_all_pnts[test_all_pts,] <- c(cat_points[test_all_pts,1],dist_i_to_pnt)
      com_all_pnts[test_all_pts,] <- c(cat_points[test_all_pts,1],format(as.numeric(dist_i_to_pnt), scientific=F))
      index_num <- which(com_all_pnts == min(com_all_pnts)) - n_all
      less_rest_inf_pop <- com_all_pnts[index_num]
    }
  } else {
    if (inferred_pop_1SD != "-"){
      less_rest_inf_pop <- inferred_pop_1SD
    } else {
      if (inferred_pop_2SD != "-") {
        less_rest_inf_pop <- inferred_pop_2SD 
      } else {
        less_rest_inf_pop <- inferred_pop_3SD
      }
    }}  

  vector_to_add_df <- c(vector_to_add_df, less_rest_inf_pop)

  
  for (cluster in 1:length(mat_meanx_meany_rad[,1])) {
    x <- as.numeric(mat_meanx_meany_rad[cluster,2])
    y <- as.numeric(mat_meanx_meany_rad[cluster,3])
    dist_i_to_cluster_centriod <- sqrt( ((x - pc_x)^2) + ((y - pc_y)^2) )
    #cluster_dist_i <- paste(dist_i_to_cluster_centriod)
    cluster_dist_i <- format(round(dist_i_to_cluster_centriod, 5), nsmall = 5)
    vector_to_add_df <- c(vector_to_add_df, cluster_dist_i)
  }

  df[iterator, ] = vector_to_add_df
}

write.table(df, inferredpopreport, sep = "\t", quote = F, row.names = F)

dev.off() # close pdf

