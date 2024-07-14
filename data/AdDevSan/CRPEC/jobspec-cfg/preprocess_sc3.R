

library(Seurat)
library(SeuratDisk)
library(SingleCellExperiment)
library(SC3)
library(jsonlite)

# TODO: for each loop run, get sample_200 file ending with loop run from sample.200.directory


  # TODO: subset sce with sample_200 into sce_200
  # TODO: predict SC3 cluster into initial_cluster
    # TODO: get optimal cluster
    # TODO: predict
  # TODO: write initial_cluster into json {initial_cluster_<loop_run>.json} into initial.cluster.directory


input.h5seurat.file.path = "get from input"
input.barcodes.directory.path = "get from input"
output.directory = "get from input"





h5seurat_preprocess_to_sce <- function(input.h5seurat.file.path){
  aseurat<- LoadH5Seurat(input.h5seurat.file.path)
  asce <- as.SingleCellExperiment(aseurat)
  rowData(asce)$feature_symbol <- rownames(asce)
  log_counts <- log1p(counts(asce))
  assay(asce, "logcounts") <- log_counts
  counts(asce) <- as.matrix(counts(asce))
  logcounts(asce) <- as.matrix(logcounts(asce))
  
  return(asce)
}




#Function to Subset SCE by Sample 200

subset_sce_by_sample_200 <- function(sce, sample_200_path) {
  barcodes_200 <- read.table(sample_200_path, header = FALSE)$V1
  sce_200 <- sce[, colnames(sce) %in% barcodes_200]
  return(sce_200)
}



create_json_object <- function(keys, values) {
  # Ensure 'values' is treated as numeric
  values_numeric <- as.numeric(as.character(values))
  
  # Create a named list where each key is associated with its value
  json_list <- setNames(as.list(values_numeric), keys)
  
  # Convert the named list to a JSON object
  json_output <- toJSON(json_list, pretty = TRUE, auto_unbox = TRUE)
  
  # Return the JSON output
  return(json_output)
}

process_and_save_clusters <- function(asce, sample_200_path, output_directory) {
  # Read the barcodes
  barcodes_200 <- read.table(sample_200_path, header = FALSE)$V1
  
  # Subset the SCE object by barcodes
  asce_200 <- asce[, colnames(asce) %in% barcodes_200]
  
  # Perform SC3 clustering
  asce_200 <- sc3(asce_200, k_estimator = TRUE)
  
  # Retrieve the estimated k
  k_est <- metadata(asce_200)$sc3$k_est
  cat(paste0("Estimated k: ", k_est, "\n"))
  
  # Extract the clustering results
  cluster_col_name <- paste0("sc3_", k_est, "_clusters")
  clusters <- asce_200[[cluster_col_name]]
  
  json_output <- create_json_object(colnames(asce_200), as.numeric(as.character(clusters)))
  
  #cat(json_output)
  filename_prefix <- gsub("sample", "initial", tools::file_path_sans_ext(basename(sample_200_path)))
  output_filename = file.path(output_directory, paste0(filename_prefix, ".json"))
  write(json_output, output_filename)
  

  cat("Cluster dictionary saved as JSON: ", output_filename, "\n")
}








# Example usage:
# asce <- Your code to generate the SingleCellExperiment object
# temprorary
input.h5seurat.file.path = "runs/archive/full_dataset.h5seurat"
input.barcodes.directory.path = "runs/CRPEC_run_trial/sample_200"
output.directory = "runs/CRPEC_run_trial/initial_clusters"



asce <- h5seurat_preprocess_to_sce(input.h5seurat.file.path = input.h5seurat.file.path)
input.barcodes.directory.path <- "runs/CRPEC_run_trial/sample_200"
output.directory <- "runs/CRPEC_run_trial/initial_clusters"

# Assume asce is your pre-loaded SingleCellExperiment object
sample_200_files <- list.files(input.barcodes.directory.path, 
                               pattern = "^sample_200_\\d+\\.tsv$", 
                               full.names = TRUE)

for (sample_200_path in sample_200_files) {
  process_and_save_clusters(asce, sample_200_path, output.directory)
}



### ARCHIVE ################

sample.200.path = "runs/CRPEC_run_trial/sample_200/sample_200_1.tsv"
barcodes_200 <- read.table(sample.200.path, header = FALSE)$V1
asce_200 <- asce[, colnames(asce) %in% barcodes_200]
asce_200 <- sc3(asce_200, k_estimator = TRUE)