#!/usr/bin/env Rscript

packages <- c("optparse", "Seurat", "SeuratDisk")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# The `anndata` library might not be directly available in CRAN or Bioconductor.
# If you have Python code or functionality that requires anndata, ensure Python and anndata are installed in your environment.
# The R `reticulate` package can be used to interface with Python, including anndata, from R.


# Load required libraries
library(optparse)
library(Seurat)
library(SeuratDisk)



# Define the function to convert .h5ad file to .h5seurat
h5ad_to_h5seurat <- function(input.h5ad.file){
  
  
  # Convert .h5ad to .h5seurat
  h5seurat_path <- paste0(input.h5ad.file, ".h5seurat")
  Convert(input.h5ad.file, dest = "h5seurat", overwrite = TRUE)
  cat("h5seurat file created : ", h5seurat_path, "\n")
}




# Define command-line options
option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = "", 
              help = "Path to the input .h5ad file", metavar = "FILE")
)

# Parse command-line arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Validate input
if (opt$input == "") {
  stop("No input .h5ad file provided. Use -i or --input to specify the .h5ad file.", call. = FALSE)
}


# MAIN - Call the conversion function
h5ad_to_h5seurat(input.h5ad.file = opt$input)


