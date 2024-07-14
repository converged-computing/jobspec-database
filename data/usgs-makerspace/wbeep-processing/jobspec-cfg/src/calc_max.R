library(ncdf4)
library(dplyr)

vars <- c(
  "soil_moist_tot",
  "pkwater_equiv",
  "hru_intcpstor",
  "hru_impervstor",
  "gwres_stor",
  "dprst_stor_hru"
)

vars_data <- c()

n_hrus <- 114958

for(var in vars) {
  
  message(sprintf("Reading in NetCDF data for %s", var))
  
  # The file is NetCDF
  fn <- sprintf("historical_%s_out.nc", var)
  nc <- nc_open(fn)
  
  data_nc <- ncvar_get(nc, var, start = c(1,1), count=c(n_hrus, -1))
  data_nc <- data_nc * 25.4 #convert to mm
  hruids <- ncvar_get(nc, "nhru")
  
  if(length(vars_data) == 0) {
    vars_data <- data_nc
  } else {
    vars_data <- vars_data + data_nc
  }
  
  hru_max <- as.data.frame(apply(vars_data, 1, max, na.rm = TRUE)) %>%
    mutate(hruid <- hruids)
  colnames(hru_max) <- c("max_value", "hruid")
  write.csv(hru_max, sprintf("max_%s.csv", var), row.names = FALSE)
  
  nc_close(nc)
  gc() # garbage cleanup
  
}


  

