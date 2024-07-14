source("r/experiments/semi_synthetic/synthetic_AP.R")
source("r/experiments/synthetic/synthetic_dataset.R")

args = commandArgs(trailingOnly=TRUE)
seed = ceiling(as.numeric(args[1]))
result_file = args[2]
K <- ceiling(as.numeric(args[3]))
alpha_dirichlet <-  as.numeric(args[4])
n_anchors <-  as.numeric(args[5])
delta_anchor <- as.numeric(args[6])
N =  as.numeric(args[7])
#matlab_path = DEFAULT_MATLAB
error <- c()



a_zipf = 1
#vary_by_topic = FALSE
zipf_offset = 2.7
alpha_dirichlet = 1


for (p in c(c(10000))){
  for (n in c(500, 1000, 2000, 5000, 7500, 10000, 15000, 20000, 50000)){
    for (vary_by_topic in c(TRUE, FALSE)){
      if (K <5){
        VHMethod = "SVS"
      }else{
        VHMethod = "SP"
      }
      test <- run_synthetic_experiment(n, K, p, alpha=alpha_dirichlet, a_zipf=a_zipf,
                                       n_anchors=n_anchors, delta_anchor=delta_anchor, N=N,
                                       seed=  seed, 
                                       VHMethod=VHMethod,
                                       data_generation_method=1,
                                       normalize_counts = TRUE, 
                                       offset_zipf=zipf_offset,
                                       vary_by_topic=vary_by_topic)
      error_temp = test$error
      error_temp["Khat_huy"]=test$Khat_huy
      error_temp["Khat_huy_thresh"] = test$Khat_huy_thresh
      error_temp["Khat_olga"]= test$Khat_olga
      error_temp["Khat_olga_thresh"] = test$Khat_olga_thresh
      error_temp["Khat_tracy"]=test$Khat_tracy
      error_temp["Khat_tracy_thresh"] = test$Khat_tracy_thresh
      error_temp["N"] = N
      error_temp["n"] = n
      error_temp["p"] = p
      error_temp["seed"] =  100 * seed
      error_temp["n_anchors"] =  n_anchors
      error_temp["alpha"] = a_zipf
      error_temp["alpha_dirichlet"] = alpha_dirichlet
      error_temp["zipf_offset"] = zipf_offset
      error_temp["delta_anchor"] = delta_anchor
      error_temp["exp"] = result_file
      error_temp["vary_by_topic"] = vary_by_topic
      error_temp["VHMethod"] = VHMethod
      error <- rbind(error,
                     error_temp)
      write_csv(error, paste0(getwd(), paste0("/r/experiments/synthetic/results/",paste0(result_file, '_delta_anchor', delta_anchor, '_K_' , K), ".csv")))
    }
    
  }
  
}



