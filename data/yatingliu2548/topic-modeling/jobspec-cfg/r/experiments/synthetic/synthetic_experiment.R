source("r/experiments/semi_synthetic/synthetic_AP.R")
source("r/experiments/synthetic/synthetic_dataset.R")

args = commandArgs(trailingOnly=TRUE)
seed = ceiling(as.numeric(args[1]))
result_file = args[2]
K = ceiling(as.numeric(args[3]))
matlab_path = args[4]
a_zipf = as.numeric(args[5]
		    )
delta_anchor = as.numeric(args[6])
#matlab_path = DEFAULT_MATLAB
error <- c()


print(paste0("Delta anchor is: ", delta_anchor))
anchors = c(0, 1, 5)
tot = sapply(anchors, function(x){x * K})
p = 10000

for (exp_seed in 1:1){
  for (n in c(c(100, 250, 500, 750, 1000), 2000)){
    for (N in sort(c(50, 100, 300, 500, 750, 1000, 2000, 3000, 5000, 10000), decreasing=TRUE)){
    #for (N in c(5000, 10000)){
      for (n_anchors in anchors){
        for (p in c(1000, 5000, 10000)){
         #for (p in c(50000, 100000)){
	  #for (a_zipf in c(0.5, 1, 2, 4)){
           #K=3
           #seed = 1
           if (K <5){
              VHMethod = "SVS"
            }else{
              VHMethod = "SP"
            }
            test <- run_synthetic_experiment(n, K, p, alpha=0.5, a_zipf=a_zipf,
                                             n_anchors=n_anchors, delta_anchor=delta_anchor, N=N,
                                             seed=100 * seed + exp_seed, VHMethod=VHMethod,
                                             data_generation_method=1,
                                             normalize_counts = TRUE)
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
            error_temp["seed"] =  100 * seed + exp_seed
            error_temp["n_anchors"] =  n_anchors
            error_temp["alpha"] = a_zipf
            error_temp["exp"] = result_file
            error_temp["exp_n"] = exp_seed
            error_temp["VHMethod"] = VHMethod
        error_temp["delta_anchor"] = delta_anchor
	    error <- rbind(error,
                           error_temp)
            write_csv(error, paste0(getwd(), paste0("/r/experiments/synthetic/results/new_exp_",paste0(result_file, '_delta_anchor', delta_anchor, '_K_' , K), ".csv")))
            
          #}
        }
      }
    }
          
  }
        
}



