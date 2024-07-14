using JLD2
include("MDPModelFunctions2.jl")

# array job stuff
is_array_job = true
run_idx = is_array_job ? parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) : 5
on_cluster = true

if on_cluster
    to_save_folder = "/scratch/gpfs/erussek/Memory_Models/tanou_parameter_search"
else
    to_save_folder = "/Users/erussek/Dropbox/Griffiths_Lab_Stuff/data/Memory_Models/tanou_parameter_search"
end

mkpath(joinpath(to_save_folder,"exp1"))

## Specify parameters for each job...
# 21 epsilon values

eps_vals = collect(1:-.02:.02) # 17

# quanta values
q_vals = collect(2:2:100) # could go to one... 

mem_slopes = [.05, .1, .2]

# re-run with new values so we can see some concavity...
NT_vals = [1600, 3200] # run w these now... 

job_eps = []
job_q = []
job_nt = []
job_mem_slopes = []


for ep in eps_vals
    for q in q_vals
        for NT_idx in 1:length(NT_vals)
            for MS_idx = 1:length(mem_slopes)
            
                nt = NT_vals[NT_idx]
                ms = mem_slopes[MS_idx]

                push!(job_eps, ep)
                push!(job_q, q)
                push!(job_nt, nt)
                push!(job_mem_slopes, ms)
                
            end
        end
    end
end


n_jobs_total = length(job_nt)

println("N_Jobs_Total: $n_jobs_total")

n_jobs_per_run = 50# 
n_runs = Int(ceil(n_jobs_total/n_jobs_per_run))
println("N_Runs: $n_runs")

# run_job_idx = 

jobs_per_run = reshape(Vector(1:(n_jobs_per_run*n_runs)), (n_jobs_per_run, n_runs))'

these_jobs = jobs_per_run[run_idx,:]

N_Trials = 1000 # consider making this 1000

for this_job_idx in these_jobs
    
    if (this_job_idx > n_jobs_total)
        break
    end

    local N_Quanta = job_q[this_job_idx]
    local epsilon = job_eps[this_job_idx]
    local NT_per_Second = job_nt[this_job_idx]
    local mem_slope = job_mem_slopes[this_job_idx]

    println("Job: $this_job_idx, N_Quanta: $N_Quanta, epsilon: $epsilon, NT_per_Second: $NT_per_Second, mem_slope: $mem_slope")
    
    local file_name = "N_Quanta_$(N_Quanta)_epsilon_$(epsilon)_NT_per_Second_$(NT_per_Second)_memslope_$(mem_slope).jld2"
    
    local job_res_1 = sim_tanoue_exp1(epsilon, N_Quanta, NT_per_Second; mem_slope = mem_slope, return_last_only=true, N_Trials = N_Trials);
    local full_file_path = joinpath(to_save_folder,"exp1",file_name)
    jldsave(full_file_path; job_res_1)

    
    # testing this out -- garbage collection
    GC.gc(true)
    
end


