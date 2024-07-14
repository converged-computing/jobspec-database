using Pkg
Pkg.activate(".")
# Pkg.instantiate()  # Not needed if we've already installed packages into this project when we prepared the inputs

using CSV, DataFrames, Printf
using Random

function calc_pi(n::Integer)
   x = rand(n)
   y = rand(n)
   num_in_unit_circle = sum(x.*x.+y.*y .< 1)
   num_in_unit_circle * 4/n
end

job_array_data = CSV.read("ex1_job_array_in.csv")
job_arrayid = parse(Int64,ARGS[1])


idx = findfirst(x-> x==job_arrayid, job_array_data[:array_id] )
@assert idx != nothing
@assert 1 <= idx  <= size(job_array_data, 1)

n = job_array_data[idx,:n]
s = job_array_data[idx,:seed]

Random.seed!(s)

pi_estimate = calc_pi(n)
println("# After ", n, " itterations, array_id= ", job_arrayid, " estimated pi to be...")
println(pi_estimate)

output_filename = @sprintf("ex1_out_%02d.csv",job_arrayid) 
df_out = DataFrame(result = [pi_estimate]) 
CSV.write(output_filename, df_out)

