using JSON
using NCDatasets
using Mustache
using Dates
using CSV, Tables
using DataFrames

NORMALIZED_LAD = [0. 0.12465747 0.2203851 0.57814588 0.07681155]
# # JOB_DIR = ENV["TMPDIR"]
# JOB_DIR = joinpath(homedir(), "palm", "current_version", "JOBS")

# Get the BOA trial_dir from the command line arguments
job_name = string(ARGS[1])
trial_dir = string(ARGS[2])

# Read in the json file into a variable called data
data = JSON.parsefile(joinpath(trial_dir, "parameters.json"))

if data["io_config"] == "slurm"
    JOB_DIR = ENV["TMPDIR"]
elseif data["io_config"] == "d3#"
    JOB_DIR = joinpath(homedir(), "palm", "current_version", "JOBS")
else
    error("io_config must be either slurm or d3#")
end
# Concatenate job_name_now with dirname of trial_dir to ensure unique job_name
# job_name = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS") * "_" * basename(trial_dir)
# job_name = "$(basename(trial_dir))_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))"
submit_dir = joinpath(JOB_DIR, job_name)

println(data)
println(submit_dir)

# Write the dictionary to a JSON file
open(joinpath(trial_dir, "submit_dir.json"), "w") do f
    JSON.print(f, Dict("submit_dir" => submit_dir))
end

mkpath(joinpath(submit_dir, "INPUT"))
# mkpath(joinpath(submit_dir, "USER_CODE"))

# Calculate the number of rows
n_rows = Int(floor(data["domain_y"] / (data["row_width"] + data["spacing_width"])))

# create the domain matrix
domain = zeros(data["domain_x"], data["domain_y"], length(NORMALIZED_LAD))
lad = NORMALIZED_LAD .* data["lai"]

# Loop over the rows and set row_widths to lai and row_spacing to 0
for i in 1:n_rows
    for j in 1:data["row_width"]
        domain[:, (i-1)*(data["row_width"] + data["spacing_width"]) + j, :] .= lad
    end
end
println(size(domain))
# println(domain)

df = DataFrame(domain[:,:,3], :auto)

CSV.write(joinpath(trial_dir, "domain.csv"), df, writeheader=false)

# open(joinpath(trial_dir, "domain.txt"), "w") do io
#     writedlm(io, domain)
# end

# Copy the custom user_module.f90 file to the user_code directory
# cp(joinpath(@__DIR__, "user_module.f90"), joinpath(submit_dir, "USER_CODE", "user_module.f90"))
# copy topo over

println("Copying topo")
cp(joinpath(@__DIR__, "topo"), joinpath(submit_dir, "INPUT", "$(job_name)_topo"))

println("Copying netcdf")
netcdf_path = joinpath(@__DIR__, "static.nc")
netcdf_out_path = joinpath(submit_dir, "INPUT", "$(job_name)_static")
# Copy the file from netcdf_path to netcdf_out_path
cp(netcdf_path, netcdf_out_path)
# modify to add the lad variable
ds = Dataset(netcdf_out_path,"a")
println("Editing netcdf")
# Define the variables temperature with the attribute units
defVar(ds, "lad", domain, ("x","y", "zlad"), attrib = Dict(
    "long_name" => "leaf area density",
    "units" => "m2/m3",
    "_FillValue" => -9999.0
    ))
    close(ds);

println("Reading in palm config p3d")
# Modify the palm config file with variables from the boa config
# cp(joinpath(@__DIR__, "palm_config_template.txt"), joinpath(submit_dir, "INPUT", "$(job_name)_p3d"))
io = open(joinpath(@__DIR__, "palm_config_template.txt"))
config_in = read(io, String)
close(io)

println("Editing palm config and writing out")
open(joinpath(submit_dir, "INPUT", "$(job_name)_p3d"), "w") do f
    config_out = Mustache.render(
    config_in,
    Dict(
        "output_start_time"=>data["output_start_time"],
        "output_end_time"=>data["output_end_time"],
        "domain_x"=>data["domain_x"] - 1,  # subtract 1 to account for 0 indexing in fortran
        "domain_y"=>data["domain_y"] - 1)
    )
    write(f, config_out)
end
println("Finished copying input files")

# run(Cmd(`palmrun -r $job_name -c intel -X96 -T24 -q parallel -t $(output_end_time * 3) -a $io_config -b -v -m 4315`, dir=joinpath(homedir(), "palm", "current_version")))

# io = open(joinpath(submit_dir, "INPUT", "$(job_name)_p3d"), "w+")
# config_template_str = read(io, String);
# templated_str = Mustache.render(
#     config_template_str,
#     Dict(
#         "output_start_time"=>data["output_start_time"],
#         "output_end_time"=>data["output_end_time"],
#         "domain_x"=>data["domain_x"] - 1,  # subtract 1 to account for 0 indexing in fortran
#         "domain_y"=>data["domain_y"] - 1)
# )
# println(templated_str)
# write(io, templated_str)

# run(`palmrun -r $job_name -c intel -X96 -T24 -q parallel -t $(output_end_time * 3) -a $io_config -b -v -m 4315`)
# run(`bash palm_slurm.sh $(job_name) $(output_end_time) $(trial_dir) $(io_config)`);