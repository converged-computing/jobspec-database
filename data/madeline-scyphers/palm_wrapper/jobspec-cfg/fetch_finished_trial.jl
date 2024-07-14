using JSON
# using FileIO
using NCDatasets
using Mustache
using Dates
import Statistics

function get_mean_from_netcdf(netcdf_path)
    ds = NCDatasets.Dataset(netcdf_path, "r")
    mean = Statistics.mean(skipmissing(ds["s"][:]))
    return mean
end

# Get the BOA trial_dir from the command line arguments
trial_dir::String = string(ARGS[1])

JOB_DIR::String = joinpath(homedir(), "palm", "current_version", "JOBS")

data::Dict = JSON.parsefile(joinpath(trial_dir, "submit_dir.json"))
submit_dir::String = data["submit_dir"]
job_name::String = basename(submit_dir)

output_file::String = joinpath(submit_dir, "OUTPUT", "$(job_name)_3d.nc")

if isfile(output_file)
    scalar_concentration = @time get_mean_from_netcdf(output_file)
    output = Dict("scalar_concentration" => scalar_concentration)
else
    output = Dict("trial_status" => "FAILED")
end

open(joinpath(trial_dir, "output.json"), "w") do f
    JSON.print(f, output)
end
