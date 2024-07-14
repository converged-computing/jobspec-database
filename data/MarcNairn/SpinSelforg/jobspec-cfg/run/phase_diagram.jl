include("../src/selforg_core.jl")
include("../src/custom_functions.jl")

sim_index = parse(Int, ARGS[1])

### PARAMETERS ###
N = parse(Int, ARGS[2]); ### ATOM NUMBER (read from console)

kappa = 100
omegauv = 5:2:60; ### LIST OF ATOMIC FREQUENCIES
Deltac = -kappa;
Ntraj = 1 ### TRAJECTORY NUMBER
tlist = (0.0, 10.0) ### SIM TIME
deltaD = 10; ### DOPPLER WIDTH ~ temperature 
gvrange = 5:2:70; ### UNNORMALISED COUPLING

gv = sqrt.(-1 * (kappa^2 + Deltac^2) / (2 * Deltac) / N) * sqrt.(gvrange); ### RESCALE COUPLING STRENGTH

### PARAMETER ARRAY
p_array = [
    System_p(0.0, 0.0, g, g, omegau, Deltac, kappa, deltaD/2, N, tlist, Ntraj)
    for omegau in omegauv for g in gv
]


sim = extract_solution(many_trajectory_solver(p_array, saveat=0.1, seed=abs(rand(Int)), dt=1e-5));
JLD2.jldopen("short_time_sim_N=$(N)_i=$(sim_index).jld2", "w") do file
        write(file, "solution", sim)
    end

