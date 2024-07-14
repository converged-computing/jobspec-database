using Pkg
Pkg.instantiate()
using ArgParse
using Printf
using Oceananigans
@info "Loaded Oceananigans"
using Oceananigans.Units
using Oceananigans.Advection: WENO5
using Oceananigans.OutputWriters, Oceananigans.Fields
using SpecialFunctions: erf
using CUDA: has_cuda_gpu


#++++ Read and parse initial arguments
"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()
    @add_arg_table! settings begin

        "--factor"
            help = "Factor to divide Nh and Nz for"
            default = 32
            arg_type = Int

        "--simname"
            help = "Setup and name of jet in jetinfo.jl"
            default = "PNN_CItest1"
            arg_type = String
    end
    return parse_args(settings)
end
args = parse_command_line_arguments()
factor = args["factor"]
simname = args["simname"]
#-----


#++++ Figure out modifiers
sep = "_"
try
    global topology, jet, modifier = split(simname, sep)
    global AMD = modifier=="AMD" ? true : false
    global noflux = modifier=="NF" ? true : false
    global f2 = modifier=="f2" ? true : false
    global f4 = modifier=="f4" ? true : false
    global f8 = modifier=="f8" ? true : false
catch e
    global topology, jet = split(simname, "_")
    global AMD = false
    global noflux = false
    global f2 = false
    global f4 = false
    global f8 = false
end
#----


#++++ Change resolution based on the name
if f2
    factor = 2
elseif f4
    factor = 4
elseif f8
    factor = 8
end
#----


#++++ Figure out number of dimensions
if topology == "PNN"
    ndims = 3
elseif topology == "FNN"
    ndims = 2
else
    throw(AssertionError("Topology must be either PNN and FNN"))
end
#----


#++++ Figure out architecture
if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
@info "Starting $(ndims)d jet $jet with a dividing factor of $factor and a $arch architecture\n"
#-----


#++++ Get primary simulation parameters
include("jetinfo.jl")

if ndims==3 # 3D LES simulation
    simulation_nml = getproperty(SurfaceJetSimulations(Ny=100*2^5, Nz=2^7, ThreeD=true), Symbol(jet))
    LES = true
else # 2D DNS simulation
    simulation_nml = getproperty(SurfaceJetSimulations(), Symbol(jet))
    LES = false
end
@unpack name, f_0, u_0, N2_inf, Ny, Nz, Ly, Lz, σ_y, σ_z, y_0, z_0, νz, sponge_frac = simulation_nml

prefix = topology

pickup = any(startswith("chk.$simname"), readdir("data"))
#-----

#++++ Set GRID
if ndims==3
    Nx = Ny÷16
    Lx = (Ly / Ny) * Nx
else
    Nx = factor
    Lx = 6 * (Ly / Ny) * Nx
end
topology = (Periodic, Bounded, Bounded)

grid = RegularRectilinearGrid(size=(Nx÷factor, Ny÷factor, Nz÷factor),
                              x=(0, Lx),
                              y=(0, Ly),
                              z=(-Lz, 0), 
                              topology=topology)
@info "" grid
#-----


#++++ Calculate secondary parameters
b₀ = u_0 * f_0
ρ₀ = 1027
T_inertial = 2*π/f_0
y_r = y_0 + √2/4 * σ_y
z_r = 0
Ro_r = - √2 * u_0 * (z_0/σ_z-1) * exp(-1/8) / (2*f_0*σ_y)
Ri_r = N2_inf * σ_z^2 * exp(1/4) / u_0^2
νh = νz * (grid.Δy / grid.Δz)^(4/3)

secondary_params = merge((LES=Int(LES), ρ_0=ρ₀, b_0=b₀,), (;y_r, z_r, Ro_r, Ri_r, T_inertial, νh))

global_attributes = merge(simulation_nml, secondary_params)
@info "" global_attributes
#-----



# Set up Geostrophic flow
#++++++
const n2_inf = N2_inf
const Hz = grid.Lz
const Hy = grid.Ly
const sig_z = σ_z
const sig_y = σ_y
const u₀ = u_0
const y₀ = y_0
const z₀ = z_0
const f₀ = f_0
@inline fy(ψ) = exp(-ψ^2)
@inline intgaussian(ψ) = √π/2 * (erf(ψ) + 1)
@inline umask(Y, Z) = Z * fy(Y)
@inline bmask(Y, Z) = (1/sig_z) * (sig_y * intgaussian(Y))

u_g(x, y, z, t) = +u₀ * umask((y-y₀)/sig_y, ((z-z₀)/sig_z +1))
@inline background_strat(z) = n2_inf * (z+Hz)
b_g(x, y, z, t) = -f₀ * u₀ * bmask((y-y₀)/sig_y, ((z-z₀)/sig_z +1)) + background_strat(z)
#-----

# Setting BCs
#++++
U_top_bc = FluxBoundaryCondition(0)
U_bot_bc = FluxBoundaryCondition(0)
if noflux
    B_bc = FluxBoundaryCondition(0)
else
    B_bc = GradientBoundaryCondition(N2_inf)
end

ubc = FieldBoundaryConditions(top = U_top_bc, bottom = U_bot_bc,)
vbc = FieldBoundaryConditions()
wbc = FieldBoundaryConditions()
bbc = FieldBoundaryConditions(bottom = B_bc, top = B_bc,)
#-----


# Set-up sponge layer
#++++
@inline heaviside(X) = ifelse(X < 0, zero(X), one(X))
@inline mask2nd(X) = heaviside(X) * X^2
const frac = sponge_frac

@inline function north_mask(x, y, z)
    y₁ = Hy; y₀ = y₁ - Hy*frac
    return mask2nd((y - y₀)/(y₁ - y₀))
end
@inline function south_mask(x, y, z)
    y₁ = 0; y₀ = y₁ + Hy*frac
    return mask2nd((y - y₀)/(y₁ - y₀))
end
@inline full_mask(x, y, z) = north_mask(x, y, z) + south_mask(x, y, z)

const rate = 1/10minutes
full_sponge_0 = Relaxation(rate=rate, mask=full_mask, target=0)
full_sponge_u = Relaxation(rate=rate, mask=full_mask, target=u_g)
forcing = (u=full_sponge_u, v=full_sponge_0, w=full_sponge_0)
#-----



# Set up ICs and/or Background Fields
#++++
const amplitude = 1e-6
u_ic(x, y, z) = u_g(x, y, z, 0) + amplitude*randn()
v_ic(x, y, z) = + amplitude*randn()
w_ic(x, y, z) = + amplitude*randn()
b_ic(x, y, z) = b_g(x, y, z, 0) #+ 1e-8*randn()
#-----


# Define model!
#++++
if LES
    import Oceananigans.TurbulenceClosures: SmagorinskyLilly, AnisotropicMinimumDissipation
    νₘ, κₘ = 1.0e-6, 1.5e-7
    if AMD
        closure = AnisotropicMinimumDissipation(ν=νₘ, κ=κₘ)
    else
        closure = SmagorinskyLilly(C=0.16, ν=νₘ, κ=κₘ)
    end
else
    import Oceananigans.TurbulenceClosures: AnisotropicDiffusivity, IsotropicDiffusivity
    closure = AnisotropicDiffusivity(νh=νh, κh=νh, νz=νz, κz=νz)
end
model = NonhydrostaticModel(architecture = arch,
                            grid = grid,
                            advection = WENO5(),
                            timestepper = :RungeKutta3,
                            coriolis = occursin("slosh", lowercase(simname)) ? FPlane(f=0) : FPlane(f=f₀), # do we want sloshing?
                            tracers = (:b,),
                            buoyancy = BuoyancyTracer(),
                            boundary_conditions = (b=bbc, u=ubc, v=vbc, w=wbc),
                            forcing = forcing,
                            closure = closure,
                            )
println("\n", model, "\n")
#-----


# Adding the ICs
#++++
set!(model, u=u_ic, v=v_ic, w=w_ic, b=b_ic)

v̄ = sum(model.velocities.v.data.parent) / (grid.Nx * grid.Ny * grid.Nz)
w̄ = sum(model.velocities.w.data.parent) / (grid.Nx * grid.Ny * grid.Nz)
model.velocities.v.data.parent .-= v̄
model.velocities.w.data.parent .-= w̄
#-----


# Define time-stepping
#++++
u_scale = abs(u₀)
Δt = 1/2 * min(grid.Δz, grid.Δy) / u_scale
wizard = TimeStepWizard(cfl=0.9,
                        diffusive_cfl=0.9,
                        Δt=Δt, max_change=1.02, min_change=0.2, max_Δt=Inf, min_Δt=0.1seconds)
#----


# Finally define Simulation!
#++++
if ndims==3 # 3D LES simulation
    stop_time = min(12*T_inertial, 20days)
else # 2D DNS simulation
    stop_time = min(2.5*T_inertial, 20days)
end
include("diagnostics.jl")
start_time = 1e-9*time_ns()
using Oceanostics: SingleLineProgressMessenger
simulation = Simulation(model, Δt=wizard, 
                        stop_time=stop_time,
                        wall_time_limit=23.5hours,
                        iteration_interval=1,
                        progress=SingleLineProgressMessenger(LES=LES, initial_wall_time_seconds=start_time),
                        stop_iteration=Inf,)
@info "" simulation
#-----


# DIAGNOSTICS
#++++
const ρ0 = ρ₀
checkpointer = construct_outputs(model, simulation, LES=LES, simname=simname, frac=frac, ndims=ndims)
#-----


# Run the simulation!
#+++++
println("\n", simulation,
        "\n",)
if has_cuda_gpu() run(`nvidia-smi`) end


@printf("---> Starting run!\n")
run!(simulation, pickup=true)

using Oceananigans.OutputWriters: write_output!
write_output!(checkpointer, model)
#-----
