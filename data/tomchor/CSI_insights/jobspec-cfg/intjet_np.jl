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
using CUDA: has_cuda


# Parse initial arguments
#++++
"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()
    @add_arg_table! settings begin

        "--factor"
            help = "Factor to divide Nh and Nz for"
            default = 8
            arg_type = Int

        "--arch"
            help = "CPU or GPU"
            default = "CPU"
            arg_type = String

        "--fullname"
            help = "Setup and name of jet in jetinfo.jl"
            default = "I3d_CIjet01"
            arg_type = String
    end
    return parse_args(settings)
end
args = parse_command_line_arguments()
factor = args["factor"]
arch = eval(Meta.parse(args["arch"]*"()"))
fullname = args["fullname"]


setup, jet = split(fullname, "_")
ndims = parse(Int, strip(setup, ['I', 'd']))
jet = Symbol(jet)

if ndims ∉ [2,3] # Validade input
    throw(AssertionError("Dimensions must be 2 or 3"))
end

@printf("Starting %sd jet %s with a dividing factor of %d and a %s architecture\n", 
        ndims,
        jet,
        factor,
        arch,)
#-----


# Get simulation parameters
#++++
as_background=false
include("jetinfo.jl")

if ndims==3 # 3D LES simulation
    simulation_nml = getproperty(InteriorJetSimulations(Ny=400*2^3, Nz=2^8, three_d=true), jet)
    prefix = "PNN"
    LES = true
else # 2D DNS simulation
    simulation_nml = getproperty(InteriorJetSimulations(three_d=false), jet)
    prefix = "FNN"
    LES = false
end
@unpack name, f0, u₀, N2_inf, N2_pyc, Ny, Nz, Ly, Lz, σy, σz, y₀, z₀, νz, sponge_frac = simulation_nml

simname = "$(prefix)_$(name)"
pickup = any(startswith("chk.$simname"), readdir("data"))
#-----

# Set GRID
#++++  GRID
if ndims==3
    Nx = Ny÷32
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
println("\n", grid, "\n")
#-----


# Calculate secondary parameters
#++++
b₀ = u₀ * f0
ρ₀ = 1027
T_inertial = 2*π/f0
νh = νz * (grid.Δy / grid.Δz)^(4/3)

global_attributes = merge(simulation_nml,
                          (LES=Int(LES),
                          u_0=u₀, y_0=y₀, z_0=z₀,
                          b0 = b₀, T_inertial = T_inertial, νh=νh),
                         )
println("\n", global_attributes, "\n")
#-----


# Set up Geostrophic flow
#++++++
const n2_inf = N2_inf
const n2_pyc = N2_pyc
const Hz = grid.Lz
const Hy = grid.Ly
const sig_z = σz
const sig_y = σy
const u_0 = u₀
const y_0 = y₀
const z_0 = z₀
const z_c = -40
const z_m = z_c - n2_pyc/n2_inf*(z_c+Hz)
const f_0 = f0

@inline gaussian(ψ) = exp(-ψ^2)
@inline intgaussian(ψ) = √π/2 * (erf(ψ) + 1)
@inline umask(Y, Z) = gaussian(Y) * gaussian(Z)
@inline bmask(Y, Z) = (sig_y * intgaussian(Y)) * (-2Z * gaussian(Z) / sig_z)

u_g(x, y, z, t) = u_0 * umask((y-y_0)/sig_y, (z-z_0)/sig_z)
b_g(x, y, z, t) = -u_0*f_0 * bmask((y-y_0)/sig_y, (z-z_0)/sig_z) + n2_inf * (z+Hz)
@inline dudz_g(x, y, z, t) = + u_0 * gaussian((y-y_0)/sig_y) * (-2(z-z_0)/sig_z^2) * gaussian((z-z_0)/sig_z)
#-----

# Setting BCs
#++++
if as_background
    @inline surface_grad(x, y, t) = -dudz_g(x, y, 0, t)
    @inline bottom_grad(x, y, t) = -dudz_g(x, y, -Hz, t)
    U_top_bc = GradientBoundaryCondition(surface_grad)
    U_bot_bc = GradientBoundaryCondition(bottom_grad)
    B_bc = GradientBoundaryCondition(0)
else
    U_top_bc = FluxBoundaryCondition(0)
    U_bot_bc = FluxBoundaryCondition(0)
    B_bc = GradientBoundaryCondition(N2_inf)
end

ubc = UVelocityBoundaryConditions(grid, 
                                  top = U_top_bc,
                                  bottom = U_bot_bc,
                                  )
vbc = VVelocityBoundaryConditions(grid, 
                                  top = FluxBoundaryCondition(0),
                                  bottom = FluxBoundaryCondition(0),
                                  )
wbc = WVelocityBoundaryConditions(grid, 
                                  )
bbc = TracerBoundaryConditions(grid, 
                               bottom = B_bc,
                               top = B_bc,
                               )
#-----


# Set-up sponge layer
#++++
@inline heaviside(X) = ifelse(X < 0, zero(X), one(X))
@inline mask2nd(X) = heaviside(X) * X^2
const frac = sponge_frac

@inline function bottom_mask(x, y, z)
    z₁ = -Hz; z₀ = z₁ + Hz*frac
    return mask2nd((z - z₀)/(z₁ - z₀))
end
@inline function top_mask(x, y, z)
    z₁ = +Hz; z₀ = z₁ - Hz*frac
    return mask2nd((z - z₀)/(z₁ - z₀))
end
@inline function north_mask(x, y, z)
    y₁ = Hy; y₀ = y₁ - Hy*frac
    return mask2nd((y - y₀)/(y₁ - y₀))
end
@inline function south_mask(x, y, z)
    y₁ = 0; y₀ = y₁ + Hy*frac
    return mask2nd((y - y₀)/(y₁ - y₀))
end
full_mask(x, y, z) = north_mask(x, y, z) + south_mask(x, y, z) #+ bottom_mask(x, y, z) + top_mask(x, y, z)

const rate = 1/30minutes
full_sponge_0 = Relaxation(rate=rate, mask=full_mask, target=0)
if as_background
    forcing = (u=full_sponge_0, v=full_sponge_0, w=full_sponge_0)
else
    full_sponge_u = Relaxation(rate=rate, mask=full_mask, target=u_g)
    full_sponge_b = Relaxation(rate=rate, mask=full_mask, target=b_g)
    forcing = (u=full_sponge_u, v=full_sponge_0, w=full_sponge_0)
end
#-----



# Set up ICs and/or Background Fields
#++++
const kick = 0
if as_background
    println("\nSetting geostrophic jet as BACKGROUND\n")
    u_ic(x, y, z) = + kick*randn()
    v_ic(x, y, z) = + kick*randn()
    w_ic(x, y, z) = + kick*randn()
    b_ic(x, y, z) = + 1e-8*randn()

    bg_fields = (u=u_g, b=b_g,)
else
    println("\nSetting geostrophic jet as an INITIAL CONDITION\n")
    u_ic(x, y, z) = u_g(x, y, z, 0) + kick*randn()
    v_ic(x, y, z) = + kick*randn()
    w_ic(x, y, z) = + kick*randn()
    b_ic(x, y, z) = b_g(x, y, z, 0) + 1e-8*randn()

    bg_fields = NamedTuple()
end
#-----


# Define our model!
#++++
if LES
    import Oceananigans.TurbulenceClosures: SmagorinskyLilly, AnisotropicMinimumDissipation
    closure = SmagorinskyLilly(C=0.16)
    #closure = AnisotropicMinimumDissipation()
else
    import Oceananigans.TurbulenceClosures: AnisotropicDiffusivity, IsotropicDiffusivity
    closure = AnisotropicDiffusivity(νh=νh, κh=νh, νz=νz, κz=νz)
end
model_kwargs = (architecture = arch,
                grid = grid,
                advection = WENO5(),
                timestepper = :RungeKutta3,
                coriolis = occursin("slosh", lowercase(simname)) ? FPlane(f=0) : FPlane(f=f0), # do we want sloshing?
                tracers = (:b,),
                buoyancy = BuoyancyTracer(),
                boundary_conditions = (b=bbc, u=ubc, v=vbc, w=wbc),
                forcing = forcing,
                background_fields = bg_fields,
                )
model = IncompressibleModel(; model_kwargs..., closure=closure)
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


# Define time-stepping and printing
#++++
u_scale = abs(u₀)
Δt = 1/2 * min(grid.Δz, grid.Δy) / u_scale
wizard = TimeStepWizard(cfl=0.9,
                        diffusive_cfl=0.9,
                        Δt=Δt, max_change=1.02, min_change=0.2, max_Δt=Inf, min_Δt=0.2seconds)
#-----


# Finally define Simulation!
#++++
if ndims==3 # 3D LES simulation
    stop_time = min(10*T_inertial, 20days)
else # 2D DNS simulation
    stop_time = min(10*T_inertial, 20days)
end
include("diagnostics.jl")
start_time = 1e-9*time_ns()
using Oceanostics: SingleLineProgressMessenger
simulation = Simulation(model, Δt=wizard, 
                        stop_time=stop_time,
                        wall_time_limit=23.5hours,
                        iteration_interval=5,
                        progress=SingleLineProgressMessenger(LES=LES, initial_wall_time_seconds=start_time),
                        stop_iteration=Inf,)
println("\n", simulation, "\n")
#-----


# DIAGNOSTICS
#++++
const ρ0 = ρ₀
checkpointer = construct_outputs(model, simulation, LES=LES, simname=simname, frac=frac)
#-----


# Run the simulation!
#+++++
println("\n", simulation,
        "\n",)
if has_cuda() run(`nvidia-smi`) end


@printf("---> Starting run!\n")
run!(simulation, pickup=true)

using Oceananigans.OutputWriters: write_output!
write_output!(checkpointer, model)
#-----
