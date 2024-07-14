### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 67cb574d-7bd6-40d9-9dc3-d57f4226cc83
begin
	if @isdefined PlutoRunner  # running inside Pluto
		import Pkg
		Pkg.activate("..")
		using Revise
	end
end

# ╔═╡ b6abba94-db07-4095-98c9-443e31832e7d
using Optimisers, StatsBase, Zygote, Lux, DifferentialEquations, ComponentArrays, ParameterSchedulers, Random, Distributed, ForwardDiff, LuxCore, Dates, JLD2, SciMLSensitivity, JLD2, Random123, Distributions, DiffEqBase, ChainRulesCore, DiffEqGPU

# ╔═╡ d1440209-78f7-4a9a-9101-a06ad2534e5d
using NeuralSDEExploration, Plots, PlutoUI, PlutoArgs

# ╔═╡ db557c9a-24d6-4008-8225-4b8867ee93db
begin
	if @isdefined PlutoRunner  # running inside Pluto
		Revise.retry()
	else
		ENV["GKSwstype"]="nul" # no GTK for plots
		println(join(ARGS, " "))
	end
end

# ╔═╡ d38b3460-4c01-4bba-b726-150d207c020b
TableOfContents(title="Latent SDE")

# ╔═╡ 13ef3cd9-7f58-459e-a659-abc35b550326
begin
	if @isdefined PlutoRunner  # running inside Pluto
		md"Enable training (if in notebook) $(@bind enabletraining CheckBox())"		
	else
		enabletraining = true
	end
end


# ╔═╡ ff15555b-b1b5-4b42-94a9-da77daa546d0
md"""
# Model Definition
"""

# ╔═╡ c799a418-d85e-4f9b-af7a-ed667fab21b6
println("Running on $(Threads.nthreads()) threads")

# ╔═╡ 0eec0598-7520-47ec-b13a-a7b9da550014
md"""
### Constants
"""

# ╔═╡ cb3a270e-0f2a-4be3-9ab3-ea5e4c56d0e7
md"""
Used model: $(@bind model_name Arg("model", Select(["sun", "fhn", "ou", "gbm", "zero", "diffusion", "const", "exp", "sine"]), short_name="m")), CLI arg: `--model`, `-m` (required!)
"""

# ╔═╡ 95bdd676-d8df-4fef-bdd7-cce85b717018
md"""
Noise term size: $(@bind noise_term Arg("noise", NumberField(0f0:1f0, 0.135f0), required=false)), CLI arg: `--noise`
"""

# ╔═╡ 4c3b8784-368d-49c3-a875-c54960ec9be5
md"""
Number of timeseries in data: $(@bind n Arg("num-data", NumberField(1:1000000, default=5000), required=false)), CLI arg: `--num-data`
"""

# ╔═╡ a65a7405-d1de-4de5-9391-dcb971af0413
md"""
Timestep size: $(@bind dt Arg("dt", NumberField(0.0f0:1.0f0, 0.05f0), required=false)), CLI arg: `--dt`
"""

# ╔═╡ e6a71aae-9d81-45a9-af9a-c4188dda2787
md"""
Timespan start of data generation: $(@bind tspan_start_data Arg("tspan-start-data", NumberField(0f0:100.0f0, 0f0), required=false)), CLI arg: `--tspan-start-data`
"""

# ╔═╡ 71a38a66-dd66-4000-b664-fc3f04f6d4b8
md"""
Timespan end of data generation: $(@bind tspan_end_data Arg("tspan-end-data", NumberField(0.5f0:100f0, 1f0), required=false)), CLI arg: `--tspan-end-data`
"""

# ╔═╡ bae92f09-2e87-4a1e-aa2e-906f33985f6d
md"""
Timespan start of training data: $(@bind tspan_start_train Arg("tspan-start-train", NumberField(0.5f0:100f0, tspan_start_data), required=false)), CLI arg: `--tspan-start-train`
"""

# ╔═╡ bd7acf1a-c09a-4531-ad2c-b5e7e28af382
md"""
Timespan end of training data: $(@bind tspan_end_train Arg("tspan-end-train", NumberField(0.5f0:100f0, tspan_end_data), required=false)), CLI arg: `--tspan-end-train`
"""

# ╔═╡ 42ece6c1-9e8a-45e7-adf4-6f353da6a4e5
md"""
Timespan start of model: $(@bind tspan_start_model Arg("tspan-start-model", NumberField(0.5f0:100.0f0, tspan_start_train), required=false)), CLI arg: `--tspan-start-model`
"""

# ╔═╡ 3665efa6-6527-4771-82fd-285c3c0f8b41
md"""
Timespan end of model: $(@bind tspan_end_model Arg("tspan-end-model", NumberField(0.5f0:100f0, tspan_end_train), required=false)), CLI arg: `--tspan-end-model`
"""

# ╔═╡ 10c206ef-2321-4c64-bdf4-7f4e9934d911
md"""
Likelihood scale
$(@bind scale Arg("scale", NumberField(0.001f0:1.0f0, 0.01f0), required=false)).
CLI arg: `--scale`
"""

# ╔═╡ fe7e2889-88de-49b3-b20b-342357596bfc
tspan_train = (Float32(tspan_start_train), Float32(tspan_end_train))

# ╔═╡ de70d89a-275d-49d2-9da4-4470c869e56e
tspan_data = (Float32(tspan_start_data), Float32(tspan_end_data))

# ╔═╡ 986c442a-d02e-42d4-bda4-f66a1c92f799
tspan_model = (Float32(tspan_start_model), Float32(tspan_end_model))

# ╔═╡ 9a89a97c-da03-4887-ac8c-ef1f5264436e
println((num_data=n, dt=dt, tspan_train=tspan_train, tspan_data=tspan_data, tspan_model=tspan_model))

# ╔═╡ c441712f-e4b2-4f4a-83e1-aad558685288
function steps(tspan, dt)
	return Int(ceil((tspan[2] - tspan[1]) / dt))
end

# ╔═╡ 7e6256ef-6a0a-40cc-aa0a-c467b3a524c4
md"""
### Data Generation
"""

# ╔═╡ 2da6bbd4-8036-471c-b94e-10182cf8a834
(initial_condition, model) = if model_name == "sun"
	(
		[only(rand(Normal(260e0, 40e0), 1)) for i in 1:n],
		NeuralSDEExploration.ZeroDEnergyBalanceModel(0.425, 0.4, 1363, 0.6 * 5.67e-8, noise_term)
	)
elseif model_name == "fhn"
	(
		[[only(rand(Normal(0e0, 2e0), 1)), only(rand(Normal(0e0, 0.1e0), 1))] for i in 1:n],
		NeuralSDEExploration.FitzHughNagumoModelGamma()
	)
elseif model_name == "ou"
	(
		[only(rand(Normal(0.5, 0.1), 1)) for i in 1:n],
		NeuralSDEExploration.OrnsteinUhlenbeck()
	)
elseif model_name == "gbm"
	(
		[only(rand(Normal(0.1, 0.03^2), 1)) for i in 1:n],
		NeuralSDEExploration.GeometricBM()
	)
elseif model_name == "diffusion"
	(
		[only(rand(Normal(0.0, 0.5^2), 1)) for i in 1:n],
		NeuralSDEExploration.OrnsteinUhlenbeck(0.0, 0.0, 1.0)
	)
elseif model_name == "const"
	(
		[only(rand(Normal(1.0, 0.4^2), 1)) for i in 1:n],
		NeuralSDEExploration.Drift(0.1, 0.0, 0.0)
	)
elseif model_name == "exp"
	(
		[only(rand(Normal(1.0, 0.4^2), 1)) for i in 1:n],
		NeuralSDEExploration.Drift(0.0, 1.0, 0.0)
	)
elseif model_name == "sine"
	(
		[only(rand(Normal(1.0, 0.4^2), 1)) for i in 1:n],
		NeuralSDEExploration.Drift(0.0, 0.0, 10.0)
	)
elseif model_name == "zero"
	(
		[0.0],
		NeuralSDEExploration.Zeroes()
	)
else
	@error "Invalid model name!"
	nothing
end

# ╔═╡ c00a97bf-5e10-4168-8d58-f4f9270258ac
solution_full = NeuralSDEExploration.series(model, initial_condition, tspan_data, steps(tspan_data, dt); seed=1)

# ╔═╡ 5691fcc5-29b3-4236-9154-59c6fede49ce
tspan_train_steps = (searchsortedlast(solution_full.t, tspan_start_train)): (searchsortedlast(solution_full.t, tspan_end_train))

# ╔═╡ 15cef7cc-30b6-499d-b968-775b3251dedb
solution = Timeseries(shuffle([(t=map(Float32, solution_full.t[tspan_train_steps]), u=map(Float32 ∘ first, x[tspan_train_steps])) for x in solution_full.u]))

# ╔═╡ 1502612c-1489-4abf-8a8b-5b2d03a68cb1
md"""
Let's also plot some example trajectories:
"""

# ╔═╡ 455263ef-2f94-4f3e-8401-f0da7fb3e493
plot(select_ts(1:40, solution_full))

# ╔═╡ f4651b27-135e-45f1-8647-64ab08c2e8e8
md"""
Let's normalize our data for training:
"""

# ╔═╡ aff1c9d9-b29b-4b2c-b3f1-1f06a9370f64
begin
    datamax = max([max(x[1]...) for x in solution.u]...) |> only
    datamin = min([min(x[1]...) for x in solution.u]...) |> only

    function normalize(x)
        return 2f0 * (Float32((x - datamin) / (datamax - datamin)) - 0.5f0)
    end
end

# ╔═╡ 5d78e254-4134-4c2a-8092-03f6df7d5092
println((datamin=datamin, datamax=datamax))

# ╔═╡ 9a5c942f-9e2d-4c6c-9cb1-b0dffd8050a0
timeseries = map_dims(x -> map(normalize, x), solution)

# ╔═╡ 8280424c-b86f-49f5-a854-91e7abcf13ec
md"""
# Neural Networks
"""

# ╔═╡ fdceee23-b91e-4b2e-af78-776c01733de3
md"""
### Constants
"""

# ╔═╡ 97d724c2-24b0-415c-b90f-6a36e877e9d1
md"""
The size of the context given to the posterior SDE:
$(@bind context_size Arg("context-size", NumberField(1:100, 2), required=false)).
CLI arg: `--context-size`
"""

# ╔═╡ d81ccb5f-de1c-4a01-93ce-3e7302caedc0
md"""
The hidden layer size for the ANNs:
$(@bind hidden_size Arg("hidden-size", NumberField(1:1000000, 64), required=false)).
CLI arg: `--hidden-size`
"""

# ╔═╡ 6489b190-f08f-466c-93c4-92a723f8e594
md"""
The hidden layer size for the prior:
$(@bind prior_size Arg("prior-size", NumberField(1:1000000, hidden_size), required=false)).
CLI arg: `--prior-size`
"""

# ╔═╡ b5721107-7cf5-4da3-b22a-552e3d56bcfa
md"""
Dimensions of the latent space:
$(@bind latent_dims Arg("latent-dims", NumberField(1:100, 2), required=false)).
CLI arg: `--latent-dims`
"""

# ╔═╡ efd438bc-13cc-457f-82c1-c6f0711079b3
md"""
Neural network depth:
$(@bind depth Arg("depth", NumberField(1:100, 2), required=false)).
CLI arg: `--depth`
"""

# ╔═╡ 03a21651-9c95-49e8-bb07-b03640f7e5b7
md"""
Fix projector to just use first dimension: $(@bind fixed_projector Arg("fixed-projector", CheckBox(), required=false))
CLI arg: `--fixed-projector`
"""

# ╔═╡ ad6247f6-6cb9-4a57-92d3-6328cbd84ecd
in_dims = latent_dims

# ╔═╡ 60b5397d-7350-460b-9117-319dc127cc7e
md"""
Use GPU: $(@bind gpu_enabled Arg("gpu", CheckBox(), required=false))
CLI arg: `--gpu`
"""

# ╔═╡ 6c0086c5-df79-4bc9-bada-f1c656525164
if gpu_enabled
	using CUDA, LuxCUDA
	println(CUDA.functional())
	CUDA.allowscalar(false)
end

# ╔═╡ 16c12354-5ab6-4c0e-833d-265642119ed2
md"""
Batch size
$(@bind batch_size Arg("batch-size", NumberField(1:200, 128), required=false)).
CLI arg: `--batch-size`
"""

# ╔═╡ f12633b6-c770-439d-939f-c41b74a5c309
md"""
Eta
$(@bind eta Arg("eta", NumberField(0.1f0:1000.0f0, 1.0f0), required=false)).
CLI arg: `--eta`
"""

# ╔═╡ 3c630a3a-7714-41c7-8cc3-601cd6efbceb
md"""
Learning rate
$(@bind learning_rate Arg("learning-rate", NumberField(0.0001f0:1000.0f0, 0.005f0), required=false)).
CLI arg: `--learning-rate`
"""

# ╔═╡ 2961879e-cb52-4980-931b-6f8de1f26fa4
md"""
Max learning rate
$(@bind max_learning_rate Arg("max-learning-rate", NumberField(0.0001f0:1000.0f0, 2*learning_rate), required=false)).
CLI arg: `--max-learning-rate`
"""

# ╔═╡ 7c23c32f-97bc-4c8d-ac54-42753be61345
md"""
Learning rate decay
$(@bind decay Arg("decay", NumberField(0.0001f0:2.0f0, 0.999f0), required=false)).
CLI arg: `--decay`
"""

# ╔═╡ 64e7bba4-fb17-4ed8-851f-de9204f0f42d
md"""
LR Cycling Enabled: $(@bind lr_cycle Arg("lr-cycle", CheckBox(), required=false))
CLI arg: `--lr-cycle`
"""

# ╔═╡ 33d53264-3c8f-4f63-9dd2-46ebd00f4e28
md"""
LR oscillation time
$(@bind lr_rate Arg("lr-rate", NumberField(1:100000, 500), required=false)).
CLI arg: `--lr-rate`
"""

# ╔═╡ 8bb3084f-5fde-413e-b0fe-8b2e19673fae
md"""
KL Annealing Enabled: $(@bind kl_anneal Arg("kl-anneal", CheckBox(), required=false))
CLI arg: `--kl-anneal`
"""

# ╔═╡ 2c64b173-d4ad-477d-afde-5f3916e922ef
md"""
KL Annealing oscillation time
$(@bind kl_rate Arg("kl-rate", NumberField(1:100000, 1000), required=false)).
CLI arg: `--kl-rate`
"""

# ╔═╡ 9767a8ea-bdda-43fc-b636-8681d150d29f
data_dims = length(solution.u[1][1]) # Dimensions of our input data.

# ╔═╡ 3db229f0-0e13-4d80-8680-58b89161db35
md"""
Use backsolve: $(@bind backsolve Arg("backsolve", CheckBox(true), required=false))
CLI arg: `--backsolve`
"""

# ╔═╡ cb1c2b2e-a2a2-45ed-9fc1-655d28f267d1
md"""
Brownian tree cache depth: $(@bind tree_depth Arg("tree-depth", NumberField(1:100, 2), required=false))
CLI arg: `--tree-depth`
"""

# ╔═╡ 7f219c33-b37b-480a-9d21-9ea8d898d5d5
md"""
Use Kidger's initial state trick: $(@bind kidger_trick Arg("kidger", CheckBox(true), required=false))
CLI arg: `--kidger`
"""

# ╔═╡ 2bb433bb-17df-4a34-9ccf-58c0cf8b4dd3
(sense, noise) = if backsolve
	(
		BacksolveAdjoint(autojacvec=ZygoteVJP(), checkpointing=true),
		function(seed, noise_size)
			rng_tree = Xoshiro(seed)
			VirtualBrownianTree(-3f0, fill(0f0, noise_size) |> Lux.gpu, tend=tspan_model[2]*2f0; tree_depth=tree_depth, rng=Threefry4x((rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32))))
		end,
	)
else
	(
		InterpolatingAdjoint(autojacvec=ZygoteVJP()),
		(seed, noise_size) -> nothing,
	)
end

# ╔═╡ db88cae4-cb25-4628-9298-5a694c4b29ef
println((context_size=context_size, hidden_size=hidden_size, latent_dims=latent_dims, data_dims=data_dims, batch_size=batch_size))

# ╔═╡ b8b2f4b5-e90c-4066-8dad-27e8dfa1d7c5
md"""
### Latent SDE Model
"""

# ╔═╡ 08759cda-2a2a-41ff-af94-5b1000c9e53f
solver = EulerHeun()

# ╔═╡ ec41b765-2f73-43a5-a575-c97a5a107c4e
println("Steps that will be derived: $(steps(tspan_model, dt))")

# ╔═╡ 63960546-2157-4a23-8578-ec27f27d5185
projector = if fixed_projector
	error("Fixed projector isn't implemented!!")
else
	Lux.Dense(latent_dims => data_dims)
end

# ╔═╡ 001c318e-b7a6-48a5-bfd5-6dd0368873ac
latent_sde = StandardLatentSDE(
	solver,
	tspan_model,
	steps(tspan_model, dt);
	data_dims=data_dims,
	latent_dims=latent_dims,
	prior_size=prior_size,
	posterior_size=hidden_size,
	diffusion_size=Int(floor(hidden_size/latent_dims)),
	depth=depth,
	rnn_size=context_size,
	context_size=context_size,
	hidden_activation=Lux.softplus,
	adaptive=false,
	# we only have this custom layer - the others are default
	projector=projector
)

# ╔═╡ 0f6f4520-576f-42d3-9126-2076a51a6e22
begin
	rng = Xoshiro()
end

# ╔═╡ 1938e122-2c05-46fc-b179-db38322530ff
md"""
# Parameters
"""

# ╔═╡ 05568880-f931-4394-b31e-922850203721
ps_, st_ = Lux.setup(rng, latent_sde)

# ╔═╡ b0692162-bdd2-4cb8-b99c-1ebd2177a3fd
ps = ComponentArray{Float32}(ps_) |> Lux.gpu

# ╔═╡ e6766502-06db-4045-af8e-9aee65a705da
st = st_ |> Lux.gpu

# ╔═╡ ee3d4a2e-0960-430e-921a-17d340af497c
md"""
Select a seed: $(@bind seed Scrubbable(481283))
"""

# ╔═╡ 3ab9a483-08f2-4767-8bd5-ae1375a62dbe
function plot_prior(priorsamples; rng=rng, tspan=latent_sde.tspan, datasize=latent_sde.datasize)
	prior = NeuralSDEExploration.sample_prior_dataspace(latent_sde, ps, st; seed=0, b=priorsamples, tspan=tspan |> Lux.gpu, datasize=datasize)
	return plot(prior, linewidth=.5,color=:black,legend=false,title="projected prior")
end

# ╔═╡ b5c6d43c-8252-4602-8232-b3d1b0bcee33
function plotmodel()
	n = 5
	rng_plot = Xoshiro(0)
	nums = sample(rng_plot, 1:length(timeseries.u), n; replace=false)
	
	posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = latent_sde(select_ts(nums, timeseries), ps, st, seed=seed) .|> Lux.cpu
	
	priorsamples = 25
	priornums = sample(rng_plot, 1:length(timeseries.u), priorsamples; replace=false)
	priorplot = plot_prior(priorsamples, rng=rng_plot)

	posteriorplot = plot(timeseries.t, posterior_data[1, :,:]', linewidth=2, legend=false, title="projected posterior")
	dataplot = plot(select_ts(nums, timeseries), linewidth=2, legend=false, title="data")
	
	timeseriesplot = plot(select_ts(priornums, timeseries), linewidth=.5, color=:black, legend=false, title="data")
	
	l = @layout [a b ; c d]
	p = plot(dataplot, posteriorplot, timeseriesplot, priorplot, layout=l)
	p
end

# ╔═╡ 225791b1-0ffc-48e2-8131-7f54848d8d83
md"""
# Training
"""

# ╔═╡ 550d8974-cd19-4d0b-9492-adb4e14a04b1
begin
	recorded_loss = []
	recorded_likelihood = []
	recorded_kl = []
	recorded_eta = []
	recorded_lr = []
end

# ╔═╡ fa43f63d-8293-43cc-b099-3b69dbbf4b6a
function plotlearning()
	plots = [
		plot(map(x -> max(1e-8, x+100.0), recorded_loss), legend=false, title="loss", yscale=:log10)
		plot(map(x -> max(1e-8, -x+100.0), recorded_likelihood), legend=false, title="-loglike", yscale=:log10)
		plot(recorded_kl, legend=false, title="kl-divergence")
		plot(recorded_eta, legend=false, title="beta")
		plot(recorded_lr, legend=false, title="learning rate")
	]	

	
	l = @layout [a ; b c; d e]
	plot(plots...; layout=l)
end

# ╔═╡ 24110995-82ce-4ba3-8307-6b6a5de88163
function loss(ps, minibatch, eta, seed)
	_, _, _, kl_divergence, likelihood = latent_sde(minibatch, ps, st; sense=sense, noise=noise, seed=seed, likelihood_scale=scale)
	return mean(-likelihood .+ (eta * kl_divergence)), mean(kl_divergence), mean(likelihood)
end


# ╔═╡ f4a16e34-669e-4c93-bd83-e3622a747a3a
function train(lr_sched, opt_state; kl_sched=Iterators.Stateful(Loop(x -> eta, 1)))
	num_samples = length(timeseries.u)
	indices = shuffle(1:num_samples)
	for iteration in 1:batch_size:num_samples - batch_size
		train_on = indices[iteration:iteration+batch_size-1]
		minibatch = select_ts(train_on, timeseries)
		
		seed = rand(rng, UInt32)

		eta = Float32(popfirst!(kl_sched))
		lr = Float32(popfirst!(lr_sched))

		l, kl_divergence, likelihood = loss(ps, minibatch, eta, seed)

		push!(recorded_loss, l)
		push!(recorded_kl, kl_divergence)
		push!(recorded_likelihood, likelihood)
		push!(recorded_eta, eta)
		push!(recorded_lr, lr)

		println("Loss: $l, KL: $kl_divergence")
		dps = Zygote.gradient(ps -> loss(ps, minibatch, eta, seed)[1], ps)
		
		#GC.gc(true)

		if kidger_trick
			dps[1].initial_prior *= length(timeseries.t)
			dps[1].initial_posterior *= length(timeseries.t)
		end
		
		Optimisers.update!(opt_state, ps, dps[1])
		Optimisers.adjust!(opt_state, lr)
	end
end

# ╔═╡ 9789decf-c384-42df-b7aa-3c2137a69a41
function exportresults(epoch)
	folder_name = if "SLURM_ARRAY_JOB_ID" in keys(ENV)
		ENV["SLURM_ARRAY_JOB_ID"] * "_" * ENV["SLURM_ARRAY_TASK_ID"]
	elseif "SLURM_JOB_ID" in keys(ENV)
		ENV["SLURM_JOB_ID"]
	else
		Dates.format(now(), "YMMddHHmm")
	end

	folder = homedir() * "/artifacts/$(folder_name)/"

	data = Dict(
		"latent_sde" => latent_sde,
		"timeseries" => timeseries,
		"ps" => ps,
		"st" => st,
		"model" => model,
		"initial_condition" => initial_condition,
		"args" => ARGS,
		"datamin" => datamin,
		"datamax" => datamax,
		"tspan_train" => tspan_train,
		"tspan_data" => tspan_data,
		"tspan_model" => tspan_model,
		"recorded_loss" => recorded_loss,
		"recorded_kl" => recorded_kl,
		"recorded_likelihood" => recorded_likelihood,
		"recorded_eta" => recorded_eta,
		"recorded_lr" => recorded_lr,
	)

	mkpath(folder)

	jldsave(folder * "$(epoch).jld"; data)
	
	modelfig = plotmodel()
	savefig(modelfig, folder * "$(epoch)_model.pdf")
	# savefig(modelfig, folder * "$(epoch)_model.tex")
	
	learningfig = plotlearning()
	savefig(learningfig, folder * "$(epoch)_learning.pdf")
	# savefig(learningfig, folder * "$(epoch)_learning.tex")
end

# ╔═╡ 124680b8-4140-4b98-9fd7-009cc225992a
loss(ps, select_ts(1:64, timeseries), 1f0, 4)[1]

# ╔═╡ 5123933d-0972-4fe3-9d65-556ecf81cf3c
ts = select_ts(1:64, timeseries)

# ╔═╡ 7a7e8e9b-ca89-4826-8a5c-fe51d96152ad
if enabletraining
	println("First Zygote call")
	@time loss(ps, ts, 1.0, 1)[1]
	@time dps = Zygote.gradient(ps -> loss(ps, ts, 1f0, 1)[1], ps)[1]
end

# ╔═╡ 67e5ae14-3062-4a93-9492-fc6e9861577f
kl_sched = if kl_anneal
	Iterators.Stateful(Loop(Sequence([Loop(x -> (eta*x)/kl_rate, kl_rate), Loop(x -> eta, kl_rate*3)], [kl_rate, kl_rate*3]), kl_rate*4))
else
	Iterators.Stateful(Loop(x -> eta, 1))
end

# ╔═╡ da2df05a-5d40-4293-98f0-abd20d6dcd2a
lr_sched = if lr_cycle
	Iterators.Stateful(CosAnneal(learning_rate, max_learning_rate, lr_rate))
else
	Iterators.Stateful(Exp(λ = learning_rate, γ = decay))
end

# ╔═╡ 78aa72e2-8188-441f-9910-1bc5525fda7a
begin
	if !(@isdefined PlutoRunner) && enabletraining  # running as job
		println("Starting training")
		opt_state_job = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipGrad(1.0), Optimisers.Adam()), ps)
		# precompile exportresults because there are some memory problems
		exportresults(0)
		for epoch in 1:1000
			train(lr_sched, opt_state_job; kl_sched=kl_sched)
			exportresults(epoch)
			GC.gc(true)
		end
	end
end

# ╔═╡ 830f7e7a-71d0-43c8-8e74-d1709b8a6707
function gifplot()
	p1 = plotmodel()
	p2 = plotlearning()
	plot(p1, p2, layout=@layout[a;b],size=(600,700))
end

# ╔═╡ 763f07e6-dd46-42d6-b57a-8f1994386302
gifplot()

# ╔═╡ 2b876f31-21c3-4782-a8a8-8da89d899719
# ╠═╡ disabled = true
#=╠═╡
if enabletraining
	opt_state_notebook = Optimisers.setup(Optimisers.Adam(), ps)
	@gif for epoch in 1:2
		train(lr_sched, opt_state_notebook; kl_sched=kl_sched)
		gifplot()
	end
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═67cb574d-7bd6-40d9-9dc3-d57f4226cc83
# ╠═db557c9a-24d6-4008-8225-4b8867ee93db
# ╠═b6abba94-db07-4095-98c9-443e31832e7d
# ╠═d1440209-78f7-4a9a-9101-a06ad2534e5d
# ╟─d38b3460-4c01-4bba-b726-150d207c020b
# ╟─13ef3cd9-7f58-459e-a659-abc35b550326
# ╟─ff15555b-b1b5-4b42-94a9-da77daa546d0
# ╟─c799a418-d85e-4f9b-af7a-ed667fab21b6
# ╟─0eec0598-7520-47ec-b13a-a7b9da550014
# ╟─cb3a270e-0f2a-4be3-9ab3-ea5e4c56d0e7
# ╟─95bdd676-d8df-4fef-bdd7-cce85b717018
# ╟─4c3b8784-368d-49c3-a875-c54960ec9be5
# ╟─a65a7405-d1de-4de5-9391-dcb971af0413
# ╟─e6a71aae-9d81-45a9-af9a-c4188dda2787
# ╟─71a38a66-dd66-4000-b664-fc3f04f6d4b8
# ╟─bae92f09-2e87-4a1e-aa2e-906f33985f6d
# ╟─bd7acf1a-c09a-4531-ad2c-b5e7e28af382
# ╟─42ece6c1-9e8a-45e7-adf4-6f353da6a4e5
# ╟─3665efa6-6527-4771-82fd-285c3c0f8b41
# ╟─10c206ef-2321-4c64-bdf4-7f4e9934d911
# ╟─fe7e2889-88de-49b3-b20b-342357596bfc
# ╟─de70d89a-275d-49d2-9da4-4470c869e56e
# ╟─986c442a-d02e-42d4-bda4-f66a1c92f799
# ╟─9a89a97c-da03-4887-ac8c-ef1f5264436e
# ╠═c441712f-e4b2-4f4a-83e1-aad558685288
# ╟─7e6256ef-6a0a-40cc-aa0a-c467b3a524c4
# ╠═2da6bbd4-8036-471c-b94e-10182cf8a834
# ╠═c00a97bf-5e10-4168-8d58-f4f9270258ac
# ╟─5691fcc5-29b3-4236-9154-59c6fede49ce
# ╠═15cef7cc-30b6-499d-b968-775b3251dedb
# ╟─1502612c-1489-4abf-8a8b-5b2d03a68cb1
# ╠═455263ef-2f94-4f3e-8401-f0da7fb3e493
# ╟─f4651b27-135e-45f1-8647-64ab08c2e8e8
# ╠═aff1c9d9-b29b-4b2c-b3f1-1f06a9370f64
# ╠═5d78e254-4134-4c2a-8092-03f6df7d5092
# ╠═9a5c942f-9e2d-4c6c-9cb1-b0dffd8050a0
# ╟─8280424c-b86f-49f5-a854-91e7abcf13ec
# ╟─fdceee23-b91e-4b2e-af78-776c01733de3
# ╟─97d724c2-24b0-415c-b90f-6a36e877e9d1
# ╟─d81ccb5f-de1c-4a01-93ce-3e7302caedc0
# ╟─6489b190-f08f-466c-93c4-92a723f8e594
# ╟─b5721107-7cf5-4da3-b22a-552e3d56bcfa
# ╟─efd438bc-13cc-457f-82c1-c6f0711079b3
# ╟─03a21651-9c95-49e8-bb07-b03640f7e5b7
# ╟─ad6247f6-6cb9-4a57-92d3-6328cbd84ecd
# ╟─60b5397d-7350-460b-9117-319dc127cc7e
# ╟─16c12354-5ab6-4c0e-833d-265642119ed2
# ╟─f12633b6-c770-439d-939f-c41b74a5c309
# ╟─3c630a3a-7714-41c7-8cc3-601cd6efbceb
# ╟─2961879e-cb52-4980-931b-6f8de1f26fa4
# ╟─7c23c32f-97bc-4c8d-ac54-42753be61345
# ╟─64e7bba4-fb17-4ed8-851f-de9204f0f42d
# ╟─33d53264-3c8f-4f63-9dd2-46ebd00f4e28
# ╟─8bb3084f-5fde-413e-b0fe-8b2e19673fae
# ╟─2c64b173-d4ad-477d-afde-5f3916e922ef
# ╟─9767a8ea-bdda-43fc-b636-8681d150d29f
# ╟─3db229f0-0e13-4d80-8680-58b89161db35
# ╟─cb1c2b2e-a2a2-45ed-9fc1-655d28f267d1
# ╟─7f219c33-b37b-480a-9d21-9ea8d898d5d5
# ╠═2bb433bb-17df-4a34-9ccf-58c0cf8b4dd3
# ╠═db88cae4-cb25-4628-9298-5a694c4b29ef
# ╠═6c0086c5-df79-4bc9-bada-f1c656525164
# ╟─b8b2f4b5-e90c-4066-8dad-27e8dfa1d7c5
# ╠═08759cda-2a2a-41ff-af94-5b1000c9e53f
# ╟─ec41b765-2f73-43a5-a575-c97a5a107c4e
# ╠═63960546-2157-4a23-8578-ec27f27d5185
# ╠═001c318e-b7a6-48a5-bfd5-6dd0368873ac
# ╠═0f6f4520-576f-42d3-9126-2076a51a6e22
# ╟─1938e122-2c05-46fc-b179-db38322530ff
# ╠═05568880-f931-4394-b31e-922850203721
# ╠═b0692162-bdd2-4cb8-b99c-1ebd2177a3fd
# ╠═e6766502-06db-4045-af8e-9aee65a705da
# ╟─ee3d4a2e-0960-430e-921a-17d340af497c
# ╠═3ab9a483-08f2-4767-8bd5-ae1375a62dbe
# ╠═b5c6d43c-8252-4602-8232-b3d1b0bcee33
# ╟─225791b1-0ffc-48e2-8131-7f54848d8d83
# ╠═550d8974-cd19-4d0b-9492-adb4e14a04b1
# ╠═fa43f63d-8293-43cc-b099-3b69dbbf4b6a
# ╠═24110995-82ce-4ba3-8307-6b6a5de88163
# ╠═f4a16e34-669e-4c93-bd83-e3622a747a3a
# ╠═9789decf-c384-42df-b7aa-3c2137a69a41
# ╠═124680b8-4140-4b98-9fd7-009cc225992a
# ╠═5123933d-0972-4fe3-9d65-556ecf81cf3c
# ╠═7a7e8e9b-ca89-4826-8a5c-fe51d96152ad
# ╠═67e5ae14-3062-4a93-9492-fc6e9861577f
# ╠═da2df05a-5d40-4293-98f0-abd20d6dcd2a
# ╠═78aa72e2-8188-441f-9910-1bc5525fda7a
# ╠═830f7e7a-71d0-43c8-8e74-d1709b8a6707
# ╠═763f07e6-dd46-42d6-b57a-8f1994386302
# ╠═2b876f31-21c3-4782-a8a8-8da89d899719
