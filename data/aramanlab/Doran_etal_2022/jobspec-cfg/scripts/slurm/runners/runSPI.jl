using DrWatson
@quickactivate "Doran_etal_2022"
using SPI, SparseArrays 
using ArgMacros, Logging, TimerOutputs
using StatsBase: sample
using Gotree_jll

include(joinpath(srcdir(), "parsephylip.jl"))

@structarguments false Args begin
    @argumentrequired String inputfile "-i" "--inputfile"
    @argtest inputfile isfile "Couldn't find the input file."
    @argtest inputfile (f)->(split(f, ".")[end] ∈ ["txt", "phy", "phylip"]) "extension not recognized; should be .phy .phylip or .txt"
    @argumentrequired String outputdir "-o" "--outputdir"
    @argumentdefault Int 100 nboot "-b" "--nboot"
    @argtest nboot n->0≤n "nboot must be positive"
    @argumentdefault Int 0 loglevel "-l" "--loglevel"
    @argumentdefault Symbol :notneeded model "-m" "--model"
end

function julia_main()::Cint
    # parse arguments
    args = Args()
    logger = ConsoleLogger(stdout, LogLevel(args.loglevel))
    global_logger(logger)
    time = TimerOutput()
    @timeit time "total" begin

    @info "Starting SPI inference"
    @info "Setting up workspace"
    # setup output dir
    mkpath(args.outputdir) != nothing || 
        ErrorException("Could not create outputdir: $(args.outputdir)")
    name = first(split(basename(args.inputfile), "."))

    @info "Running SPI" 
    @timeit time "running SPI" begin
        phydf = readphylip(args.inputfile)
        nfeats = length(phydf.seqs[1])
        M = onehotencode(phydf.seqs)
        vals, vecs = eigen(Matrix(M*M'))
        dij = calc_spi_mtx(vecs, sqrt.(max.(vals, zero(eltype(vals)))))
        dij = dij ./ nfeats
        hc = hclust(dij, linkage=:average, branchorder=:optimal)
        spitree = nwstr(hc, phydf.ids; labelinternalnodes=false)
    end #time
    
    @info "Writing out SPI Tree"
    open(joinpath(args.outputdir, name * "-tree.nw"), "w") do io
        println(io, spitree)
    end

    # Bootstrap
    if args.nboot > 0
        @info "Starting Bootstrap with $(args.nboot)"
        @timeit time "running bootstrap SPI" begin
            boottrees = Vector{String}(undef, args.nboot)
            chardf = _stringcolumntocharmtx(phydf.seqs)
            Threads.@threads for i in 1:args.nboot
                nchars = size(chardf, 2)
                colsmps = sample(1:nchars, nchars, replace=true)
                tmpM = onehotencode(chardf[:,colsmps])
                vals, vecs = eigen(Matrix(tmpM * tmpM'))
                dij = calc_spi_mtx(vecs, sqrt.(max.(vals, zero(eltype(vals)))))
                dij = dij ./ nchars
                hc = hclust(dij, linkage=:average, branchorder=:optimal)
                boottrees[i] = nwstr(hc, phydf.ids; labelinternalnodes=false)
            end
        end # time
        @info "Writing out Bootstrap trees"
        ## write out SPI boot trees
        open(joinpath(args.outputdir, name * "-boottrees.nw"), "w") do io
            for btree in boottrees
                println(io, btree)
            end
        end

        @info "using Booster to compute support values"
        ## calculate support
        run(pipeline(`$(gotree()) compute support tbe --silent \
            -i $(joinpath(args.outputdir, name * "-tree.nw")) \
            -b $(joinpath(args.outputdir, name * "-boottrees.nw")) \
            -o $(joinpath(args.outputdir, name * "-supporttree.nw"))`,
            stderr=joinpath(args.outputdir, "booster.log")))
    end 
    end # function timeit
    @info "Finishing run"
    @info "\ntiming" show(time) println("")
    return 0
end

julia_main()