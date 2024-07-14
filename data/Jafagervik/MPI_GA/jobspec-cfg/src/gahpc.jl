using MPI, MPISort
using Random
using StaticArrays

MPI.Init()

const ROOT = 0
const MAX_POP = 10
const IND_SIZE = 6
const MAX_GENERATIONS = 20

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

"""
    Macro to shorten check of root rank
"""
macro isroot()
    quote
        ROOT == rank
    end
end


const Individual = SVector{IND_SIZE,Bool}

create_individual(data::String)::Individual = SVector(map(x -> x == '1', collect(data)))


# These ones should be ran serially on each process
# It's mainly splitting the population and those algorithms that needs 
# to be considered
size(bs::Individual) = length(bs)
ones(bs::Individual) = count(c -> c, bs)
zeros(bs::Individual) = count(c -> !c, bs)

flip_at!(bs::Individual, idx::Integer) = bs.data[idx] = !bs.data[idx]
flip_all(bs::Individual) = ~bs

function flip_random_n!(bs::Individual, n::Integer)
    idxs = rand(1:IND_SIZE, n)

    for i in idxs
        flip_at!(bs, i)
    end
end

struct Population
    pop::Vector{Individual}
    size::Int64
end

Population(pop::Vector{Individual}, size::Int64) = new(pop, size)


# NOTE: requires population to be sorted
best(p::Population) = ones(first(p))

function generate_random_population(local_n::Integer)::Vector{Individual}
    local_population = Vector{Individual}(undef, local_n)

    for i in 1:local_n
        individual_string = join(rand(["0", "1"]) for _ in 1:IND_SIZE)
        local_population[i] = create_individual(individual_string)
    end

    return local_population
end


################################
#     Selection - note, more of a cutoff for now
################################

"""
    Get the best parents
    Mate these 
    return the new children
    sort
"""
function naive_selection!(p::Population, n::Integer)
    # TODO: Parallelize
    sort!(p, by=ind -> zeros(ind))
    p = p[1:n]
end


################################
#    Crossover 
################################

"""
    Scatter the parents on multiple processes
    and let each of them crossover to create children.
    Then gather these local_children into a global_children
    array.
"""
function crossover(local_parents::Vector{Individual})::Vector{Individual}
    local_children = Vector(undef, length(local_parents))
    half = Int(floor(IND_SIZE * 0.5))

    # TODO: Parallelize?
    for i in 1:length(local_parents)/2
        p1 = rand(local_parents)
        p2 = rand(local_parents)

        a = join([b ? "1" : "0" for b in local_parents[p1].data[1:half]])
        b = join([b ? "1" : "0" for b in local_parents[p2].data[half+1:IND_SIZE]])

        c = join([b ? "1" : "0" for b in local_parents[p2].data[1:half]])
        d = join([b ? "1" : "0" for b in local_parents[p1].data[half+1:IND_SIZE]])

        c1 = Individual(a * b)
        c2 = Individual(c * d)

        local_children[i] = c1
        local_children[i+1] = c2
    end

    return local_children
end


################################
#   Mutation 
################################
function mutate!(children::Vector{Individual}, α=0.01, n=2)::Vector{Individual}
    n_to_mutate = Int(α * length(children))
    individuals = rand(1:IND_SIZE, n_to_mutate)

    # TODO: Parallelize?
    for ind in individuals
        children[ind] = flip_random_n(ind, n)
    end
end


################################
#  Evaluation function 
################################
found_ideal(p::Population) = ones(first(p.pop)) == IND_SIZE



function main()
    MAX_POP % nprocs == 0 || error("Not divisible")

    local_n::Integer = MAX_POP / nprocs

    local_population = generate_random_population(local_n)

    if rank == ROOT
        population = Vector{Individual}(undef, MAX_POP)
        recv_counts = Vector{Int32}(nprocs)
    else
        population = Individual[]
        recv_counts = Vector{Int32}()
    end

    # @show rank local_population

    MPI.Gatherv!(local_population, population, recv_counts, ROOT, comm)

    if rank == ROOT
        @show population
    end


    MPI.Barrier(comm)

    if rank == ROOT
        # TODO: Sort populaiton in parallel
        sort!(population.pop, by=ind -> zeros(ind))

        # ys should contain number of 0s away
        ys = Vector{Int32}(undef, MAX_GENERATIONS)
    else
        ys = Float64[]()
    end

    gen = 1



    while gen < MAX_GENERATIONS
        if rank == ROOT
            ys[gen] = zeros(first(population.pop))

            # Check if we've found the best
            # FIXME: If one breaks than every process needs to break!
            found_ideal(population) && break

            # get parents sort of a selection
            parents::Vector{Individual} = deepcopy(population.pop[1:MAX_POP/2])
        else
            parents = Individual[]()
        end

        # TODO: Add wait?


        if rank == ROOT
            global_children = Vector{Individual}(undef, MAX_POP / 2)
        else
            global_children = Individual[]()
        end

        # TODO: Split parents up in local_parents

        # Next steps goes on on ever process

        # Mating
        children::Vector{Individual} = crossover(parents)

        # Mutate children
        mutate!(children, α=0.2, n=2)


        MPI.Gather!(children, global_children, ROOT, comm)

        MPI.waitall()


        # Add children to population
        if rank == ROOT
            append!(population, global_children)
        end

        # On root process, sort the individuals and cap of

        @isroot && naive_selection!(population, MAX_POP)

        MPI.Barrier()

        gen += 1
    end

    @isroot && println("Found best element in $gen generations.")


    MPI.Finalize()
end


Random.seed!(42)
main()
