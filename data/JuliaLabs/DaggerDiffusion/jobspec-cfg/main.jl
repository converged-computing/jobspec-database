import MPI
using MPIClusterManagers
using Dagger
using CairoMakie

using Diffusion

nx=64          # Number of grid points in x dimension
ny=64          # Number of grid points in y dimension
timesteps=30   # Number of time steps

# Notes:
# Producers can run ahead of consumer, by 2000 time steps.
# Makie has high initial latency

function process_video(dims, nx, ny, timesteps)
    nx_v = (nx-2)*dims[1]
    ny_v = (ny-2)*dims[2]
    T    = zeros(nx_v, ny_v)
    node = Observable(T)
    fig  = heatmap(node, colorrange = (0.0, 2.0))

    record(fig, "output.mkv", 1:timesteps) do t
        if t % 2 == 0
            @info "Processing frame" t
            # Process frames from workers
            results = fetch.([Dagger.@spawn single=w Diffusion.get_result() for w in workers()])
            for (coords, result) in results
                cart_x = (1:(nx-2)) .+ (coords[1] * (nx-2))
                cart_y = (1:(ny-2)) .+ (coords[2] * (ny-2))
                T[cart_x,cart_y] .= result
            end
            node[] = T
        end
    end
end 

manager = MPIClusterManagers.start_main_loop(MPI_TRANSPORT_ALL) # does not return on worker
try
    @mpi_do manager begin
        using MPI, Distributed
        wcomm = MPI.COMM_WORLD
        id = myid()
        rank = id-1
        temp_comm = MPI.Comm_dup(wcomm)
        gridcomm = MPI.Comm_split(temp_comm, rank == 0 ? MPI.MPI_UNDEFINED[] : 1, rank)
        if id != 1
            @assert MPI.Comm_size(gridcomm) == MPI.Comm_size(wcomm)-1
            gridsize = MPI.Comm_size(gridcomm)
        end

        # Create Cartesian process topology
        if id == 1
            gridsize = MPI.Comm_size(wcomm)-1
            if gridsize == 0
                println("No other processes available, start this program with at least 2 processes")
                exit(1)
            end
            print("Size: $gridsize\n")

            dims = [0,0]
            MPI.Dims_create!(gridsize, dims)
            try
                process_video(dims, nx, ny, timesteps)
            finally
                @info "Terminating workers"
                for w in workers()
                    Dagger.@spawn single=w Diffusion.terminate()
                end
                @info "Terminated workers"
            end
        else
            nprocs = MPI.Comm_size(gridcomm)
            dims = [0,0]
            MPI.Dims_create!(gridsize, dims)
            comm_cart = MPI.Cart_create(gridcomm, dims, #=periods=#[0,0], #=reorder=#0)
            coords = MPI.Cart_coords(comm_cart)

            # Initialize the heat capacity field
            T = ones(Float64, nx, ny)

            # Simulate heat diffusion
            print("Starting diffusion on rank $coords\n")
            Diffusion.diffusion(T, timesteps, comm_cart)
        end
    end
finally
    # Exit gracefully
    MPIClusterManagers.stop_main_loop(manager)
end
