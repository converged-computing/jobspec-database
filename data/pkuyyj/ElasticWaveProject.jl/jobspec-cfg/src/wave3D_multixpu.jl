# # Elastic Wave 3D
const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available

# ## ParallelStencil Initialization
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=false)
end
using ImplicitGlobalGrid, Plots, Printf, Statistics
default(size=(1400, 800), framestyle=:box, label=false, grid=false, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

import MPI

# ## Parallel Functions
@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, Sxx::Data.Array, Syy::Data.Array, Szz::Data.Array, Sxy::Data.Array, Sxz::Data.Array, Syz::Data.Array, dt::Data.Number, ρ::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @inn(Vx) = @inn(Vx) + dt/ρ * (@d_xi(Sxx)/dx + @d_yi(Sxy)/dy + @d_zi(Sxz)/dz)
    @inn(Vy) = @inn(Vy) + dt/ρ * (@d_xi(Sxy)/dx + @d_yi(Syy)/dy + @d_zi(Syz)/dz)
    @inn(Vz) = @inn(Vz) + dt/ρ * (@d_xi(Sxz)/dx + @d_yi(Syz)/dy + @d_zi(Szz)/dz)
    return
end

@parallel function compute_S!(Sxx::Data.Array, Syy::Data.Array, Szz::Data.Array, Sxy::Data.Array, Sxz::Data.Array, Syz::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dt::Data.Number, λ::Data.Number, μ::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number, damping_sxx::Data.Number)
    @all(Sxx) = damping_sxx * @all(Sxx) + dt * (λ * (@d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz) + 2μ * @d_xa(Vx)/dx)
    @all(Syy) = damping_sxx * @all(Syy) + dt * (λ * (@d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz) + 2μ * @d_ya(Vy)/dy)
    @all(Szz) = damping_sxx * @all(Szz) + dt * (λ * (@d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz) + 2μ * @d_za(Vz)/dz)
    return
end


@parallel_indices (ix, iy, iz) function compute_S_diag!(Sxx::Data.Array, Syy::Data.Array, Szz::Data.Array, Sxy::Data.Array, Sxz::Data.Array, Syz::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dt::Data.Number, λ::Data.Number, μ::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number, damping_sxy::Data.Number)
    if (ix <= size(Sxy, 1)-2 && iy <= size(Sxy, 2)-2 && iz <= size(Sxy, 3)-2)
        Sxy[ix+1, iy+1, iz] = damping_sxy * Sxy[ix+1, iy+1, iz] + dt * μ * ((Vy[ix+2, iy+1, iz] - Vy[ix, iy+1, iz])/(2*dx) + (Vx[ix+1, iy+2, iz] - Vx[ix+1, iy, iz])/(2*dy))
    end
    if (ix <= size(Sxz, 1)-2 && iy <= size(Sxz, 2)-2 && iz <= size(Sxz, 3)-2)
        Sxz[ix+1, iy, iz+1] = damping_sxy * Sxz[ix+1, iy, iz+1] + dt * μ * ((Vz[ix+2, iy, iz+1] - Vz[ix, iy, iz+1])/(2*dx) + (Vx[ix+1, iy, iz+2] - Vx[ix+1, iy, iz])/(2*dz))
    end
    if (ix <= size(Syz, 1)-2 && iy <= size(Syz, 2)-2 && iz <= size(Syz, 3)-2)
        Syz[ix, iy+1, iz+1] = damping_sxy * Syz[ix, iy+1, iz+1] + dt * μ * ((Vz[ix, iy+2, iz+1] - Vz[ix, iy, iz+1])/(2*dy) + (Vy[ix, iy+1, iz+2] - Vy[ix, iy+1, iz])/(2*dz))
    end
    return
end

function save_array(Aname,A)
    fname = string(Aname, ".bin")
    out = open(fname, "w"); write(out, A); close(out)
end

# ## Main function
"""

    Elastic 3D function on multiple CPU/GPU
    
    lx, nx      - defined as usual, length and grid size
    nt          - number of timesteps
    nout        - plotting frequency
    save_bin    - whether save 3D stress of every step in binary or not

"""
@views function elastic3D(;nx = 127, lx=120.0, nt=20000, nout=200, save_bin=false)
    ## Physics
    ly, lz     = lx, lx
    λ, μ       = 4.0, 2.0           # Lame parameters (λ, μ)
    k          = 1.0                # bulk modulus
    ρ          = 2.0                # density
    t          = 0.0                # physical time
    damping_sxx  = 1.0              # 1 - damping factor
    damping_sxy  = 0.95             # 1 - damping factor
    ## Numerics
    ny, nz     = nx, nx             # numerical grid resolution
    ## Derived numerics
    me, dims, nprocs, coords, comm = init_global_grid(nx, ny, nz) # MPI initialisation
    if USE_GPU
        select_device()             # select one GPU per MPI local rank (if >1 GPU per node)
    end                             
    dx, dy, dz = lx/(nx_g()-1), ly/(ny_g()-1), lz/(nz_g()-1)      # cell sizes
    ## Array allocations
    Vx         = @zeros(nx+1,ny  ,nz  )
    Vy         = @zeros(nx  ,ny+1,nz  )
    Vz         = @zeros(nx  ,ny  ,nz+1)
    Sxx        = @zeros(nx  ,ny  ,nz  )
    Syy        = @zeros(nx  ,ny  ,nz  )
    Szz        = @zeros(nx  ,ny  ,nz  )
    Sxy        = @zeros(nx  ,ny  ,nz  )
    Sxz        = @zeros(nx  ,ny  ,nz  )
    Syz        = @zeros(nx  ,ny  ,nz  )
    ## Initial conditions
    dt         = 0.05 * min(dx,dy,dz)/sqrt(k/ρ)/6.1 # prevent grad explosion

    ## Initialize diagonal stress tensor components, can be customized
    stress      = 500
    source_radius = 2
    Sxx .= Data.Array([stress * exp((-(x_g(ix,dx,Sxx)-0.2*lx)^2 - (y_g(iy,dy,Sxx)-0.5*ly)^2 - (z_g(iz,dz,Sxx)-0.3*lz)^2)/ (2 * source_radius^2)) for ix=1:size(Sxx,1), iy=1:size(Sxx,2), iz=1:size(Sxx,3)])
    Szz .= Data.Array([stress * exp((-(x_g(ix,dx,Szz)-0.2*lx)^2 - (y_g(iy,dy,Szz)-0.5*ly)^2 - (z_g(iz,dz,Szz)-0.3*lz)^2)/ (2 * source_radius^2)) for ix=1:size(Szz,1), iy=1:size(Szz,2), iz=1:size(Szz,3)])

    ## Preparation of visualisation
    ENV["GKSwstype"]="nul"
    if (me==0)
        loadpath = "./viz3D_out/"
        if isdir(loadpath)==false mkdir(loadpath) end;  anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
    end
    nx_v, ny_v, nz_v = (nx-2)*dims[1], (ny-2)*dims[2], (nz-2)*dims[3]
    if (nx_v*ny_v*nz_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
    ## Gather data for visualization
    Sxx_v = zeros(nx_v, ny_v, nz_v)
    Syy_v = zeros(nx_v, ny_v, nz_v)
    Szz_v = zeros(nx_v, ny_v, nz_v)
    Sxy_v = zeros(nx_v, ny_v, nz_v)
    Syz_v = zeros(nx_v, ny_v, nz_v)
    Sxz_v = zeros(nx_v, ny_v, nz_v)
    P_inn = zeros(nx-2, ny-2, nz-2) # no halo local array for visu
    ## Define inner arrays for stress tensor components
    Sxx_inn = zeros(nx-2, ny-2, nz-2)
    Syy_inn = zeros(nx-2, ny-2, nz-2)
    Szz_inn = zeros(nx-2, ny-2, nz-2)
    Sxy_inn = zeros(nx-2, ny-2, nz-2)
    Sxz_inn = zeros(nx-2, ny-2, nz-2)
    Syz_inn = zeros(nx-2, ny-2, nz-2)

    y_sl  = Int(ceil(ny_g()*0.5))
    z_sl  = Int(ceil(nz_g()*0.3))
    Xi_g  = dx:dx:(lx-dx) # inner points only
    Yi_g  = dy:dy:(ly-dy)
    Zi_g  = dz:dz:(lz-dz)
    ## Time loop
    iframe = 0
    for it = 1:nt
        if (it==11) tic() end
        @hide_communication (8, 8, 4) begin # communication/computation overlap
            @parallel compute_V!(Vx, Vy, Vz, Sxx,Syy,Szz,Sxy,Sxz,Syz, dt, ρ, dx, dy, dz)
            update_halo!(Vx, Vy, Vz)
        end
        @hide_communication (8, 8, 4) begin
            @parallel compute_S!(Sxx,Syy,Szz,Sxy,Sxz,Syz, Vx, Vy, Vz, dt, λ, μ, dx, dy, dz, damping_sxx)
            update_halo!(Sxx,Syy,Szz)
        end
        @hide_communication (8, 8, 4) begin
            @parallel compute_S_diag!(Sxx,Syy,Szz,Sxy,Sxz,Syz, Vx, Vy, Vz, dt, λ, μ, dx, dy, dz, damping_sxy)
            update_halo!(Sxy,Sxz,Syz)
        end
        t = t + dt

        ## Visualisation
        if mod(it,nout)==0
            ## Assign the inner part of each global array to the corresponding inner array
            Sxx_inn .= Array(Sxx[2:end-1, 2:end-1, 2:end-1])
            Syy_inn .= Array(Syy[2:end-1, 2:end-1, 2:end-1])
            Szz_inn .= Array(Szz[2:end-1, 2:end-1, 2:end-1])
            Sxy_inn .= Array(Sxy[2:end-1, 2:end-1, 2:end-1])
            Sxz_inn .= Array(Sxz[2:end-1, 2:end-1, 2:end-1])
            Syz_inn .= Array(Syz[2:end-1, 2:end-1, 2:end-1])

            gather!(Sxx_inn, Sxx_v)
            gather!(Syy_inn, Syy_v)
            gather!(Szz_inn, Szz_v)
            gather!(Sxy_inn, Sxy_v)
            gather!(Syz_inn, Syz_v)
            gather!(Sxz_inn, Sxz_v)

            ## Compute von Mises stress
            von_Mises_stress = sqrt.(0.5 .* ((Sxx_v - Syy_v).^2 + (Syy_v - Szz_v).^2 + (Szz_v - Sxx_v).^2) .+ 6 .* (Sxy_v.^2 + Syz_v.^2 + Sxz_v.^2))
            ## Compute sum of squared stresses
            sum_squared_stresses = Sxx_v.^2 + Syy_v.^2 + Szz_v.^2 + Sxy_v.^2 + Syz_v.^2 + Sxz_v.^2
            
            ## Binary stress saved can be modified
            stress = Sxx_v 
            ## stress = sum_squared_stresses

            ## Plot six plots to visualize in detail
            if me == 0
                if iframe < 150
                    p1=heatmap(Xi_g, Zi_g, stress[:, y_sl, :]', aspect_ratio=1, c=:viridis, title="Sxx on x-z at t=$t")
                    p2=heatmap(Xi_g, Yi_g, stress[:, :, z_sl]', aspect_ratio=1, c=:viridis, title="Sxx on x-y")
                    p3=heatmap(Xi_g, Zi_g, sum_squared_stresses[:, y_sl, :]', aspect_ratio=1, c=:viridis, title="Square Stress on x-z ")
                    p4=heatmap(Xi_g, Yi_g, sum_squared_stresses[:, :, z_sl]', aspect_ratio=1, c=:viridis, title="Square Stress on x-y")
                    p5=heatmap(Xi_g, Zi_g, von_Mises_stress[:, y_sl, :]', aspect_ratio=1, c=:viridis, title="von_Mises_stress on x-z")
                    p6=heatmap(Xi_g, Yi_g, von_Mises_stress[:, :, z_sl]', aspect_ratio=1, c=:viridis, title="von_Mises_stress on x-y")
                    plot(p1, p2, p3, p4, p5, p6,layout=(3, 2), size=(1800, 2000))
                    frame(anim)
                end

                if save_bin
                    save_array(@sprintf("%sout_Stress_%04d", loadpath, iframe+=1), convert.(Float32, Array(stress)))
                end
            end
        end
    end
    ## Performance
    wtime    = toc()
    A_eff    = ((3+6)*2)/1e9*nx*ny*nz*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nt-10)                           ## Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                          ## Effective memory throughput [GB/s]
    if (me==0) @printf("Total steps=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nt, wtime, round(T_eff, sigdigits=2)) end
    if (me==0) gif(anim, @sprintf("%swave3D.gif", loadpath), fps = 25) end
    finalize_global_grid()
    return Array(Sxx)

end

# ## Calculation
"""
    To run calculation, please use:
    elastic3D()
"""

