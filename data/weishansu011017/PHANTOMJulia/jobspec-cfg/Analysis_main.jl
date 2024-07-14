include("./src/PHANTOMJulia.jl")
using .PHANTOMJulia

"""
The PHANTOMJulia Ananlysis --- for analyzing the PHANTOM result.
    Made by Wei-Shan Su, 2024
"""

function main_analysis(directory::String, file::String)
    # General setting
    Smoothed_kernel_function :: Function = M6_spline                     # Allowed function: M4_spline, M5_spline, M6_spline, C2_Wendland, C4_Wendland, C6_Wendland
    h_mode :: String = "mean"                                            # Allowed mode: "mean", "closest", "intep"
    Rotate_xy_plane :: Bool = false                                      # Rotate the whole coordinate to the coordinate with z' axis paralleling to the direction of angular_momentum_vector of primary disc
    Output_type::String = "H5DF"                                         # Datatype of the ouput files. Allowed datatype: "dat", "H5DF"

    # Spiral analysis setting
    Spiral_Analysis = true
    radius_sp :: Float64 = 75.0                                          # Radius for analysis (au)
    n_theta_sp :: Int = 300                                              # Number of seperation in theta
    key_values_sp = ["Sigma","tilt"]                                     # The physical quantities that should be written in the output file.

    # Pitch analysis setting
    Pitch_Analysis = true
    rmin_p :: Float64 = 10.0                                             # The minimum radius for analysis (au), also used for determining the range of primary disc.
    rmax_p :: Float64 = 100.0                                            # The maximum radius for analysis (au), also used for determining the range of primary disc.
    n_radius_p :: Int = 181                                              # Number of seperation in radius
    n_theta_p :: Int = 301                                               # Number of seperation in theta
    n_zsep_p :: Int = 3                                                  # Number of seperation in z while calculating the midplane
    mid_frac :: Float64 = 0.3                                            # Fraction of midplane in the gaseous scale height i.e. H_mid = mid_frac * H_g 
    key_values_p = ["Sigma","rho","vr","vtheta","e","∇Sigmax","∇Sigmay"] # The physical quantities that should be written in the output file.

    # Setup info
    filepath = directory * "/" * file
    initial_logging(get_analysis_info(directory,file))

    # Read file
    pjdf_list = read_phantom(filepath,"all")

    # Get time (Original for making the time.dat but haven't done yet.)
    time = pjdf_list[end].params["time"]

    # Move the coordinate to primary star
    COM2primary(pjdf_list, pjdf_list[end],1)

    # Rotate the coordinate
    if Rotate_xy_plane
        rotate_to_primary_L(pjdf_list,rmin_p,rmax_p)
    end

    # Add necessary quantity for each particles for SPH intepolation.
    for i in 1:(length(pjdf_list) - 1)
        add_norm(pjdf_list[i])
        add_eccentricity(pjdf_list[i],pjdf_list[end],1.0)
        add_cylindrical(pjdf_list[i])
        add_tilt(pjdf_list[i],rmin_p,rmax_p)
        add_rho(pjdf_list[i])
    end
    if Spiral_Analysis
        # Start spiral analysis
        Gas_spiral_result = spiral_analysis(pjdf_list[1],radius_sp, n_theta_sp, Smoothed_kernel_function, h_mode)
        Dust_spiral_result = spiral_analysis(pjdf_list[2],radius_sp, n_theta_sp, Smoothed_kernel_function, h_mode)

        # Extract Spiral_analysis
        spiral_buffer = Spiral_analysis_buffer(time, radius_sp, Gas_spiral_result, Dust_spiral_result, key_values_sp)
        spiral = buffer2output(spiral_buffer)
        Write_output(file,spiral,"Spiral",Output_type)

        # Release allocation
        Gas_spiral_result = nothing
        Dust_spiral_result = nothing
        spiral_buffer = nothing
        spiral = nothing
    end

    if Pitch_Analysis
        # Get scale height grid
        midH_array = midH_analysis(pjdf_list[1],(rmin_p,rmax_p,n_radius_p),n_theta_p,mid_frac,Smoothed_kernel_function,h_mode)

        # Start pitch_analysis
        Gas_pitch_result = pitch_analysis(pjdf_list[1],(rmin_p,rmax_p,n_radius_p),n_theta_p, midH_array , n_zsep_p ,["vr","vtheta"], Smoothed_kernel_function,h_mode)
        Dust_pitch_result = pitch_analysis(pjdf_list[2],(rmin_p,rmax_p,n_radius_p),n_theta_p, midH_array , n_zsep_p ,["vr","vtheta"], Smoothed_kernel_function,h_mode)

        # 2D intepolate
        Gas_pitch_other_result = pitch_other_analysis(pjdf_list[1],(rmin_p,rmax_p,n_radius_p),n_theta_p,["e"], Smoothed_kernel_function,h_mode)
        Dust_pitch_other_result = pitch_other_analysis(pjdf_list[2],(rmin_p,rmax_p,n_radius_p),n_theta_p,["e"], Smoothed_kernel_function,h_mode)

        #Put Sigma back to result
        for column_name in keys(Gas_pitch_other_result)
            Gas_pitch_result[column_name] = Gas_pitch_other_result[column_name]
            Dust_pitch_result[column_name] = Dust_pitch_other_result[column_name]
        end
        

        # Extract Pitch_analysis
        pitch_buffer = Pitch_analysis_buffer(time, Gas_pitch_result, Dust_pitch_result, key_values_p)
        pitch = buffer2output(pitch_buffer)
        Write_output(file,pitch,"Pitch",Output_type)

        # Release allocation
        midH_array = nothing
        Sigma_g_grid = nothing
        Sigma_d_grid = nothing
        Gas_pitch_result = nothing
        Dust_pitch_result = nothing
        pitch_buffer = nothing
        pitch = nothing
        pjdf_list = nothing
    end

    @info "-------------------------------------------------------"
end

function main()
    # Commendline variable setting
    if length(ARGS) < 1
        println("Usage: julia Analysis_main.jl <filename>")
        exit(1)
    end

    directory = pwd()                                                    # The filepath of analyzed file should be written as the relative path of script.
    files = ARGS             

    First_logging()

    for file in files
        println("File: $file")
        @time_and_print begin
            main_analysis(directory,file)
        end 
    end

    @info "\nEnd analysis!"
end

main()


