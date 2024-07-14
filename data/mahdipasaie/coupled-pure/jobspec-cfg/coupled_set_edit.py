import fenics as fe
import numpy as np
from mpi4py import MPI
from tqdm import tqdm
from modad_edited import refine_mesh
from ns_edited import update_solver_on_new_mesh_ns 
from pf_edited import update_solver_on_new_mesh_pf
import time

start_time = time.time()

fe.set_log_level(fe.LogLevel.ERROR)
#################### Define Parallel Variables ####################
# Get the global communicator
comm = MPI.COMM_WORLD
# Get the rank of the process
rank = comm.Get_rank()
# Get the size of the communicator (total number of processes)
size = comm.Get_size()
#############################  END  ################################
def refine_mesh_local(mesh, rad, center, max_level): 
    xc, yc = center
    mesh_itr = mesh

    for _ in range(max_level):
        mf = fe.MeshFunction("bool", mesh_itr, mesh_itr.topology().dim(), False)
        for cell in fe.cells(mesh_itr):
            x, y, *_ = cell.midpoint().array()  # Get cell midpoint coordinates
            if (x - xc)**2 + (y - yc)**2 < 2 * rad**2:
                mf[cell] = True  # Mark cell for refinement

        mesh_itr = fe.refine(mesh_itr, mf)  # Refine the mesh

    return mesh_itr

# rounding the domain number had effect on BC!

physical_parameters_dict = {
    "dy": 0.4 ,
    "max_level": 3,
    "Nx": 250,
    "Ny": 250,
    "dt": 8E-3,
    "dy_coarse":lambda max_level, dy: 2**max_level * dy,
    "Domain": lambda Nx, Ny: [(0.0, 0.0), (Nx, Ny)],
    "seed_center": [250/2, 250/2],
    "a1": 0.8839,
    "a2": 0.6637,
    "w0": 1,
    "tau_0": 1,
    "d0": lambda w0: w0 * 0.139,
    "W0_scale": 25.888e-8,
    "tau_0_scale": 1.6381419166815996e-6,
    "ep_4": 0.05,
    "k_eq": 0.14,
    "lamda": lambda a1: 6.383,
    "D": lambda a2, lamda: a2 * lamda,
    # "at": 1 / (2 * fe.sqrt(2.0)),
    "opk": lambda k_eq: 1 + k_eq,
    "omk": lambda k_eq: 1 - k_eq,
    # "c_initial": 4,
    "u_initial": -0.55, 
    "initial_seed_radius": 8.2663782447466,
    ####################### Navier-Stokes Parameters ####################
    "gravity": lambda tau_0_scale, W0_scale: -10 * (tau_0_scale**2) / (W0_scale )  ,
    "rho_liquid": 2.45,
    "rho_solid": 2.7, # Kg/m^3s
    "mu_fluid": lambda tau_0_scale, W0_scale: 14E-4 * (tau_0_scale) / (W0_scale ** 2), # scaled here coeff just correct for dynamic (14E-4 Pa.s)
    "alpha_c": 9.2e-3,
    "viscosity_solid": lambda mu_fluid: mu_fluid *1E6 , # 1E3 changed to 1E6
    "viscosity_liquid": lambda mu_fluid: mu_fluid,
    "lid_vel_x": 0.0, 
    "lid_vel_y": -1.0,
    ###################### SOLVER PARAMETERS ######################
    "abs_tol_pf": 1E-6,  
    "rel_tol_pf": 1E-5,  
    "abs_tol_ns": 1E-6,  
    "rel_tol_ns": 1E-5,  
    'linear_solver_ns': 'mumps', 
    'nonlinear_solver_ns': 'snes',      # "newton" , 'snes'
    "preconditioner_ns": 'ilu',  
    'maximum_iterations_ns': 50, 
    'nonlinear_solver_pf': 'snes',     # "newton" , 'snes'
    'linear_solver_pf': 'mumps',       # "mumps" , "superlu_dist", 'cg', 'gmres', 'bicgstab'
    "preconditioner_pf": 'ilu',       # 'hypre_amg', 'ilu', 'jacobi'
    'maximum_iterations_pf': 50,
    ####################### Rfinement Parameters ####################
    "precentile_threshold_of_high_gradient_velocities": 98,
    "precentile_threshold_of_high_gradient_pressures": 98,
    "precentile_threshold_of_high_gradient_U": 98,
    "interface_threshold_gradient": 0.0001,
}




##################################### Defining the mesh ###############################
dy = physical_parameters_dict["dy"]
dt = physical_parameters_dict["dt"]
initial_seed_radius = physical_parameters_dict["initial_seed_radius"]
seed_center = physical_parameters_dict["seed_center"]
max_level= physical_parameters_dict["max_level"]
Nx = physical_parameters_dict["Nx"]
Ny= physical_parameters_dict["Ny"]
# Calculated values
dy_coarse = physical_parameters_dict["dy_coarse"](max_level, dy)
Domain = physical_parameters_dict["Domain"](Nx, Ny)
# Create the mesh
nx = (int)(Nx/ dy ) 
ny = (int)(Ny / dy ) 
nx_coarse = (int)(Nx/ dy_coarse ) 
ny_coarse = (int)(Ny / dy_coarse ) 
coarse_mesh = fe.RectangleMesh( fe.Point(0.0 , 0.0 ), fe.Point(Nx, Ny), nx_coarse, ny_coarse  )
Domain = [ (0.0, 0.0 ) , ( Nx, Ny ) ]
mesh = refine_mesh_local(coarse_mesh , initial_seed_radius , seed_center , max_level)
#############################  END  ################################

def write_simulation_data_to_single_file(solution_vectors, times, file, variable_names_list, extra_funcs_dict):

    # Configure file parameters
    file.parameters["rewrite_function_mesh"] = True
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True

    for Sol_Func, time, variable_names in zip(solution_vectors, times, variable_names_list):
        # Split the combined function into its components
        functions = Sol_Func.split(deepcopy=True)

        # Check if the number of variable names matches the number of functions
        if variable_names and len(variable_names) != len(functions):
            raise ValueError("The number of variable names must match the number of functions.")

        # Rename and write each function to the file
        for i, func in enumerate(functions):
            name = variable_names[i] if variable_names else f"Variable_{i}"
            func.rename(name, "solution")
            file.write(func, time)

    # Write extra functions (like viscosity, velocity_PF) if provided
    for name, func in extra_funcs_dict.items():
        func.rename(name, "solution")
        file.write(func, times[-1])  # Assuming extra functions correspond to the last time point

    file.close()

def update_time_step(physical_parameters_dict, solver_pf_information, solver_ns_information, u_max, u_min): 

    dt = physical_parameters_dict["dt"]
    D = physical_parameters_dict

    number_of_iterations_pf = solver_pf_information[0]
    number_of_iterations_ns = solver_ns_information[0]
    convergence_pf = solver_pf_information[1]
    convergence_ns = solver_ns_information[1]

    dy = physical_parameters_dict["dy"]
    D = physical_parameters_dict["D"](physical_parameters_dict["a2"], physical_parameters_dict["lamda"](physical_parameters_dict["a1"]))
    CFL_max = 0.5  # Example maximum CFL number for stability
    dt_CFL = CFL_max * dy / u_max

    dt_diffusive = dy ** 2 / (2 * D)  # Factor of 2 for safety
    # Choose the smaller of the two for safety
    dt_cfl_combined = min(dt_CFL, dt_diffusive)

    print("number_of_iterations_pf: ", number_of_iterations_pf, flush=True)
    print("number_of_iterations_ns: ", number_of_iterations_ns, flush=True)


    if number_of_iterations_pf < 5 and number_of_iterations_ns <5 and dt< 100 * dt_cfl_combined :

        dt = 2 * dt

    elif not convergence_pf or not convergence_ns:
        # Reduce the time step if the solution did not converge
        dt = 0.5 * dt

    if dt < 1E-8:
        raise ValueError("Time step is too small.")
    elif dt > 1E+3:
        raise ValueError("Time step is too large.")


    return dt

# Usage Example:
file = fe.XDMFFile( "Test5.xdmf" )


##############################################################
old_solution_vector_ns = None
old_solution_vector_pf = None
old_solution_vector_0_ns = None
old_solution_vector_0_pf = None
##############################################################
# Initializw the problem: 

ns_problem_dict = update_solver_on_new_mesh_ns(mesh, physical_parameters_dict, old_solution_vector_ns= None, old_solution_vector_0_ns=None, 
                            old_solution_vector_0_pf = None , variables_dict= None )

pf_problem_dict = update_solver_on_new_mesh_pf(mesh, physical_parameters_dict, old_solution_vector_pf= None, old_solution_vector_0_pf=None, 
                                old_solution_vector_0_ns=None, variables_dict_pf= None)

# variables for solving the problem ns
solver_ns = ns_problem_dict["solver_ns"]
solution_vector_ns = ns_problem_dict["solution_vector_ns"]
solution_vector_ns_0 = ns_problem_dict["solution_vector_ns_0"]
space_ns = ns_problem_dict["space_ns"]
variables_dict_ns = ns_problem_dict["variables_dict"]
Bc = ns_problem_dict["Bc"]
function_space_ns = ns_problem_dict["function_space_ns"]
# variables for solving the problem pf
solver_pf = pf_problem_dict["solver_pf"]
solution_vector_pf = pf_problem_dict["solution_vector_pf"]
solution_vector_pf_0 = pf_problem_dict["solution_vector_pf_0"]
spaces_pf = pf_problem_dict["spaces_pf"]
variables_dict_pf = pf_problem_dict["variables_dict_pf"]
vel_answer_on_pf_mesh = pf_problem_dict["vel_answer_on_pf_mesh"]


T = 0
####### write first solution to file ########
solution_vectors = [solution_vector_ns_0, solution_vector_pf_0]
times = [T, T]  # Assuming T_ns and T_pf are defined times for NS and PF solutions
variable_names_list = [["Vel", "Press"], ["Phi", "U"]]  # Adjust variable names as needed
extra_funcs_dict = { "velocity_PF": vel_answer_on_pf_mesh}  # Assuming these are defined

write_simulation_data_to_single_file(solution_vectors, times, file, variable_names_list, extra_funcs_dict)






for it in tqdm( range(0, 10000000) ):




    # solving the problem

    solver_pf_information = solver_pf.solve()
    solver_ns_information = solver_ns.solve()




    #definning old solution vectors
    old_solution_vector_ns = solution_vector_ns
    old_solution_vector_pf = solution_vector_pf
    old_solution_vector_0_ns = solution_vector_ns
    old_solution_vector_0_pf = solution_vector_pf

    #update the old solution vectors
    solution_vector_ns_0.assign(solution_vector_ns)
    solution_vector_pf_0.assign(solution_vector_pf)


    T += dt



    if it == 20 or it % 30 == 25 :
        # refining the mesh
        mesh, mesh_info = refine_mesh(physical_parameters_dict, coarse_mesh, solution_vector_pf, spaces_pf, solution_vector_ns, space_ns, comm )

        # define problem on the new mesh
        ns_problem_dict = update_solver_on_new_mesh_ns(mesh, physical_parameters_dict,
                                     old_solution_vector_ns= old_solution_vector_ns, old_solution_vector_0_ns= old_solution_vector_0_ns, 
                                    old_solution_vector_0_pf = old_solution_vector_0_pf , variables_dict= None )
        
        pf_problem_dict = update_solver_on_new_mesh_pf(mesh, physical_parameters_dict,
                                     old_solution_vector_pf= old_solution_vector_pf, old_solution_vector_0_pf=old_solution_vector_0_pf, 
                                    old_solution_vector_0_ns=old_solution_vector_0_ns, variables_dict_pf= None )
        
        # variables for solving the problem ns
        solver_ns = ns_problem_dict["solver_ns"]
        solution_vector_ns = ns_problem_dict["solution_vector_ns"]
        solution_vector_ns_0 = ns_problem_dict["solution_vector_ns_0"]
        space_ns = ns_problem_dict["space_ns"]
        variables_dict_ns = ns_problem_dict["variables_dict"]
        Bc = ns_problem_dict["Bc"]
        # variables for solving the problem pf
        solver_pf = pf_problem_dict["solver_pf"]
        solution_vector_pf = pf_problem_dict["solution_vector_pf"]
        solution_vector_pf_0 = pf_problem_dict["solution_vector_pf_0"]
        spaces_pf = pf_problem_dict["spaces_pf"]
        variables_dict_pf = pf_problem_dict["variables_dict_pf"]
        vel_answer_on_pf_mesh = pf_problem_dict["vel_answer_on_pf_mesh"]
        function_space_ns = ns_problem_dict["function_space_ns"]

        

    else: 

        ns_problem_dict = update_solver_on_new_mesh_ns(mesh, physical_parameters_dict, old_solution_vector_ns=None, old_solution_vector_0_ns=None, 
                                    old_solution_vector_0_pf = old_solution_vector_0_pf , variables_dict= variables_dict_ns )
        
        pf_problem_dict = update_solver_on_new_mesh_pf(mesh, physical_parameters_dict, old_solution_vector_pf= None, old_solution_vector_0_pf=None, 
                                old_solution_vector_0_ns=old_solution_vector_0_ns, variables_dict_pf= variables_dict_pf)
        

        # variables for solving the problem
        variables_dict_ns = ns_problem_dict["variables_dict"] # new added
        solution_vector_ns = variables_dict_ns["solution_vector_ns"]
        solution_vector_ns_0 = variables_dict_ns["solution_vector_ns_0"]
        space_ns = variables_dict_ns["spaces_ns"]
        Bc = ns_problem_dict["Bc"]
        solver_ns = ns_problem_dict["solver_ns"]
        # variables for solving the problem pf  
        variables_dict_pf = pf_problem_dict["variables_dict_pf"] # new added
        solution_vector_pf = variables_dict_pf["solution_vector_pf"]
        solution_vector_pf_0 = variables_dict_pf["solution_vector_pf_0"]
        spaces_pf = variables_dict_pf["spaces_pf"]
        vel_answer_on_pf_mesh = variables_dict_pf["v_answer_on_pf_mesh"]
        solver_pf = pf_problem_dict["solver_pf"]


        


    ####### write first solution to file ########
    if it % 20 == 0: 
        solution_vectors = [solution_vector_ns_0, solution_vector_pf_0]
        times = [T, T]  # Assuming T_ns and T_pf are defined times for NS and PF solutions
        variable_names_list = [["Vel", "Press"], ["Phi", "U"]]  # Adjust variable names as needed
        extra_funcs_dict = {"velocity_PF": vel_answer_on_pf_mesh}  # Assuming these are defined
        write_simulation_data_to_single_file(solution_vectors, times, file, variable_names_list, extra_funcs_dict)
































    # ###################### Information about simulation ######################
        
    # if  (it == 20 or it % 30 == 25 ):


    #     # extract the information from the solver
    #     number_of_iterations_pf = solver_pf_information[0]
    #     number_of_iterations_ns = solver_ns_information[0]
    #     convergence_pf = solver_pf_information[1]
    #     convergence_ns = solver_ns_information[1]
    #     u_max, u_min = compute_global_velocity_extremes(solution_vector_ns_0, function_space_ns, comm)
    #     # Compute the Reynolds number
    #     reynolds_number = u_max * Nx / viscosity_liquid 
    #     # mesh information: 
    #     n_cells = mesh_info['n_cells']
    #     hmin = mesh_info['hmin']
    #     hmax = mesh_info['hmax']
    #     dx_min = mesh_info['dx_min']
    #     dx_max = mesh_info['dx_max']
    #     # real time in seconds and number of iterations
    #     current_time = time.time() # Capture the current time
    #     current_elapsed_time_seconds = current_time - start_time
    #     T_seconds = T * tau_0_scale
    #     time_need_to_fininsh_seconds = estimate_completion_time(it, T_seconds, current_elapsed_time_seconds, 1, dt* tau_0_scale)
    #     time_need_to_fininsh_hours = time_need_to_fininsh_seconds/60

    #     if rank == 0 :
    #          print(f"""
    #             Simulation Results and Parameters:
    #             - Phase Field Solver:
    #                 Number of Iterations: {number_of_iterations_pf}
    #             - Navier-Stokes Solver:
    #                 Number of Iterations: {number_of_iterations_ns}
    #             - Velocity Extremes:
    #                 Maximum Velocity (u_max): {u_max:.8f}
    #                 Minimum Velocity (u_min): {u_min:.8f}
    #             - Reynolds Number:
    #                 Re = {reynolds_number:.5f}
    #             - Mesh Information:
    #                 Number of Cells: {n_cells}
    #                 Minimum Cell Diameter (dx_min): {dx_min:.4f}
    #                 Maximum Cell Diameter (dx_max): {dx_max:.4f}
    #             - Simulation Time:
    #                 Real Time in Seconds: {T_seconds:.10e}
    #                 Estimated Time to Finish simulation in hours: {time_need_to_fininsh_hours:.10e}
    #             """, flush=True)







    





