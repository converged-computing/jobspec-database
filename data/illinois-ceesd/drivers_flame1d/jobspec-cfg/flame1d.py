"""mirgecom driver for the Y0 demonstration.

Note: this example requires a *scaled* version of the Y0
grid. A working grid example is located here:
github.com:/illinois-ceesd/data@y0scaled
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import yaml
import logging
import sys
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from functools import partial

from arraycontext import thaw, freeze
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.dof_desc import BoundaryDomainTag
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
)
from mirgecom.restart import (
    write_restart_file
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools
# from mirgecom.checkstate import compare_states
from mirgecom.integrators import (
    rk4_step,
    lsrk54_step,
    lsrk144_step,
    euler_step
)
from mirgecom.steppers import advance_state
from mirgecom.boundary import PrescribedFluidBoundary
from mirgecom.fluid import make_conserved
from mirgecom.initializers import PlanarDiscontinuity
from mirgecom.transport import SimpleTransport
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import GasModel, make_fluid_state
import cantera

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info, logmgr_set_time, LogUserQuantity,
    set_sim_state
)


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


h1 = logging.StreamHandler(sys.stdout)
f1 = SingleLevelFilter(logging.INFO, False)
h1.addFilter(f1)
root_logger = logging.getLogger()
root_logger.addHandler(h1)
h2 = logging.StreamHandler(sys.stderr)
f2 = SingleLevelFilter(logging.INFO, True)
h2.addFilter(f2)
root_logger.addHandler(h2)
root_logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, casename="flame1d",
         user_input_file=None, restart_file=None, use_profiling=False,
         use_logmgr=False, use_lazy_eval=False, actx_class=None):
    """Drive the 1D Flame example."""
    if actx_class is None:
        raise RuntimeError("Array context class missing.")

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    restart_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    snapshot_pattern = restart_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    # logging and profiling
    log_path = "log_data/"
    logname = log_path + casename + ".sqlite"

    if rank == 0:
        import os
        log_dir = os.path.dirname(logname)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    logmgr = initialize_logmgr(use_logmgr,
        filename=logname, mode="wo", mpi_comm=comm)

    cl_ctx = ctx_factory()
    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    if use_lazy_eval:
        actx = actx_class(comm, queue, mpi_base_tag=12000)
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    # default i/o frequencies
    nviz = 100
    nrestart = 100
    nhealth = 100
    nstatus = 1

    # default timestepping control
    integrator = "rk4"
    current_dt = 1e-9
    t_final = 1.e-3

    # default health status bounds
    health_pres_min = 1.0
    health_pres_max = 2.0e6
    health_mass_frac_min = -1.0e-9
    health_mass_frac_max = 1.0 + 1.e-9
    health_temp_min = 1.0
    health_temp_max = 3000.0
    # discretization and model control
    order = 1
    char_len = 0.0001
    fuel = "C2H4"

    if user_input_file:
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            input_data = None
        input_data = comm.bcast(input_data, root=0)

        try:
            nviz = int(input_data["nviz"])
        except KeyError:
            pass
        try:
            nrestart = int(input_data["nrestart"])
        except KeyError:
            pass
        try:
            nhealth = int(input_data["nhealth"])
        except KeyError:
            pass
        try:
            nstatus = int(input_data["nstatus"])
        except KeyError:
            pass
        try:
            current_dt = float(input_data["current_dt"])
        except KeyError:
            pass
        try:
            t_final = float(input_data["t_final"])
        except KeyError:
            pass
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            char_len = float(input_data["char_len"])
        except KeyError:
            pass
        try:
            integrator = input_data["integrator"]
        except KeyError:
            pass
        try:
            fuel = input_data["fuel"]
        except KeyError:
            pass
        try:
            health_pres_min = float(input_data["health_pres_min"])
        except KeyError:
            pass
        try:
            health_pres_max = float(input_data["health_pres_max"])
        except KeyError:
            pass
        try:
            health_mass_frac_min = float(input_data["health_mass_frac_min"])
        except KeyError:
            pass
        try:
            health_mass_frac_max = float(input_data["health_mass_frac_max"])
        except KeyError:
            pass

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step

    allowed_fuels = ["H2", "C2H4"]
    if fuel not in allowed_fuels:
        error_message = "Invalid fuel selection: {}".format(fuel)
        raise RuntimeError(error_message)

    if rank == 0:
        print("#### Simluation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tTime integration {integrator}")
        print(f"\tFuel: {fuel}")
        print("#### Simluation control data: ####")

    dim = 2
    current_cfl = 1.0
    current_t = 0
    constant_cfl = False
    current_step = 0

    vel_burned = np.zeros(shape=(dim,))
    vel_unburned = np.zeros(shape=(dim,))

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    # -- Pick up a CTI for the thermochemistry config
    # --- Note: Users may add their own CTI file by dropping it into
    # ---       mirgecom/mechanisms alongside the other CTI files.

    from mirgecom.mechanisms import get_mechanism_cti
    if fuel == "C2H4":
        mech_cti = get_mechanism_cti("uiuc")
    elif fuel == "H2":
        mech_cti = get_mechanism_cti("sanDiego")
        # mech_cti = get_mechanism_cti("sanDiego_trans")

    cantera_soln = cantera.Solution(phase_id="gas", source=mech_cti)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixutre mole fractions are needed to
    # set up the initial state in Cantera.
    temp_unburned = 300.0
    temp_ignition = 1500.0
    # Parameters for calculating the amounts of fuel, oxidizer, and inert species
    if fuel == "C2H4":
        stoich_ratio = 3.0
    if fuel == "H2":
        stoich_ratio = 0.5
    equiv_ratio = 1.0
    ox_di_ratio = 0.21
    # Grab the array indices for the specific species, ethylene, oxygen, and nitrogen
    i_fu = cantera_soln.species_index(fuel)
    i_ox = cantera_soln.species_index("O2")
    i_di = cantera_soln.species_index("N2")
    x = np.zeros(nspecies)
    # Set the species mole fractions according to our desired fuel/air mixture
    x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
    x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
    x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
    # Uncomment next line to make pylint fail when it can't find cantera.one_atm
    one_atm = cantera.one_atm  # pylint: disable=no-member
    # one_atm = 101325.0
    pres_unburned = one_atm

    # Let the user know about how Cantera is being initilized
    print(f"Input state (T,P,X) = ({temp_unburned}, {pres_unburned}, {x}")
    # Set Cantera internal gas temperature, pressure, and mole fractios
    cantera_soln.TPX = temp_unburned, pres_unburned, x
    # Pull temperature, total density, mass fractions, and pressure from Cantera
    # We need total density, and mass fractions to initialize the fluid/gas state.
    y_unburned = np.zeros(nspecies)
    can_t, rho_unburned, y_unburned = cantera_soln.TDY

    # *can_t*, *can_p* should not differ (significantly) from user's initial data,
    # but we want to ensure that we use exactly the same starting point as Cantera,
    # so we use Cantera's version of these data.

    # now find the conditions for the burned gas
    cantera_soln.equilibrate("TP")
    temp_burned, rho_burned, y_burned = cantera_soln.TDY
    pres_burned = cantera_soln.P

    from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
    pyrometheus_mechanism = make_pyrometheus_mechanism_class(cantera_soln)(actx.np)

    kappa = 1.3e-2  # Pr = mu*cp/kappa = 0.75
    mu = 1.e-5
    species_diffusivity = 1.e-5 * np.ones(nspecies)
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa,
                                      species_diffusivity=species_diffusivity)

    eos = PyrometheusMixture(pyrometheus_mechanism, temperature_guess=temp_unburned)
    species_names = pyrometheus_mechanism.species_names
    gas_model = GasModel(eos=eos, transport=transport_model)

    print(f"Pyrometheus mechanism species names {species_names}")
    print(f"Unburned (T,P,Y) = ({temp_unburned}, {pres_unburned}, {y_unburned}")
    print(f"Burned (T,P,Y) = ({temp_burned}, {pres_burned}, {y_burned}")

    flame_start_loc = 0.10

    # use the burned conditions with a lower temperature
    bulk_init = PlanarDiscontinuity(dim=dim,
                                    disc_location=flame_start_loc,
                                    sigma=0.0005,
                                    nspecies=nspecies,
                                    temperature_right=temp_ignition,
                                    temperature_left=temp_unburned,
                                    pressure_right=pres_burned,
                                    pressure_left=pres_unburned,
                                    velocity_right=vel_burned,
                                    velocity_left=vel_unburned,
                                    species_mass_right=y_burned,
                                    species_mass_left=y_unburned)

    def _symmetry_func(nodes, eos, cv, **kwargs):
        dim = len(nodes)
        momentum = 1.0*cv.momentum
        momentum[1] = -momentum[1]
        return make_conserved(dim=dim, mass=cv.mass, momentum=momentum,
                              energy=cv.energy, species_mass=cv.species_mass)

    def dummy(nodes, eos, cv, **kwargs):
        return 1.0*cv

    def _flow_bnd(nodes, cv, eos, flow_vel, flow_pres, flow_temp, flow_spec,
                  **kwargs):
        ones = 0*cv.mass + 1.0
        pressure = flow_pres * ones
        temperature = flow_temp * ones
        velocity = 0*cv.velocity + flow_vel
        y = make_obj_array([flow_spec[i] * ones
                            for i in range(nspecies)])

        mass = eos.get_density(pressure, temperature, y)
        specmass = mass * y
        mom = mass * velocity
        internal_energy = eos.get_internal_energy(temperature=temperature,
                                                   species_mass_fractions=y)
        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass * (internal_energy + kinetic_energy)

        return make_conserved(dim=cv.dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)

    def _inflow_func(nodes, cv, eos, **kwargs):
        return _flow_bnd(nodes, cv, eos, flow_vel=vel_burned,
                         flow_pres=pres_burned, flow_temp=temp_ignition,
                         flow_spec=y_burned, **kwargs)

    def _outflow_func(nodes, cv, eos, **kwargs):
        return _flow_bnd(nodes, cv, eos, flow_vel=vel_unburned,
                        flow_pres=pres_unburned, flow_temp=temp_unburned,
                        flow_spec=y_unburned, **kwargs)

    def _boundary_state_func(dcoll, dd_bdry, gas_model, state_minus, init_func,
                             **kwargs):
        actx = state_minus.array_context
        bnd_discr = dcoll.discr_from_dd(dd_bdry)
        nodes = thaw(bnd_discr.nodes(), actx)
        return make_fluid_state(init_func(nodes=nodes, eos=gas_model.eos,
                                          cv=state_minus.cv, **kwargs),
                                gas_model=gas_model,
                                temperature_seed=state_minus.temperature)

    def _inflow_boundary_state(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return _boundary_state_func(dcoll, dd_bdry, gas_model, state_minus,
                                    _inflow_func, **kwargs)

    def _outflow_boundary_state(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return _boundary_state_func(dcoll, dd_bdry, gas_model, state_minus,
                                    _outflow_func, **kwargs)

    def _symmetry_boundary_state(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return _boundary_state_func(dcoll, dd_bdry, gas_model, state_minus,
                                    _symmetry_func, **kwargs)

    wall_symmetry = \
        PrescribedFluidBoundary(boundary_state_func=_symmetry_boundary_state)
    inflow_boundary = \
        PrescribedFluidBoundary(boundary_state_func=_inflow_boundary_state)
    outflow_boundary = \
        PrescribedFluidBoundary(boundary_state_func=_outflow_boundary_state)

    boundaries = {BoundaryDomainTag("Inflow"): inflow_boundary,
                  BoundaryDomainTag("Outflow"): outflow_boundary,
                  BoundaryDomainTag("Wall"): wall_symmetry}

    restart_step = None
    if restart_file is None:
        box_ll = (0.0, 0.0)
        box_ur = (0.2, 0.00125)
        num_elements = (int((box_ur[0]-box_ll[0])/char_len),
                            int((box_ur[1]-box_ll[1])/char_len))

        from meshmode.mesh.generation import generate_regular_rect_mesh
        generate_mesh = partial(generate_regular_rect_mesh,
                                a=box_ll,
                                b=box_ur,
                                n=num_elements,
                                boundary_tag_to_face={
                                    "Inflow": ["+x"],
                                    "Outflow": ["-x"],
                                    "Wall": ["+y", "-y"]})
        local_mesh, global_nelements = (
            generate_and_distribute_mesh(comm, generate_mesh))
        local_nelements = local_mesh.nelements

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["num_parts"]

    if rank == 0:
        logging.info("Making discretization")
    dcoll = EagerDGDiscretization(actx,
                                  local_mesh,
                                  order=order,
                                  mpi_communicator=comm)
    nodes = thaw(dcoll.nodes(), actx)

    def get_fluid_state(cv, temperature_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temperature_seed)

    def get_temperature_update(cv, temperature):
        y = cv.species_mass_fractions
        e = eos.internal_energy(cv) / cv.mass
        return make_obj_array(
            [pyrometheus_mechanism.get_temperature_update_energy(e, temperature, y)]
        )

    compute_temperature_update = actx.compile(get_temperature_update)
    create_fluid_state = actx.compile(get_fluid_state)

    temperature_seed = can_t

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")
        # for Discontinuity initial conditions
        current_cv = bulk_init(x_vec=nodes, eos=eos, time=0.)
        # for uniform background initial condition
        #current_state = bulk_init(nodes, eos=eos)
    else:
        current_t = restart_data["t"]
        current_step = restart_step

        if restart_order != order:
            restart_dcoll = EagerDGDiscretization(
                actx,
                local_mesh,
                order=restart_order,
                mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd("vol"),
                restart_dcoll.discr_from_dd("vol"))

            current_cv = connection(restart_data["cv"])
            temperature_seed = connection(restart_data["temperature_seed"])
        else:
            current_cv = restart_data["cv"]
            temperature_seed = restart_data["temperature_seed"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)

    current_state = create_fluid_state(current_cv, temperature_seed)
    temperature_seed = current_state.temperature

    vis_timer = None
    log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)
        #logmgr_add_package_versions(logmgr)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "------- step walltime: {value:6g} s, "),
            ("t_log.max", "log walltime: {value:6g} s\n")])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    visualizer = make_visualizer(dcoll)

    initname = "flame1d"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    from pytools.obj_array import make_obj_array

    def get_fluid_state(cv, temperature_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=temperature_seed)

    def get_temperature_update(cv, temperature):
        y = cv.species_mass_fractions
        e = eos.internal_energy(cv) / cv.mass
        return make_obj_array(
            [pyrometheus_mechanism.get_temperature_update_energy(e, temperature, y)]
        )

    compute_temperature_update = actx.compile(get_temperature_update)
    create_fluid_state = actx.compile(get_fluid_state)

    from mirgecom.viscous import get_viscous_timestep

    def get_dt(state):
        return make_obj_array([get_viscous_timestep(dcoll, state=state)])

    compute_dt = actx.compile(get_dt)

    from mirgecom.viscous import get_viscous_cfl

    def get_cfl(state, dt):
        return make_obj_array([get_viscous_cfl(dcoll, dt, state=state)])

    compute_cfl = actx.compile(get_cfl)

    def get_production_rates(cv, temperature):
        return make_obj_array([eos.get_production_rates(cv, temperature)])

    compute_production_rates = actx.compile(get_production_rates)

    def vol_min_loc(x):
        from grudge.op import nodal_min_loc
        return actx.to_numpy(nodal_min_loc(dcoll, "vol", x))[()]

    def vol_max_loc(x):
        from grudge.op import nodal_max_loc
        return actx.to_numpy(nodal_max_loc(dcoll, "vol", x))[()]

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(dcoll, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(dcoll, "vol", x))[()]

    def my_write_viz(step, t, dt, cv, dv, ts_field):
        reaction_rates, = compute_production_rates(cv, dv.temperature)
        viz_fields = [("CV_rho", cv.mass),
                      ("CV_rhoU", cv.momentum[0]),
                      ("CV_rhoV", cv.momentum[1]),
                      ("CV_rhoE", cv.energy),
                      ("pressure", dv.pressure),
                      ("temperature", dv.temperature),
                      ("reaction_rates", reaction_rates),
                      ("dt" if constant_cfl else "cfl", ts_field)]
        # species mass fractions
        viz_fields.extend(
            ("Y_"+species_names[i], cv.species_mass_fractions[i])
            for i in range(nspecies))
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)

    def my_write_restart(step, t, cv, temperature_seed):
        rst_fname = snapshot_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "state": cv,
                "temperature_seed": temperature_seed,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(cv, dv):
        health_error = False
        pressure = thaw(freeze(dv.pressure, actx), actx)
        temperature = thaw(freeze(dv.temperature, actx), actx)

        if global_reduce(check_naninf_local(dcoll, "vol", pressure), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_range_local(dcoll, "vol", pressure,
                                           health_pres_min, health_pres_max),
                         op="lor"):
            health_error = True
            logger.info(f"{rank=}: Pressure range violation ({health_pres_min},"
                        f"{health_pres_max}).")
            max_pressure = actx.to_numpy(dcoll.norm(dv.pressure, np.inf))[()]
            logger.info(f"{rank=}: {max_pressure=}")

        if global_reduce(check_naninf_local(dcoll, "vol", temperature), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")

        if global_reduce(check_range_local(dcoll, "vol", temperature,
                                           health_temp_min, health_temp_max),
                         op="lor"):
            health_error = True
            logger.info(f"{rank=}: Temperature range violation.")

        for i in range(nspecies):
            if global_reduce(check_range_local(dcoll, "vol",
                                               cv.species_mass_fractions[i],
                                               health_mass_frac_min,
                                               health_mass_frac_max),
                             op="lor"):
                health_error = True
                logger.info(f"{rank=}: species mass fraction range violation.")

        # temperature_update is the next temperature update in the
        # `get_temperature` Newton solve. The relative size of this
        # update is used to gauge convergence of the current temperature
        # after a fixed number of Newton iters.
        # Note: The local max jig below works around a very long compile
        # in lazy mode.
        from grudge import op
        temp_update, = compute_temperature_update(cv, temperature)
        temp_resid = thaw(freeze(temp_update, actx), actx) / temperature
        temp_resid = (actx.to_numpy(op.nodal_max_loc(dcoll, "vol", temp_resid)))
        if temp_resid > 1e-8:
            health_error = True
            logger.info(f"{rank=}: Temperature is not converged {temp_resid=}.")

        return health_error

    def my_health_report(cv, dv):
        logger.info("Simulation global status report.")
        pressure = thaw(freeze(dv.pressure, actx), actx)
        temperature = thaw(freeze(dv.temperature, actx), actx)
        p_min = vol_min(pressure)
        p_max = vol_max(pressure)
        temp_min = vol_min(temperature)
        temp_max = vol_max(temperature)
        rho_min = vol_min(cv.mass)
        rho_max = vol_max(cv.mass)
        from pytools.obj_array import obj_array_vectorize
        vel_min = obj_array_vectorize(lambda x: vol_min(x),
                                      cv.velocity)
        vel_max = obj_array_vectorize(lambda x: vol_max(x),
                                      cv.velocity)
        y_min = obj_array_vectorize(lambda x: vol_min(x),
                                      cv.species_mass_fractions)
        y_max = obj_array_vectorize(lambda x: vol_max(x),
                                      cv.species_mass_fractions)
        energy_min = vol_min(cv.energy)
        energy_max = vol_max(cv.energy)

        logger.info(f" ---- Density range ({rho_min: 1.5e}, {rho_max: 1.5e})")
        for i in range(dim):
            logger.info(f" ---- Velocity range [{i}] ({vel_min[i]: 1.5e},"
                        f" {vel_max[i]: 1.5e})")
        logger.info(f" ---- Energy range ({energy_min: 1.9e}, {energy_max: 1.9e})")
        for i in range(nspecies):
            logger.info(f" ---- Mass fraction range [{i}] ({y_min[i]: 1.5e},"
                        f" {y_max[i]: 1.5e}) ({species_names[i]})")
        logger.info(f" ---- Pressure range ({p_min: 1.9e}, {p_max: 1.9e})")
        logger.info(f" ---- Temperature range ({temp_min: 5g}, {temp_max: 5g})")

    def my_get_timestep(t, dt, state):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            cfl = current_cfl
            ts_field = cfl * compute_dt(state)[0]
            ts_field = thaw(freeze(ts_field, actx), actx)
            dt = global_reduce(vol_min_loc(ts_field), op="min", comm=comm)
        else:
            ts_field = compute_cfl(state, current_dt)[0]
            cfl = global_reduce(vol_max_loc(ts_field), op="max", comm=comm)

        return ts_field, cfl, min(t_remaining, dt)

    def my_pre_step(step, t, dt, state):
        cv, tseed = state
        fluid_state = create_fluid_state(cv=cv, temperature_seed=tseed)
        dv = fluid_state.dv

        try:

            if logmgr:
                logmgr.tick_before()

            ts_field, cfl, dt = my_get_timestep(t, dt, fluid_state)
            log_cfl.set_quantity(cfl)

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            if do_health:
                health_errors = global_reduce(my_health_check(cv, dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    # my_health_report(state, dv)
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, cv=cv, temperature_seed=tseed)

            if do_viz:
                my_write_viz(step=step, t=t, dt=dt, cv=cv, dv=dv, ts_field=ts_field)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, dt=dt, cv=cv, dv=dv, ts_field=ts_field)
            my_write_restart(step=step, t=t, cv=cv, temperature_seed=tseed)
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        cv, tseed = state
        fluid_state = create_fluid_state(cv=cv, temperature_seed=tseed)

        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, cv, gas_model.eos)
            logmgr.tick_after()
        return make_obj_array([fluid_state.cv, fluid_state.temperature]), dt

    def my_rhs(t, state):
        cv, tseed = state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed)
        cv_rhs = (
            ns_operator(dcoll, state=fluid_state, time=t, boundaries=boundaries,
                        gas_model=gas_model) +
            eos.get_species_source_terms(cv=cv, temperature=fluid_state.temperature)
        )
        return make_obj_array([cv_rhs, 0*tseed])

    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, stepper_state) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=make_obj_array([current_state.cv, temperature_seed]),
                      dt=current_dt, t_final=t_final, t=current_t,
                      istep=current_step)
    current_cv, tseed = stepper_state
    current_state = make_fluid_state(current_cv, gas_model, tseed)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = current_state.dv
    ts_field, cfl, dt = my_get_timestep(current_t, current_dt, current_state)
    my_write_viz(step=current_step, t=current_t, dt=current_dt,
                 cv=current_state.cv, dv=final_dv, ts_field=ts_field)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv,
                     temperature_seed=tseed)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com 1D Flame Driver")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("-i", "--input_file",  type=ascii,
                        dest="input_file", nargs="?", action="store",
                        help="simulation config file")
    parser.add_argument("-c", "--casename",  type=ascii,
                        dest="casename", nargs="?", action="store",
                        help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()
    lazy = args.lazy
    if args.profile:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    # for writing output
    casename = "flame1d"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if args.input_file:
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")

    main(restart_file=restart_file, user_input_file=input_file,
         actx_class=actx_class, use_profiling=args.profile,
         use_lazy_eval=lazy, use_logmgr=args.log)
