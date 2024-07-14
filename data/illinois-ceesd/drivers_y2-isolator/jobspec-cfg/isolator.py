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
import logging
import sys
import yaml
import numpy as np
import pyopencl as cl
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
import math
from pytools.obj_array import make_obj_array
from functools import partial

from mirgecom.discretization import create_discretization_collection
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import BoundaryDomainTag
from grudge.op import nodal_max, nodal_min
from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info,
    logmgr_set_time,
    set_sim_state, logmgr_add_device_memory_usage
)

from mirgecom.navierstokes import ns_operator
from mirgecom.artificial_viscosity import \
    av_laplacian_operator, smoothness_indicator
from mirgecom.simutil import (
    check_step,
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    get_sim_timestep,
    force_evaluation
)
from mirgecom.restart import write_restart_file
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.integrators import (rk4_step, lsrk54_step, lsrk144_step,
                                  euler_step)

from mirgecom.fluid import make_conserved
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    IsothermalWallBoundary,
)
#from mirgecom.initializers import (Uniform, PlanarDiscontinuity)
from mirgecom.eos import IdealSingleGas
from mirgecom.transport import SimpleTransport
from mirgecom.gas_model import GasModel, make_fluid_state


class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


#root_logger = logging.getLogger()

#h1 = logging.StreamHandler(sys.stdout)
#f1 = SingleLevelFilter(logging.INFO, False)
#h1.addFilter(f1)
#root_logger.addHandler(h1)

#h2 = logging.StreamHandler(sys.stderr)
#f2 = SingleLevelFilter(logging.DEBUG, True)
#h2.addFilter(f2)
#root_logger.addHandler(h2)

#root_logger.setLevel(logging.DEBUG)
#logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)
#logger.debug("A DEBUG message")
#logger.info("An INFO message")
#logger.warning("A WARNING message")
#logger.error("An ERROR message")
#logger.critical("A CRITICAL message")


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


def get_mesh(dim, mesh_filename, read_mesh=True):
    """Get the mesh."""
    from meshmode.mesh.io import read_gmsh
    mesh = partial(read_gmsh, filename=mesh_filename, force_ambient_dim=dim)

    return mesh


def sponge(cv, cv_ref, sigma):
    return sigma*(cv_ref - cv)


class InitSponge:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    """
    def __init__(self, *, x0, thickness, amplitude):
        r"""Initialize the sponge parameters.

        Parameters
        ----------
        x0: float
            sponge starting x location
        thickness: float
            sponge extent
        amplitude: float
            sponge strength modifier
        """

        self._x0 = x0
        self._thickness = thickness
        self._amplitude = amplitude

    def __call__(self, x_vec, *, time=0.0):
        """Create the sponge intensity at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        time: float
            Time at which solution is desired. The strength is (optionally)
            dependent on time
        """
        xpos = x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        x0 = zeros + self._x0

        return self._amplitude * actx.np.where(
            actx.np.greater(xpos, x0),
            (zeros + ((xpos - self._x0)/self._thickness) *
            ((xpos - self._x0)/self._thickness)),
            zeros + 0.0
        )


def getIsentropicPressure(mach, P0, gamma):
    pressure = (1. + (gamma - 1.)*0.5*mach**2)
    pressure = P0*pressure**(-gamma / (gamma - 1.))
    return pressure


def getIsentropicTemperature(mach, T0, gamma):
    temperature = (1. + (gamma - 1.)*0.5*mach**2)
    temperature = T0/temperature
    return temperature


def getMachFromAreaRatio(area_ratio, gamma, mach_guess=0.01):
    error = 1.0e-8
    nextError = 1.0e8
    g = gamma
    M0 = mach_guess
    while nextError > error:
        R = (((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))/M0
            - area_ratio)
        dRdM = (2*((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
               / (2*g - 2)*(g - 1)/(2/(g + 1) + ((g - 1)/(g + 1)*M0*M0)) -
               ((2/(g + 1) + ((g - 1)/(g + 1)*M0*M0))**(((g + 1)/(2*g - 2))))
               * M0**(-2))
        M1 = M0 - R/dRdM
        nextError = abs(R)
        M0 = M1

    return M1


def get_y_from_x(x, data):
    """
    Return the linearly interpolated the value of y
    from the value in data(x,y) at x
    """

    if x <= data[0][0]:
        y = data[0][1]
    elif x >= data[-1][0]:
        y = data[-1][1]
    else:
        ileft = 0
        iright = data.shape[0]-1

        # find the bracketing points, simple subdivision search
        while iright - ileft > 1:
            ind = int(ileft+(iright - ileft)/2)
            if x < data[ind][0]:
                iright = ind
            else:
                ileft = ind

        leftx = data[ileft][0]
        rightx = data[iright][0]
        lefty = data[ileft][1]
        righty = data[iright][1]

        dx = rightx - leftx
        dy = righty - lefty
        y = lefty + (x - leftx)*dy/dx
    return y


def get_theta_from_data(data):
    """
    Calculate theta = arctan(dy/dx)
    Where data[][0] = x and data[][1] = y
    """

    theta = data.copy()
    for index in range(1, theta.shape[0]-1):
        #print(f"index {index}")
        theta[index][1] = np.arctan((data[index+1][1]-data[index-1][1]) /
                          (data[index+1][0]-data[index-1][0]))
    theta[0][1] = np.arctan(data[1][1]-data[0][1])/(data[1][0]-data[0][0])
    theta[-1][1] = np.arctan(data[-1][1]-data[-2][1])/(data[-1][0]-data[-2][0])
    return theta


class InitACTII:
    r"""Solution initializer for flow in the ACT-II facility

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=2, nspecies=0, geom_top, geom_bottom,
            P0, T0, temp_wall, temp_sigma, vel_sigma
    ):
        r"""Initialize mixture parameters.

        Parameters
        ----------
        dim: int
            specifies the number of dimensions for the solution
        P0: float
            stagnation pressure
        T0: float
            stagnation temperature
        temp_wall: float
            wall temperature
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        geom_top: numpy.ndarray
            coordinates for the top wall
        geom_bottom: numpy.ndarray
            coordinates for the bottom wall
        """

        # check number of points in the geometry
        #top_size = geom_top.size
        #bottom_size = geom_bottom.size

        self._dim = dim
        self._P0 = P0
        self._T0 = T0
        self._geom_top = geom_top
        self._geom_bottom = geom_bottom
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        # TODO, calculate these from the geometry files
        self._throat_height = 3.61909e-3
        self._x_throat = 0.283718298

    def __call__(self, dcoll, x_vec, eos, *, time=0.0):
        """Create the solution state at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        eos:
            Mixture-compatible equation-of-state object must provide
            these functions:
            `eos.get_density`
            `eos.get_internal_energy`
        time: float
            Time at which solution is desired. The location is (optionally)
            dependent on time
        """
        if x_vec.shape != (self._dim,):
            raise ValueError(f"Position vector has unexpected dimensionality,"
                             f" expected {self._dim}.")

        xpos = x_vec[0]
        ypos = x_vec[1]
        if self._dim == 3:
            zpos = x_vec[2]
        ytop = 0*x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        ones = zeros + 1.0

        gamma = eos.gamma()
        gas_const = eos.gas_const()

        mach = zeros
        ytop = zeros
        ybottom = zeros
        theta = zeros

        theta_geom_top = get_theta_from_data(self._geom_top)
        theta_geom_bottom = get_theta_from_data(self._geom_bottom)

        # process the mesh piecemeal, one interval at a time
        # linearly interpolate between the data points
        area_ratio = ((self._geom_top[0][1] - self._geom_bottom[0][1]) /
                      self._throat_height)
        if self._geom_top[0][0] < self._x_throat:
            mach_left = getMachFromAreaRatio(area_ratio=area_ratio,
                                             gamma=gamma,
                                             mach_guess=0.01)
        elif self._geom_top[0][0] > self._x_throat:
            mach_left = getMachFromAreaRatio(area_ratio=area_ratio,
                                             gamma=gamma,
                                             mach_guess=1.01)
        else:
            mach_left = 1.0
        x_left = self._geom_top[0][0]
        ytop_left = self._geom_top[0][1]
        ybottom_left = self._geom_bottom[0][1]
        theta_top_left = theta_geom_top[0][1]
        theta_bottom_left = theta_geom_bottom[0][1]

        for ind in range(1, self._geom_top.shape[0]):
            area_ratio = ((self._geom_top[ind][1] - self._geom_bottom[ind][1]) /
                          self._throat_height)
            if self._geom_top[ind][0] < self._x_throat:
                mach_right = getMachFromAreaRatio(area_ratio=area_ratio,
                                                 gamma=gamma,
                                                 mach_guess=0.01)
            elif self._geom_top[ind][0] > self._x_throat:
                mach_right = getMachFromAreaRatio(area_ratio=area_ratio,
                                                 gamma=gamma,
                                                 mach_guess=1.01)
            else:
                mach_right = 1.0
            ytop_right = self._geom_top[ind][1]
            ybottom_right = self._geom_bottom[ind][1]
            theta_top_right = theta_geom_top[ind][1]
            theta_bottom_right = theta_geom_bottom[ind][1]

            # interpolate our data
            x_right = self._geom_top[ind][0]

            dx = x_right - x_left
            dm = mach_right - mach_left
            dyt = ytop_right - ytop_left
            dyb = ybottom_right - ybottom_left
            dtb = theta_bottom_right - theta_bottom_left
            dtt = theta_top_right - theta_top_left

            local_mach = mach_left + (xpos - x_left)*dm/dx
            local_ytop = ytop_left + (xpos - x_left)*dyt/dx
            local_ybottom = ybottom_left + (xpos - x_left)*dyb/dx
            local_theta_bottom = theta_bottom_left + (xpos - x_left)*dtb/dx
            local_theta_top = theta_top_left + (xpos - x_left)*dtt/dx

            local_theta = (local_theta_bottom +
                           (local_theta_top - local_theta_bottom) /
                           (local_ytop - local_ybottom)*(ypos - local_ybottom))

            # extend just a a little bit to catch the edges
            left_edge = actx.np.greater(xpos, x_left - 1.e-6)
            right_edge = actx.np.less(xpos, x_right + 1.e-6)
            inside_block = left_edge*right_edge

            mach = actx.np.where(inside_block, local_mach, mach)
            ytop = actx.np.where(inside_block, local_ytop, ytop)
            ybottom = actx.np.where(inside_block, local_ybottom, ybottom)
            theta = actx.np.where(inside_block, local_theta, theta)

            mach_left = mach_right
            ytop_left = ytop_right
            ybottom_left = ybottom_right
            theta_bottom_left = theta_bottom_right
            theta_top_left = theta_top_right
            x_left = x_right

        pressure = getIsentropicPressure(
            mach=mach,
            P0=self._P0,
            gamma=gamma
        )
        temperature = getIsentropicTemperature(
            mach=mach,
            T0=self._T0,
            gamma=gamma
        )

        # modify the temperature in the near wall region to match the
        # isothermal boundaries
        sigma = self._temp_sigma
        wall_temperature = self._temp_wall
        smoothing_top = actx.np.tanh(sigma*(actx.np.abs(ypos-ytop)))
        smoothing_bottom = actx.np.tanh(sigma*(actx.np.abs(ypos-ybottom)))
        smoothing_fore = ones
        smoothing_aft = ones
        z0 = 0.
        z1 = 0.035
        if self._dim == 3:
            smoothing_fore = actx.np.tanh(sigma*(actx.np.abs(zpos-z0)))
            smoothing_aft = actx.np.tanh(sigma*(actx.np.abs(zpos-z1)))

        smooth_temperature = (wall_temperature +
            (temperature - wall_temperature)*smoothing_top*smoothing_bottom *
                                             smoothing_fore*smoothing_aft)

        # make a little region along the top of the cavity where we don't want
        # the temperature smoothed
        xc_left = zeros + 0.65163 + 0.0004
        xc_right = zeros + 0.72163 - 0.0004
        yc_top = zeros - 0.006
        yc_bottom = zeros - 0.01

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        top_edge = actx.np.less(ypos, yc_top)
        bottom_edge = actx.np.greater(ypos, yc_bottom)
        inside_block = left_edge*right_edge*top_edge*bottom_edge
        temperature = actx.np.where(inside_block, temperature, smooth_temperature)

        mass = pressure/temperature/gas_const
        velocity = np.zeros(self._dim, dtype=object)
        # the magnitude
        velocity[0] = mach*actx.np.sqrt(gamma*pressure/mass)

        # modify the velocity in the near-wall region to have a tanh profile
        # this approximates the BL velocity profile
        sigma = self._vel_sigma
        smoothing_top = actx.np.tanh(sigma*(actx.np.abs(ypos-ytop)))
        smoothing_bottom = actx.np.tanh(sigma*(actx.np.abs(ypos-ybottom)))
        smoothing_fore = ones
        smoothing_aft = ones
        if self._dim == 3:
            smoothing_fore = actx.np.tanh(sigma*(actx.np.abs(zpos-z0)))
            smoothing_aft = actx.np.tanh(sigma*(actx.np.abs(zpos-z1)))
        velocity[0] = (velocity[0]*smoothing_top*smoothing_bottom *
                       smoothing_fore*smoothing_aft)

        # split into x and y components
        velocity[1] = velocity[0]*actx.np.sin(theta)
        velocity[0] = velocity[0]*actx.np.cos(theta)

        # zero out the velocity in the cavity region, let the flow develop naturally
        # initially in pressure/temperature equilibrium with the exterior flow
        zeros = 0*xpos
        xc_left = zeros + 0.65163 - 0.000001
        #xc_right = zeros + 0.72163 + 0.000001
        xc_right = zeros + 0.73
        yc_top = zeros - 0.0083245
        yc_bottom = zeros - 0.0283245
        xc_bottom = zeros + 0.70163
        wall_theta = np.sqrt(2)/2.

        left_edge = actx.np.greater(xpos, xc_left)
        right_edge = actx.np.less(xpos, xc_right)
        top_edge = actx.np.less(ypos, yc_top)
        inside_cavity = left_edge*right_edge*top_edge

        # smooth the temperature at the cavity walls
        sigma = self._temp_sigma
        smoothing_front = actx.np.tanh(sigma*(actx.np.abs(xpos-xc_left)))
        smoothing_bottom = actx.np.tanh(sigma*(actx.np.abs(ypos-yc_bottom)))
        wall_dist = (wall_theta*(ypos - yc_bottom) -
                     wall_theta*(xpos - xc_bottom))
        smoothing_slant = actx.np.tanh(sigma*(actx.np.abs(wall_dist)))
        cavity_temperature = (wall_temperature +
            (temperature - wall_temperature) *
             smoothing_front*smoothing_bottom*smoothing_slant)
        temperature = actx.np.where(inside_cavity, cavity_temperature, temperature)

        mass = pressure/temperature/gas_const

        # zero of the velocity
        velocity[0] = actx.np.where(inside_cavity, zeros, velocity[0])

        mom = velocity*mass
        energy = (pressure/(gamma - 1.0)) + np.dot(mom, mom)/(2.0*mass)
        return make_conserved(
            dim=self._dim,
            mass=mass,
            momentum=mom,
            energy=energy
        )


class UniformModified:
    r"""Solution initializer for a uniform flow with boundary layer smoothing.

    Similar to the Uniform initializer, except the velocity profile is modified
    so that the velocity goes to zero at y(min, max)

    The smoothing comes from a hyperbolic tangent with weight sigma

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self, *, dim=1, nspecies=0, pressure=1.0, temperature=2.5,
            velocity=None, mass_fracs=None,
            temp_wall, temp_sigma, vel_sigma,
            ymin=0., ymax=1.0
    ):
        r"""Initialize uniform flow parameters.

        Parameters
        ----------
        dim: int
            specify the number of dimensions for the flow
        nspecies: int
            specify the number of species in the flow
        temperature: float
            specifies the temperature
        pressure: float
            specifies the pressure
        velocity: numpy.ndarray
            specifies the flow velocity
        temp_wall: float
            wall temperature
        temp_sigma: float
            near-wall temperature relaxation parameter
        vel_sigma: float
            near-wall velocity relaxation parameter
        ymin: flaot
            minimum y-coordinate for smoothing
        ymax: float
            maximum y-coordinate for smoothing
        """
        if velocity is not None:
            numvel = len(velocity)
            myvel = velocity
            if numvel > dim:
                dim = numvel
            elif numvel < dim:
                myvel = np.zeros(shape=(dim,))
                for i in range(numvel):
                    myvel[i] = velocity[i]
            self._velocity = myvel
        else:
            self._velocity = np.zeros(shape=(dim,))

        if mass_fracs is not None:
            self._nspecies = len(mass_fracs)
            self._mass_fracs = mass_fracs
        else:
            self._nspecies = nspecies
            self._mass_fracs = np.zeros(shape=(nspecies,))

        if self._velocity.shape != (dim,):
            raise ValueError(f"Expected {dim}-dimensional inputs.")

        self._pressure = pressure
        self._temperature = temperature
        self._dim = dim
        self._temp_wall = temp_wall
        self._temp_sigma = temp_sigma
        self._vel_sigma = vel_sigma
        self._ymin = ymin
        self._ymax = ymax

    def __call__(self, x_vec, *, eos, **kwargs):
        """
        Create a uniform flow solution at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Nodal coordinates
        eos: :class:`mirgecom.eos.IdealSingleGas`
            Equation of state class with method to supply gas *gamma*.
        """

        ypos = x_vec[1]
        actx = ypos.array_context
        ymax = 0.0*x_vec[1] + self._ymax
        ymin = 0.0*x_vec[1] + self._ymin
        ones = (1.0 + x_vec[0]) - x_vec[0]

        pressure = self._pressure * ones
        temperature = self._temperature * ones

        # modify the temperature in the near wall region to match
        # the isothermal boundaries
        sigma = self._temp_sigma
        wall_temperature = self._temp_wall
        smoothing_min = actx.np.tanh(sigma*(actx.np.abs(ypos-ymin)))
        smoothing_max = actx.np.tanh(sigma*(actx.np.abs(ypos-ymax)))
        temperature = (wall_temperature +
                       (temperature - wall_temperature)*smoothing_min*smoothing_max)

        velocity = make_obj_array([self._velocity[i] * ones
                                   for i in range(self._dim)])
        y = make_obj_array([self._mass_fracs[i] * ones
                            for i in range(self._nspecies)])
        if self._nspecies:
            mass = eos.get_density(pressure, temperature, y)
        else:
            mass = pressure/temperature/eos.gas_const()
        specmass = mass * y

        sigma = self._vel_sigma
        # modify the velocity profile from uniform
        smoothing_max = actx.np.tanh(sigma*(actx.np.abs(ypos-ymax)))
        smoothing_min = actx.np.tanh(sigma*(actx.np.abs(ypos-ymin)))
        velocity[0] = velocity[0]*smoothing_max*smoothing_min

        mom = mass*velocity
        if self._nspecies:
            internal_energy = eos.get_internal_energy(temperature=temperature,
                                                      species_mass=specmass)
        else:
            internal_energy = pressure/(eos.gamma() - 1)
        kinetic_energy = 0.5 * np.dot(mom, mom)/mass
        energy = internal_energy + kinetic_energy

        return make_conserved(dim=self._dim, mass=mass, energy=energy,
                              momentum=mom, species_mass=specmass)


@mpi_entry_point
def main(ctx_factory=cl.create_some_context, restart_filename=None,
         use_profiling=False, use_logmgr=True, user_input_file=None,
         use_overintegration=False, actx_class=None, casename=None,
         lazy=False):

    if actx_class is None:
        raise RuntimeError("Array context class missing.")

    # control log messages
    logger = logging.getLogger(__name__)
    logger.propagate = False

    if (logger.hasHandlers()):
        logger.handlers.clear()

    # send info level messages to stdout
    h1 = logging.StreamHandler(sys.stdout)
    f1 = SingleLevelFilter(logging.INFO, False)
    h1.addFilter(f1)
    logger.addHandler(h1)

    # send everything else to stderr
    h2 = logging.StreamHandler(sys.stderr)
    f2 = SingleLevelFilter(logging.INFO, True)
    h2.addFilter(f2)
    logger.addHandler(h2)

    """Drive the Y0 example."""
    cl_ctx = ctx_factory()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nparts = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    if casename is None:
        casename = "mirgecom"

    # logging and profiling
    log_path = "log_data/"
    # log_path = ""
    logname = log_path + casename + ".sqlite"

    if rank == 0:
        import os
        log_dir = os.path.dirname(logname)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    comm.Barrier()

    logmgr = initialize_logmgr(use_logmgr,
        filename=logname, mode="wo", mpi_comm=comm)

    if use_profiling:
        queue = cl.CommandQueue(cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    # main array context for the simulation
    from mirgecom.simutil import get_reasonable_memory_pool
    alloc = get_reasonable_memory_pool(cl_ctx, queue)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000, allocator=alloc)
    else:
        actx = actx_class(comm, queue, allocator=alloc, force_device_scalars=True)

    # default i/o junk frequencies
    nviz = 500
    nhealth = 1
    nrestart = 5000
    nstatus = 1

    # default timestepping control
    integrator = "rk4"
    current_dt = 1e-8
    t_final = 1e-7
    current_t = 0
    current_step = 0
    current_cfl = 1.0
    constant_cfl = False

    # default health status bounds
    health_pres_min = 1.0e-1
    health_pres_max = 2.0e6

    # discretization and model control
    order = 1
    alpha_sc = 0.3
    s0_sc = -5.0
    kappa_sc = 0.5
    dim = 2
    mesh_filename = "data/isolator.msh"

    # material properties
    mu = 1.0e-5
    mu_override = False  # optionally read in from input

    if user_input_file:
        input_data = None
        if rank == 0:
            with open(user_input_file) as f:
                input_data = yaml.load(f, Loader=yaml.FullLoader)
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
            alpha_sc = float(input_data["alpha_sc"])
        except KeyError:
            pass
        try:
            kappa_sc = float(input_data["kappa_sc"])
        except KeyError:
            pass
        try:
            s0_sc = float(input_data["s0_sc"])
        except KeyError:
            pass
        try:
            mu_input = float(input_data["mu"])
            mu_override = True
        except KeyError:
            pass
        try:
            order = int(input_data["order"])
        except KeyError:
            pass
        try:
            dim = int(input_data["dimen"])
        except KeyError:
            pass
        try:
            integrator = input_data["integrator"]
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
            mesh_filename = input_data["mesh_filename"]
        except KeyError:
            pass

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    s0_sc = np.log10(1.0e-4 / np.power(order, 4))
    if rank == 0:
        print(f"Shock capturing parameters: alpha {alpha_sc}, "
              f"s0 {s0_sc}, kappa {kappa_sc}")

    if rank == 0:
        print("\n#### Simluation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        print(f"\tcurrent_dt = {current_dt}")
        print(f"\tt_final = {t_final}")
        print(f"\torder = {order}")
        print(f"\tdimen = {dim}")
        print(f"\tTime integration {integrator}")
        print("#### Simluation control data: ####\n")

    timestepper = rk4_step
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "lsrk144":
        timestepper = lsrk144_step

    # }}}
    # working gas: O2/N2 #
    #   O2 mass fraction 0.273
    #   gamma = 1.4
    #   cp = 37.135 J/mol-K,
    #   rho= 1.977 kg/m^3 @298K
    gamma = 1.4
    mw_o2 = 15.999*2
    mw_n2 = 14.0067*2
    mf_o2 = 0.273
    # visocsity @ 400C, Pa-s
    mu_o2 = 3.76e-5
    mu_n2 = 3.19e-5
    mu_mix = mu_o2*mf_o2 + mu_n2*(1-mu_o2)  # 3.3456e-5
    mw = mw_o2*mf_o2 + mw_n2*(1.0 - mf_o2)
    r = 8314.59/mw
    cp = r*gamma/(gamma - 1)
    Pr = 0.75

    if mu_override:
        mu = mu_input
    else:
        mu = mu_mix

    kappa = cp*mu/Pr

    if rank == 0:
        print("\n#### Simluation material properties: ####")
        print(f"\tmu = {mu}")
        print(f"\tkappa = {kappa}")
        print(f"\tPrandtl Number  = {Pr}")

    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)

    #
    # nozzle inflow #
    #
    # stagnation tempertuare 2076.43 K
    # stagnation pressure 2.745e5 Pa
    #
    # isentropic expansion based on the area ratios between the inlet (r=54e-3m) and
    # the throat (r=3.167e-3)
    #
    vel_inflow = np.zeros(shape=(dim,))
    vel_outflow = np.zeros(shape=(dim,))
    total_pres_inflow = 2.745e5
    total_temp_inflow = 2076.43

    throat_height = 3.61909e-3
    inlet_height = 54.129e-3
    outlet_height = 28.54986e-3
    inlet_area_ratio = inlet_height/throat_height
    outlet_area_ratio = outlet_height/throat_height

    inlet_mach = getMachFromAreaRatio(area_ratio=inlet_area_ratio,
                                      gamma=gamma,
                                      mach_guess=0.01)
    pres_inflow = getIsentropicPressure(mach=inlet_mach,
                                        P0=total_pres_inflow,
                                        gamma=gamma)
    temp_inflow = getIsentropicTemperature(mach=inlet_mach,
                                           T0=total_temp_inflow,
                                           gamma=gamma)
    rho_inflow = pres_inflow/temp_inflow/r
    vel_inflow[0] = inlet_mach*math.sqrt(gamma*pres_inflow/rho_inflow)

    if rank == 0:
        print("#### Simluation initialization data: ####")
        print(f"\tinlet Mach number {inlet_mach}")
        print(f"\tinlet temperature {temp_inflow}")
        print(f"\tinlet pressure {pres_inflow}")
        print(f"\tinlet rho {rho_inflow}")
        print(f"\tinlet velocity {vel_inflow[0]}")
        #print(f"final inlet pressure {pres_inflow_final}")

    outlet_mach = getMachFromAreaRatio(area_ratio=outlet_area_ratio,
                                       gamma=gamma,
                                       mach_guess=1.1)
    pres_outflow = getIsentropicPressure(mach=outlet_mach,
                                         P0=total_pres_inflow,
                                         gamma=gamma)
    temp_outflow = getIsentropicTemperature(mach=outlet_mach,
                                            T0=total_temp_inflow,
                                            gamma=gamma)
    rho_outflow = pres_outflow/temp_outflow/r
    vel_outflow[0] = outlet_mach*math.sqrt(gamma*pres_outflow/rho_outflow)

    if rank == 0:
        print(f"\toutlet Mach number {outlet_mach}")
        print(f"\toutlet temperature {temp_outflow}")
        print(f"\toutlet pressure {pres_outflow}")
        print(f"\toutlet rho {rho_outflow}")
        print(f"\toutlet velocity {vel_outflow[0]}")
        print("#### Simluation initialization data: ####\n")

    eos = IdealSingleGas(gamma=gamma, gas_const=r)
    gas_model = GasModel(eos=eos, transport=transport_model)

    # read geometry files
    geometry_bottom = None
    geometry_top = None
    if rank == 0:
        from numpy import loadtxt
        geometry_bottom = loadtxt("nozzleBottom.dat", comments="#", unpack=False)
        geometry_top = loadtxt("nozzleTop.dat", comments="#", unpack=False)
    geometry_bottom = comm.bcast(geometry_bottom, root=0)
    geometry_top = comm.bcast(geometry_top, root=0)

    # parameters to adjust the shape of the initialization
    vel_sigma = 2000
    temp_sigma = 2500
    temp_wall = 300

    bulk_init = InitACTII(dim=dim,
                          geom_top=geometry_top, geom_bottom=geometry_bottom,
                          P0=total_pres_inflow, T0=total_temp_inflow,
                          temp_wall=temp_wall, temp_sigma=temp_sigma,
                          vel_sigma=vel_sigma)

    _inflow_init = UniformModified(
        dim=dim,
        temperature=temp_inflow,
        pressure=pres_inflow,
        velocity=vel_inflow,
        temp_wall=temp_wall,
        temp_sigma=temp_sigma,
        vel_sigma=vel_sigma,
        ymin=-0.0270645,
        ymax=0.0270645
    )

    _outflow_init = UniformModified(
        dim=dim,
        temperature=temp_outflow,
        pressure=pres_outflow,
        velocity=vel_outflow,
        temp_wall=temp_wall,
        temp_sigma=temp_sigma,
        vel_sigma=vel_sigma,
        ymin=-0.016874377,
        ymax=0.011675488
    )

    def _boundary_state_func(dcoll, dd_bdry, gas_model, actx, init_func, **kwargs):
        bnd_discr = dcoll.discr_from_dd(dd_bdry)
        nodes = actx.thaw(bnd_discr.nodes())
        return make_fluid_state(init_func(x_vec=nodes, eos=gas_model.eos,
                                          **kwargs), gas_model)

    def _inflow_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return _boundary_state_func(dcoll, dd_bdry, gas_model,
                                    state_minus.array_context,
                                    _inflow_init, **kwargs)

    def _outflow_state_func(dcoll, dd_bdry, gas_model, state_minus, **kwargs):
        return _boundary_state_func(dcoll, dd_bdry, gas_model,
                                    state_minus.array_context,
                                    _outflow_init, **kwargs)

    inflow = PrescribedFluidBoundary(boundary_state_func=_inflow_state_func)
    outflow = PrescribedFluidBoundary(boundary_state_func=_outflow_state_func)
    wall = IsothermalWallBoundary()

    boundaries = {
        BoundaryDomainTag("inflow"): inflow,
        BoundaryDomainTag("outflow"): outflow,
        BoundaryDomainTag("wall"): wall,
    }

    viz_path = "viz_data/"
    vizname = viz_path + casename
    restart_path = "restart_data/"
    restart_pattern = (
        restart_path + "{cname}-{step:06d}-{rank:04d}.pkl"
    )
    if restart_filename:  # read the grid from restart data
        restart_filename = f"{restart_filename}-{rank:04d}.pkl"

        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_filename)
        current_step = restart_data["step"]
        current_t = restart_data["t"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert restart_data["nparts"] == nparts
    else:  # generate the grid from scratch
        if rank == 0:
            print(f"Reading mesh from {mesh_filename}")
        local_mesh, global_nelements = generate_and_distribute_mesh(
            comm, get_mesh(dim=dim, mesh_filename=mesh_filename))
        local_nelements = local_mesh.nelements

    if rank == 0:
        logger.info("Making discretization")

    dcoll = create_discretization_collection(
        actx, local_mesh, order=order, mpi_communicator=comm)

    from grudge.dof_desc import DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    if rank == 0:
        logger.info("Done making discretization")

    # initialize the sponge field
    sponge_thickness = 0.09
    sponge_amp = 1.0/current_dt/1000
    sponge_x0 = 0.9

    sponge_init = InitSponge(x0=sponge_x0, thickness=sponge_thickness,
                             amplitude=sponge_amp)
    sponge_sigma = sponge_init(x_vec=actx.thaw(dcoll.nodes()))
    ref_cv = bulk_init(dcoll=dcoll, x_vec=actx.thaw(dcoll.nodes()),
                       eos=eos, time=0)

    vis_timer = None

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "step walltime: {value:6g} s")
            #("t_log.max", "log walltime: {value:6g} s")
        ])

        try:
            logmgr.add_watches(["memory_usage.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

    if rank == 0:
        logger.info("Before restart/init")

    if restart_filename:
        if rank == 0:
            logger.info("Restarting soln.")
        current_cv = restart_data["cv"]
        if restart_order != order:
            restart_dcoll = create_discretization_collection(
                actx, local_mesh, order=restart_order, mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd("vol"),
                restart_dcoll.discr_from_dd("vol")
            )
            current_cv = connection(restart_data["cv"])
        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)
    else:
        # Set the current state from time 0
        if rank == 0:
            logger.info("Initializing soln.")
        current_cv = bulk_init(dcoll=dcoll, x_vec=actx.thaw(dcoll.nodes()),
                                  eos=eos, time=0)

    current_state = make_fluid_state(current_cv, gas_model)

    visualizer = make_visualizer(dcoll)

    #    initname = initializer.__class__.__name__
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order, nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=casename,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)

    def get_fluid_state(cv):
        return make_fluid_state(cv=cv, gas_model=gas_model)

    create_fluid_state = actx.compile(get_fluid_state)

    def my_write_status(dt, cfl, dv):
        status_msg = f"-------- dt = {dt:1.3e}, cfl = {cfl:1.4f}"
        temp = dv.temperature
        pres = dv.pressure
        from grudge.op import nodal_min_loc, nodal_max_loc
        pmin = global_reduce(
            actx.to_numpy(nodal_min_loc(dcoll, "vol", pres)), op="min")
        pmax = global_reduce(
            actx.to_numpy(nodal_max_loc(dcoll, "vol", pres)), op="max")
        dv_status_msg = (
            f"\n-------- P (min, max) (Pa) = ({pmin:1.9e}, {pmax:1.9e})")
        tmin = global_reduce(
            actx.to_numpy(nodal_min_loc(dcoll, "vol", temp)), op="min")
        tmax = global_reduce(
            actx.to_numpy(nodal_max_loc(dcoll, "vol", temp)), op="max")
        dv_status_msg += (
            f"\n-------- T (min, max) (K)  = ({tmin:7g}, {tmax:7g})")
        status_msg += dv_status_msg
        status_msg += "\n"

        if rank == 0:
            logger.info(status_msg)

    def my_write_viz(step, t, cv, dv, ts_field, alpha_field):
        tagged_cells = smoothness_indicator(dcoll, cv.mass, s0=s0_sc,
                                            kappa=kappa_sc)

        mach = (actx.np.sqrt(np.dot(cv.velocity, cv.velocity)) /
                            dv.speed_of_sound)
        viz_fields = [("cv", cv),
                      ("dv", dv),
                      ("mach", mach),
                      ("velocity", cv.velocity),
                      ("sponge_sigma", sponge_sigma),
                      ("alpha", alpha_field),
                      ("tagged_cells", tagged_cells),
                      ("dt" if constant_cfl else "cfl", ts_field)]
        write_visfile(dcoll=dcoll, io_fields=viz_fields, visualizer=visualizer,
                      vizname=vizname, comm=comm, step=step, t=t, overwrite=True)

    def my_write_restart(step, t, cv):
        restart_fname = restart_pattern.format(cname=casename, step=step, rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "local_mesh": local_mesh,
                "cv": cv,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, restart_data, restart_fname, comm)

    def my_health_check(dv):
        health_error = False
        if check_naninf_local(dcoll, "vol", dv.pressure):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_range_local(dcoll, "vol", dv.pressure,
                                     health_pres_min, health_pres_max),
                                     op="lor"):
            health_error = True
            p_min = actx.to_numpy(nodal_min(dcoll, "vol", dv.pressure))
            p_max = actx.to_numpy(nodal_max(dcoll, "vol", dv.pressure))
            logger.info(f"Pressure range violation ({p_min=}, {p_max=})")

        return health_error

    def my_get_viscous_timestep(dcoll, state, alpha):
        """Routine returns the the node-local maximum stable viscous timestep.

        Parameters
        ----------
        dcoll: grudge.eager.EagerDGDiscretization
            the discretization collection to use
        state: :class:`~mirgecom.gas_model.FluidState`
            Full fluid state including conserved and thermal state
        alpha: :class:`~meshmode.DOFArray`
            Arfifical viscosity

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The maximum stable timestep at each node.
        """
        from grudge.dt_utils import characteristic_lengthscales

        length_scales = characteristic_lengthscales(state.array_context, dcoll)

        nu = 0
        d_alpha_max = 0

        if state.is_viscous:
            from mirgecom.viscous import get_local_max_species_diffusivity
            nu = state.viscosity/state.mass_density
            d_alpha_max = \
                get_local_max_species_diffusivity(
                    state.array_context,
                    state.species_diffusivity
                )

        return (
            length_scales / (state.wavespeed
            + ((nu + d_alpha_max + alpha) / length_scales))
        )

    def my_get_viscous_cfl(dcoll, dt, state, alpha):
        """Calculate and return node-local CFL based on current state and timestep.

        Parameters
        ----------
        dcoll: :class:`grudge.eager.EagerDGDiscretization`
            the discretization collection to use
        dt: float or :class:`~meshmode.dof_array.DOFArray`
            A constant scalar dt or node-local dt
        state: :class:`~mirgecom.gas_model.FluidState`
            Full fluid state including conserved and thermal state
        alpha: :class:`~meshmode.DOFArray`
            Arfifical viscosity

        Returns
        -------
        :class:`~meshmode.dof_array.DOFArray`
            The CFL at each node.
        """
        return dt / my_get_viscous_timestep(dcoll, state=state, alpha=alpha)

    def my_get_timestep(t, dt, state, alpha):
        t_remaining = max(0, t_final - t)
        if constant_cfl:
            ts_field = current_cfl * my_get_viscous_timestep(dcoll, state=state,
                                                             alpha=alpha)
            from grudge.op import nodal_min
            dt = actx.to_numpy(nodal_min(dcoll, "vol", ts_field))
            cfl = current_cfl
        else:
            ts_field = my_get_viscous_cfl(dcoll, dt=dt, state=state, alpha=alpha)
            from grudge.op import nodal_max
            cfl = actx.to_numpy(nodal_max(dcoll, "vol", ts_field))

        return ts_field, cfl, min(t_remaining, dt)

    def my_get_alpha(dcoll, state, alpha):
        """ Scale alpha by the element characteristic length """
        from grudge.dt_utils import characteristic_lengthscales
        array_context = state.array_context
        length_scales = characteristic_lengthscales(array_context, dcoll)

        #from mirgecom.fluid import compute_wavespeed
        #wavespeed = compute_wavespeed(eos, state)

        vmag = array_context.np.sqrt(np.dot(state.velocity, state.velocity))
        #alpha_field = alpha*wavespeed*length_scales
        alpha_field = alpha*vmag*length_scales
        #alpha_field = wavespeed*0 + alpha*current_step
        #alpha_field = state.mass

        return alpha_field

    def my_pre_step(step, t, dt, state):
        fluid_state = force_evaluation(actx, create_fluid_state(cv=state))
        cv = fluid_state.cv
        dv = fluid_state.dv

        try:

            if logmgr:
                logmgr.tick_before()

            alpha_field = my_get_alpha(dcoll, fluid_state, alpha_sc)
            ts_field, cfl, dt = my_get_timestep(t, dt, fluid_state, alpha_field)

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)
            do_status = check_step(step=step, interval=nstatus)

            if do_health:
                health_errors = global_reduce(my_health_check(dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.warning("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_status:
                my_write_status(dt=dt, cfl=cfl, dv=dv)

            if do_restart:
                my_write_restart(step=step, t=t, cv=cv)

            if do_viz:
                my_write_viz(step=step, t=t, cv=cv, dv=dv,
                             ts_field=ts_field, alpha_field=alpha_field)

        except MyRuntimeError:
            if rank == 0:
                logger.error("Errors detected; attempting graceful exit.")
            my_write_viz(step=step, t=t, cv=cv, dv=dv, ts_field=ts_field,
                         alpha_field=alpha_field)
            my_write_restart(step=step, t=t, cv=cv)
            raise

        dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl, t_final,
                              constant_cfl)

        return state, dt

    def my_post_step(step, t, dt, state):
        # Logmgr needs to know about EOS, dt, dim?
        # imo this is a design/scope flaw
        if logmgr:
            set_dt(logmgr, dt)
            set_sim_state(logmgr, dim, state, eos)
            logmgr.tick_after()
        return state, dt

    def my_rhs(t, state):
        fluid_state = make_fluid_state(cv=state, gas_model=gas_model)
        alpha_field = my_get_alpha(dcoll, fluid_state, alpha_sc)
        return (
            ns_operator(dcoll, state=fluid_state, time=t, boundaries=boundaries,
                        gas_model=gas_model, quadrature_tag=quadrature_tag)
            + av_laplacian_operator(dcoll, fluid_state=fluid_state,
                                    boundaries=boundaries,
                                    boundary_kwargs={"time": t,
                                                     "gas_model": gas_model},
                                    alpha=alpha_field, s0=s0_sc, kappa=kappa_sc,
                                    quadrature_tag=quadrature_tag)
            + sponge(cv=fluid_state.cv, cv_ref=ref_cv, sigma=sponge_sigma)
        )

    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    current_step, current_t, current_cv = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      istep=current_step, dt=current_dt,
                      t=current_t, t_final=t_final,
                      state=current_state.cv)
    current_state = create_fluid_state(current_cv)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")
    final_dv = current_state.dv
    alpha_field = my_get_alpha(dcoll, current_state, alpha_sc)
    ts_field, cfl, dt = my_get_timestep(t=current_t, dt=current_dt,
                                        state=current_state, alpha=alpha_field)
    my_write_status(dt=dt, cfl=cfl, dv=final_dv)

    my_write_viz(step=current_step, t=current_t, cv=current_state.cv, dv=final_dv,
                 ts_field=ts_field, alpha_field=alpha_field)
    my_write_restart(step=current_step, t=current_t, cv=current_state.cv)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    finish_tol = 1e-16
    assert np.abs(current_t - t_final) < finish_tol


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO)

    #root_logger = logging.getLogger()

    #logging.debug("A DEBUG message")
    #logging.info("An INFO message")
    #logging.warning("A WARNING message")
    #logging.error("An ERROR message")
    #logging.critical("A CRITICAL message")

    import argparse
    parser = argparse.ArgumentParser(
        description="MIRGE-Com Isentropic Nozzle Driver")
    parser.add_argument("-r", "--restart_file", type=ascii, dest="restart_file",
                        nargs="?", action="store", help="simulation restart file")
    parser.add_argument("-i", "--input_file", type=ascii, dest="input_file",
                        nargs="?", action="store", help="simulation config file")
    parser.add_argument("-c", "--casename", type=ascii, dest="casename", nargs="?",
                        action="store", help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=False,
                        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
                        help="enable lazy evaluation [OFF]")
    parser.add_argument("--overintegration", action="store_true",
        help="use overintegration in the RHS computations")

    args = parser.parse_args()

    # for writing output
    casename = "isolator"
    if args.casename:
        print(f"Custom casename {args.casename}")
        casename = args.casename.replace("'", "")
    else:
        print(f"Default casename {casename}")
    lazy = args.lazy
    if args.profile:
        if lazy:
            raise ValueError("Can't use lazy and profiling together.")

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)

    restart_filename = None
    if args.restart_file:
        restart_filename = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_filename}")

    input_file = None
    if args.input_file:
        input_file = args.input_file.replace("'", "")
        print(f"Using user input from file: {input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    main(restart_filename=restart_filename, user_input_file=input_file,
         use_profiling=args.profile, use_logmgr=args.log,
         use_overintegration=args.overintegration, lazy=lazy,
         actx_class=actx_class, casename=casename)

# vim: foldmethod=marker
