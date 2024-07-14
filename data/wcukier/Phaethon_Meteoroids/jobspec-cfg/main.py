################################################################################
# main.py                                                                      #
# Author: Wolf Cukier                                                          #
# Main file for geminids simulation.                                           #
# python main.py $run_num $model_num $n_particles $age                         #
################################################################################

## Imports ##
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from tqdm import tqdm
import scipy
import spiceypy as spice
import rebound
import reboundx
import os
import sys


from geminids.constants import *


## Local Imports ##
from geminids.generateBeta import particles as particles
from geminids.perihelion import perihelion as perihelion
from geminids.cometary_start import init_loc as init_loc, max_beta


## Main -- python main.py $run_num $model_num $age ##
if (__name__ == "__main__"):
    MIN_A = 0.2 # min semi-major axis cutoff

    # Initialize
    n_escaped = 0
    k = int(sys.argv[1])
    model = int(sys.argv[2])
    # dr = os.getcwd()
    if   (model == 0): subdir = "novel"
    elif (model == 1): subdir = "vel"
    elif (model == 2): subdir = "distr" #aka cometary model
    elif (model == 3): subdir = "novel_comet"   # "_comet" refers to composition
    elif (model == 4): subdir = "vel_comet"     # not formation
    elif (model == 5): subdir = "distr_comet"
    elif (model == 6): subdir = "novel_old"
    else:
            print("second argument must be an integer between 0 and 6")
            raise

    print(k)
    t_and_r = {}

    window = 2
    pts_per_year = 1000
    n_particles = int(sys.argv[3])

    # Allow for age to not be an arg
    try:
            age = int(sys.argv[4])
    except:
            age = 2000

    # Fetch initial data
    n_years = age
    if (age==2000):
            try:
                    y0 = np.load("data/perihelion.npy", allow_pickle=True)
                    t_start = np.mean(np.load("data/t_start.npy", allow_pickle=True))
                    orbit = np.load("data/orig_orbit.npy", allow_pickle=True)
            except:
                    y0, t_start, orbit = perihelion()
    else:
            y0, t_start, orbit = perihelion(age=age)

    n_part = 10000
    if (model % 3 == 2): #distr models
        b_max = max_beta(int(k/1000), orbit, n=1000); 
        n_part = 100000
    else: b_max = .052


    beta, mass, vel = particles(n_part, model, max_b=b_max) #non-cometary models
    
    #slice to current run
    beta = beta[n_particles*k: n_particles*(k+1)]
    mass = mass[n_particles*k: n_particles*(k+1)]
    vel = vel[n_particles*k: n_particles*(k+1)] 
    print(vel*au, flush=True)

    n = n_particles

    # Initialize simulation
    sim = rebound.Simulation()
    sim.tstep = .001


    spice.furnsh("data/meta.tm")
    sim.units = ('s', 'AU', 'Msun')
    sim.exit_max_distace = 10.
    sim.add(m=1.) # Sun

    [yj, lt] = spice.spkezr("JUPITER BARYCENTER", t_start, "ECLIPJ2000", "NONE", "SUN")
    yj = spice.convrt(yj, "KM", "AU")
    sim.add(m=0.000954588, x=yj[0], y=yj[1], z=yj[2], vx=yj[3], vy=yj[4], vz=yj[5])

    [e_pos, lt] = spice.spkezr('EARTH', t_start, 'ECLIPJ2000', 'NONE', 'SUN')
    e_pos = spice.convrt(e_pos, 'KM', 'AU')
    sim.add(m=MASS_E/MASS_SUN, x=e_pos[0], y=e_pos[1], z=e_pos[2], vx=e_pos[3], vy=e_pos[4], vz=e_pos[5])

    [mr_pos, lt] = spice.spkezr('4', t_start, 'ECLIPJ2000', 'NONE', 'SUN')
    mr_pos = spice.convrt(mr_pos, 'KM', 'AU')
    sim.add(m=MASS_MR/MASS_SUN, x=mr_pos[0], y=mr_pos[1], z=mr_pos[2], vx=mr_pos[3], vy=mr_pos[4], vz=mr_pos[5])

    [v_pos, lt] = spice.spkezr('VENUS', t_start, 'ECLIPJ2000', 'NONE', 'SUN')
    v_pos = spice.convrt(v_pos, 'KM', 'AU')
    sim.add(m=MASS_V/MASS_SUN, x=v_pos[0], y=v_pos[1], z=v_pos[2], vx=v_pos[3], vy=v_pos[4], vz=v_pos[5])

    [hg_pos, lt] = spice.spkezr('MERCURY', t_start, 'ECLIPJ2000', 'NONE', 'SUN')
    hg_pos = spice.convrt(hg_pos, 'KM', 'AU')
    sim.add(m=MASS_HG/MASS_SUN, x=hg_pos[0], y=hg_pos[1], z=hg_pos[2], vx=hg_pos[3], vy=hg_pos[4], vz=hg_pos[5])

    # make the simulation ignore the mass of the dust
    n_active = len(sim.particles)
    sim.n_active = n_active
    sim.Nactive = n_active
    sim.collision = "none"

    t = 0
    r = 0

    # Add particles
    sim.move_to_hel()
    for i in range(n):
        if ((model % 3) == 2): # Cometary Model -- many starting positions
            y0, r, t = init_loc(int(k/100), orbit)
            t_and_r[str(i)] = [t, r]
        if ((model % 3) != 1): vel[i] = [0,0,0] # Not velocity model -- set vel to 9
        sim.add(x = y0[0]+1e-10*np.random.rand(), y=y0[1], z=y0[2], vx=y0[3]+vel[i,0], vy = y0[4]+vel[i,1], vz = y0[5]+vel[i,2], hash = f"{i}")
    sim.move_to_com()

    # Initialize radiation forces
    rebx = reboundx.Extras(sim)
    rf = rebx.load_force("radiation_forces")
    rebx.add_force(rf)
    rf.params["c"] = 0.002004


    # Initialize timesteps
    Noutputs = window*pts_per_year
    year = spice.jyear()
    times = np.linspace(int((n_years-window)*year), int(n_years*year), int(Noutputs))

    sim.move_to_com()
    ps = sim.particles

    # Add information about the particle beta
    for i in range(n):
        ps[i+n_active].params["beta"] = beta[i]

    # Initialize output
    xy = np.zeros((int(Noutputs), n, 5))
    oribtal_elements = np.zeros((n_years-window, n, 5))


    # Save particle parameters
    np.save(f"output/{subdir}/beta{k}.npy", beta)
    np.save(f"output/{subdir}/mass{k}.npy", mass)

    # Simulate for age-2 years
    for i in tqdm(range(int((n_years-window)))):

        # Remove close particles particles with small a or just within 0.01 AU of sun
        for j, p in enumerate(sim.particles[n_active:]):
            h = p.hash
            o = p.calculate_orbit(primary = ps[0])
            if sim.particles[0] ** p < .01 or o.a < MIN_A: 
                try:
                    sim.remove(hash=h)
                    n_escaped +=1
                    print(f"Number escaped: {n_escaped} d:{sim.particles[0]**p}, a:{o.a}, beta:{beta[j]}", file=sys.stderr)
                except: pass

        #Remove far particles
        escaped = True
        while(escaped == True):
            try:
                sim.integrate(int(i*year))

                escaped = False
            except:
                n_escaped += 1
                print(f"Number escaped: {n_escaped}", file=sys.stderr)
                for j in range(sim.N):
                    p = sim.particles[j]
                    d2 = p.x*p.x + p.y*p.y + p.z*p.z
                    if (d2>sim.exit_max_distance**2):
                        index=p.hash # cache index rather than remove here since our loop would go beyond end of particles array
                print(index, file = sys.stderr)
                try: sim.remove(hash=index)
                except: escaped = False
        for j in range(n):
            try:
                p = sim.particles[f"{j}"]
                o = p.calculate_orbit(primary = ps[0])
                oribtal_elements[i][j] = [o.a, o.e, o.inc, o.omega, o.Omega]
            except Exception as e:
                print(f"Error in orbital elements: {e}", file=sys.stderr, flush=True)
                oribtal_elements[i][j] = [np.nan, np.nan, np.nan, np.nan, np.nan]


    # Save particle states
    np.save(f"output/{subdir}/elements{k}.npy", oribtal_elements)
    sim.save(f"output/{subdir}/sim{k}.bin")
    print(f"Elements should be saved for run {k}", file=sys.stderr)
    print(oribtal_elements, file=sys.stderr)


    #Take lots of samples in the last ~2 years of orbit
    for i, time in enumerate(tqdm(times)):

            #Remove far particles
        escaped = True
        while(escaped == True):
            try:
                sim.integrate(time)
                escaped = False
            except:
                n_escaped += 1
                print(f"Number escaped: {n_escaped}", file=sys.stderr)
                for j in range(sim.N):
                    p = sim.particles[j]
                    d2 = p.x*p.x + p.y*p.y + p.z*p.z
                    if d2>sim.exit_max_distance**2:
                        index=p.hash # cache index rather than remove here since our loop would go beyond end of particles array
                print(index, file=sys.stderr)
                try: sim.remove(hash=index)
                except: escaped = False


        # Generate output
        for j in range(n):
            try:
                p = sim.particles[f"{j}"]
                o = p.calculate_orbit(primary = ps[0])
                if ((model % 3) == 2):
                    xy[i][j] = [p.x, p.y, p.z, t_and_r[str(j)][0], t_and_r[str(j)][1]]
                else:
                    xy[i][j] = [p.x, p.y, p.z, o.a, o.e]
            except Exception as e:
                print(f"Exception: {e}")
                xy[i][j] = [np.nan, np.nan, np.nan, np.nan, np.nan]

    # Save output
    np.save(f"output/{subdir}/particles{k}.npy", xy)
    print(xy, file=sys.stderr)
    print(k, file=sys.stderr)