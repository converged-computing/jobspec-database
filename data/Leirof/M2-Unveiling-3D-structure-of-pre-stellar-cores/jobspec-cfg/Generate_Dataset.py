import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from scipy.signal import argrelextrema
import subprocess
from LRFutils import progress
from astropy import units as u
from astropy import constants as const



#==============================================================================
# INIT
#==============================================================================



# Configuration ---------------------------------------------------------------

N = 64 # resolution in pixel
space_range, space_step = np.linspace(-25, 25, N, endpoint=True, retstep=True) # Space range (1D but serve as base for 3D)

space_range *= u.pc # Space range in parsec
space_step  *= u.pc # Space step in parsec

channels = 128 # number of velocity channels

T = 10 * u.K # Kinetic temperature [K]

CO_fractional_abundance  = 1e-4 # particle / hydrogen atom
N2H_fractional_abundance = 1e-7 # particle / hydrogen atom

n_List = np.logspace(3, 6, 10, endpoint=True)
n_List *= u.cm**-3 # Density from 10^3 to 10^6 hydrogen atom per cm^-3

r_List = np.logspace(-2, 0, 10, endpoint=True)
r_List *= u.pc # Core radius from 0.02 to 1 parsec

p_List, dp = np.linspace(1.5,  2.5, 10, endpoint=True, retstep=True) # Sharpness of the plummer profile from 1.5 to 2.5
p_list = np.flip(p_List) * u.dimensionless_unscaled
dp *= u.dimensionless_unscaled

debug = False

# Environment -----------------------------------------------------------------

X, Y, Z = np.meshgrid(space_range, space_range, space_range)
R = np.sqrt(X**2 + Y**2 + Z**2)

# Global functions ------------------------------------------------------------

def plummer(M:float, d:float, r:float, p:float) -> float:

    return 3 * M /(4 * np.pi * r**3) * (1 + d**p / r**p)**(-5/2)



#==============================================================================
# LOC
#==============================================================================



# Gas cloud description -------------------------------------------------------

def write_LOC_cloud(N:int, density_cube:ndarray[float], fractional_abundance:float):

    header = np.array([N, N, N,], dtype=np.int32)
    model = np.zeros((N, N, N, 7), dtype=np.float32)

    model[:,:,:,0] = density_cube # cloud density
    model[:,:,:,1] = T # kinetic temperature
    model[:,:,:,2] = 0 # microturbulence
    model[:,:,:,3] = 0 # macroscopic velocity (x)
    model[:,:,:,4] = 0 # macroscopic velocity (y)
    model[:,:,:,5] = 0 # macroscopic velocity (z)
    model[:,:,:,6] = fractional_abundance # fractional abundance

    with open(f"data/LOC/input_cloud.bin", "wb") as file:
        header.tofile(file)
        model.tofile(file)

# Initialisation file ---------------------------------------------------------

def write_LOC_config(N:int, molecule:str, channels:int):

    config = f"""
        cloud          data/LOC/input_cloud.bin               #  cloud defined on a Cartesian grid
        molecule       data/LOC/lamda_database/{molecule}.dat #  name of the Lamda file

        points         {N} {N}             #  number of pixels in the output maps
        distance       100.0               #  cloud distance [pc]
        angle          3.0                 #  model cell size  [arcsec]
        density        1.0e4               #  scaling of densities in the cloud file
        temperature    1.0                 #  scaling of Tkin values in the cloud file
        fraction       1.0e-4              #  scaling o the fraction abundances
        velocity       0.5                 #  scaling of the velocity
        sigma          0.5                 #  scaling o the microturbulence
        isotropic      2.73                #  Tbg for the background radiation field
        levels         10                  #  number of energy levels in the calculations
        uppermost      3                   #  uppermost level tracked for convergence
        iterations     5                   #  number of iterations
        nside          2                   #  NSIDE parameter determining the number of rays (angles)
        direction      90 0.001            #  theta, phi [deg] for the direction towards the observer
        grid           3.0                 #  map pixel size [arcsec]
        spectra        1 0  2 1            #  spectra written for transitions == upper lower (pair of integers)
        transitions    1 0  2 1            #  Tex saved for transitions (upper and lower levels of each transition)
        bandwidth      1.0                 #  bandwidth in the calculations [km/s]
        channels       {channels}          #  number of velocity channels
        prefix         data/LOC/output/res #  prefix for output files
        load           level_pop.load      #  load level populations
        save           level_pop.save      #  save level populations
        stop          -1.0                 #  stop if level-populations change below a threshold
        gpu            1                   #  gpu>0 => use the first available gpu
        """.replace("\n        ","\n")

    with open("data/LOC/input_config.ini", "w") as file:
        file.write(config)

# LOC output reading functions ------------------------------------------------

def LOC_read_spectra_3D(filename:str) -> tuple[ndarray,ndarray]:

    with open(filename, 'rb') as fp:
        NRA, NDE, NCHN  =  fromfile(fp, np.int32, 3)
        V0, DV          =  fromfile(fp, np.float32, 2)
        SPE             =  fromfile(fp, np.float32).reshape(NRA, NDE,2+NCHN)
        OFF             =  SPE[:,:,0:2].copy()
        SPE             =  SPE[:,:,2:]

    return V0+arange(NCHN)*DV, SPE
    
    
def LOC_read_Tex_3D(filename):

    fp    =  open(filename, 'rb')
    NX, NY, NZ, dummy  =  fromfile(fp, np.int32, 4)
    TEX                =  fromfile(fp, np.float32).reshape(NZ, NY, NX)
    fp.close()
    return TEX 

# Run LOC ---------------------------------------------------------------------

def run_LOC():
    global debug

    subprocess.run(["python","src/LOC/LOC_OT.py","data/LOC/input_config.ini"], capture_output=debug)

    def move_file(src, dst):
        try:
            os.rename(src, dst)
        except:
            pass

    move_file("CO.dump", "data/LOC/output/CO.dump")
    move_file("N2H+.dump", "data/LOC/output/N2H+.dump")
    move_file("gauss_py.dat", "data/LOC/output/gauss_py")
    move_file("level_pop", "data/LOC/output/level_pop.save")



#==============================================================================
# SOC
#==============================================================================



# Dust cloud description -----------------------------------------------------

def write_SOC_cloud(N:int, density_cube:ndarray[float]):

    header = np.array([N, N, N, 1, N**3, N**3], dtype=np.int32)

    with open(f"data/SOC/input_cloud.bin", "wb") as file:
        header.tofile(file)
        density_cube.astype(np.float32).tofile(file)

# Initialisation file ---------------------------------------------------------

def write_SOC_config(N:int, space_step:float):

    config = f"""
        gridlength      {space_step}                      # root grid cells have a size of 0.01 pc each
        cloud           data/SOC/input_cloud.bin  # density field (reference to a file)
        mapping         {N} {N} 1.0               # output 64x64 pixels, pixel size == root-grid cell size
        density         1.0                       # scale values read from tmp.cloud
        seed           -1.0                       # random seed for random numbers
        directions      0.0  0.0                  # observer in direction (theta,phi)
        optical         data/SOC/aSilx.dust       # dust optical parameters
        dsc             data/SOC/aSilx.dsc  2500  # dust scattering function
        bgpackets       999999                    # photon packages simulated from the background
        background      data/SOC/bg_intensity.bin # background intensity at the
        iterations      1                         # one iteration is enough
        prefix          data/LOC/output/res       # prefix for output files
        absorbed        absorbed.save             # save absorptions to a file
        emitted         emitted.save              # save dust emission to a file
        noabsorbed                                # actually, we integrate absorptions on the fly and skip the *file*
        temperature     temperature.save          # save dust temperatures
        device          g                         # run calculations on a GPU
        CLT                                       # temperature caclulations done on the device
        CLE                                       # emission calculations done on the device
        """.replace("\n        ","\n")

    with open("data/SOC/input_config.ini", "w") as file:
        file.write(config)

# Run SOC ---------------------------------------------------------------------

def run_SOC():
    global debug

    subprocess.run(["python","src/SOC/ASOC.py","data/SOC/input_config.ini"], capture_output=debug)

    def move_file(src, dst):
        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))
        try:
            os.rename(src, dst)
        except:
            pass

    move_file("absorbed.save", "data/SOC/output/") 
    move_file("emitted.save", "data/SOC/output/")   
    move_file("map_dir_00.bin", "data/SOC/output/")
    move_file("temperature.save", "data/SOC/output/")
    move_file("packet.info", "data/SOC/output/")

# SOC output reading functions ------------------------------------------------

def read_SOC_output():
    freq     = loadtxt('data/SOC/freq.dat')            # read the frequencies
    nfreq    = len(freq)                      # ... number of frequencies

    with open('data/SOC/output/map_dir_00.bin', 'rb') as fp:  # open the surface-brightness file
        dims     = fromfile(fp, np.int32, 2)                     # read map dimensions in pixels
        S        = fromfile(fp, np.float32).reshape(nfreq, dims[0], dims[1]) # read intensities
        S       *= 1.0e-5                                     # convert surface brightness from Jy/sr to MJy/sr

    with open('data/SOC/output/temperature.save', 'rb') as fp:
        NX, NY, NZ, _, _, _ = fromfile(fp, np.int32, 6)
        T = fromfile(fp, np.float32).reshape(NZ, NY, NX)

    return freq, S, T

# Plot ------------------------------------------------------------------------

def plot_SOC():
    freq, S = read_SOC_output()
    figure(1, figsize=(8,2.8))
    ifreq    = argmin(abs(freq-2.9979e8/250.0e-6)) # Choose frequency closest to 250um
    ax = subplot(121)
    imshow(S[ifreq,:,:])                      # plot the 250um surface brightness map
    title("Surface brightness")
    colorbar()
    text(1.34, 0.5, r'$S_{\nu} \/ \/ \rm (MJy \/ sr^{-1})$', transform=ax.transAxes,
    va='center', rotation=90)
    fp       = open('data/SOC/output/temperature.save', 'rb')            # open the file with dust temperatures
    NX, NY, NZ, LEVELS, CELLS, CELLS_LEVEL_1 = fromfile(fp, np.int32, 6)
    T        = fromfile(fp, np.float32).reshape(NZ, NY, NX) # T is simply a 64x64x64 cell cube
    ax = subplot(122)
    imshow(T[NZ//2, :, :])                    # plot cross section through the model centre
    title("Dust temperature")
    colorbar()
    text(1.34, 0.5, r'$T_{\rm dust} \/ \/ \rm (K)$', transform=ax.transAxes, va='center', rotation=90)
    plt.savefig("data/SOC/output/plot.png", dpi=300)



#==============================================================================
# GENERATE DATASET
#==============================================================================


# Limit of max outer density to get a good SOC image
# Above this limit, the heated part of the gaz get out of the observation frame

# max_outer_layer_density = 5e-2 # H atom . cm^-3
# print("Max outer layer density: ", max_outer_layer_density, "H atom . cm^-3")

bar = progress.Bar(len(n_List) * len(r_List) * len(p_List))

if not os.path.isdir("data/dataset"):
    os.mkdir("data/dataset")

fig = plt.figure(figsize=(15,5))
ax_nr = fig.add_subplot(131)
ax_np = fig.add_subplot(132)
ax_rp = fig.add_subplot(133)

for i in range(len(n_List)): 
    for j in range(len(r_List)): 
        for k,p in enumerate(p_List):
            
            if i==0:
                n_H = (np.random.rand() * np.abs(n_List[i] - n_List[i+1])) + min(n_List[i], n_List[i+1])
            elif 0 < i < len(n_List) - 1:
                n_H = (np.random.rand() * np.abs(n_List[i-1] - n_List[i+1])) + min(n_List[i-1], n_List[i+1])
            elif i == len(n_List) - 1:
                n_H = (np.random.rand() * np.abs(n_List[i-1] - n_List[i])) + min(n_List[i-1], n_List[i])

            if j==0:
                r = (np.random.rand() * np.abs(r_List[j] - r_List[j+1])) + min(r_List[j], r_List[j+1])
            elif 0 < j < len(r_List) - 1:
                r = (np.random.rand() * np.abs(r_List[j-1] - r_List[j+1])) + min(r_List[j-1], r_List[j+1])
            elif j == len(r_List) - 1:
                r = (np.random.rand() * np.abs(r_List[j-1] - r_List[j])) + min(r_List[j-1], r_List[j])

            p = p + np.random.rand() * dp

            ax_nr.scatter(n_H, r, color="black", s=0.5)
            ax_np.scatter(n_H, p, color="black", s=0.5)
            ax_rp.scatter(r, p, color="black", s=0.5)

            # Updating progress bar
            n_simu = i*len(r_List)*len(p_List) + j*len(p_List) + k
            bar(n_simu)


            # Generate cloud --------------------------------------------------

            profile = plummer(1*u.Msun, R, r, p)
            density_cube = n_H * profile / np.max(profile)

            # Computing mass --------------------------------------------------
            
            hydrogen_mass = const.m_p # mass of a proton
            mu = 2.8
            
            dV = (space_step)**3
            dV = dV.to(u.cm**3)

            n_tot = np.sum(density_cube) * dV

            mass = mu * hydrogen_mass * n_tot

            mass = mass.to(u.Msun)

            simu_name = f"data/dataset/{n_simu}_n={n_H.value:.2f}_r={r.value:.2f}_p={p.value:.2f}_m={mass.value:.2f}"

            # Avoid recomputing done simulations
            if os.path.isfile(f"{simu_name}.npz"):
                continue

            # CO simulation ---------------------------------------------------

            write_LOC_cloud(N, density_cube, CO_fractional_abundance)
            write_LOC_config(N, "CO", channels)
            run_LOC()

            CO_v, CO_cube = LOC_read_spectra_3D("data/LOC/output/res_CO_01-00.spe")
            
            # N2H+ simulation -------------------------------------------------

            write_LOC_cloud(N, density_cube, N2H_fractional_abundance)
            write_LOC_config(N, "N2H+", channels)
            run_LOC()

            N2H_v, N2H_cube = LOC_read_spectra_3D("data/LOC/output/res_N2H+_01-00.spe")

            # Dust simulation -------------------------------------------------

            write_SOC_cloud(N, density_cube.value)
            write_SOC_config(N, space_step.value)
            run_SOC()

            freq, S, T = read_SOC_output()
            # ifreq    = argmin(abs(freq-2.9979e8/250.0e-6)) # Choose frequency closest to 250um
            dust_cube = S # S[ifreq,:,:]

            # Save data -------------------------------------------------------

            # plt.figure()
            # plt.imshow(dust_image)
            # cbar = plt.colorbar()
            # cbar.set_label(r"Surface brightness (MJy . sr$^{-1}$)")
            # plt.savefig(f"data/dataset/{n_simu}_"
            #             + f"n={n_H.value:.2f}_"
            #             + f"r={r.value:.2f}_"
            #             + f"p={p:.2f}_"
            #             + f"m={float(mass.value):.2f}_Dust-map.png", dpi=300)
            # plt.close()

            np.savez_compressed(f"{simu_name}.npz",
                CO_cube = CO_cube,
                N2H_cube = N2H_cube,
                CO_v = CO_v,
                N2H_v = N2H_v,
                dust_freq = freq,
                dust_cube = dust_cube,
                dust_temperature = T,
                density_cube = density_cube.value,
                n_H = n_H.value,
                r = r.value,
                p = p.value,
                mass = mass.value,
                space_range = space_range.value,
                space_step = space_step.value,
                N = N,
            )

            if debug:
                break
        
        if debug:
            break
    
    if debug:
        break

ax_nr.set_xlabel(r"$n_H$ (H atom . cm$^{-3}$)")
ax_nr.set_ylabel(r"$r$ (pc)")
ax_nr.set_xscale('log')
ax_nr.set_yscale('log')

ax_np.set_xlabel(r"$n_H$ (H atom . cm$^{-3}$)")
ax_np.set_ylabel(r"$p$")
ax_np.set_xscale('log')

ax_rp.set_xlabel(r"$r$ (pc)")
ax_rp.set_ylabel(r"$p$")
ax_rp.set_xscale('log')

ax_np.set_title("Generation's parameter space")
fig.savefig("data/dataset_parameter_space.png", dpi=300)

# End progress bar
bar(len(n_List) * len(r_List) * len(p_List))