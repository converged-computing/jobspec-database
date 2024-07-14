import numpy as np
import math
import os
import sys
from joblib import Parallel, delayed
# functions to calculate z and a_max
def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)

sims = 30000
myexecute(f'echo "{sys.argv}"')
job_id = sys.argv[1]
output_version = 'v'+str(sys.argv[2])
G = 6.67408e-11
c = 2.99792458e8
M_SUN = 2.0e30
T_sun = 4.925490947e-6

def a_to_pdot(P_s, acc_ms2):
    return P_s * acc_ms2 /c

def a_from_pdot(P_s, pdot):
    return pdot * c / P_s

def period_modified(p0,pdot,no_of_samples,tsamp,fft_size):
    if (fft_size==0.0):
        return p0 - pdot*float(1<<(no_of_samples.bit_length()-1))*tsamp/2
    else:
        return p0 - pdot*float(fft_size)*tsamp/2

def calculate_spin(f=None, fdot=None, p=None, pdot=None):
    # calculate p and pdot from f and fdot
    if f is not None and fdot is not None:
        p = 1 / f
        pdot = -fdot / (f**2)
    # calculate f and fdot from p and pdot
    elif p is not None and pdot is not None:
        f = 1 / p
        fdot = -pdot * (p**2)
    else:
        raise ValueError("Either (f, fdot) or (p, pdot) must be provided")
        
    return f, fdot, p, pdot

def return_los_time_velocity_acceleration(  inclination,
                                            orbital_period,
                                            obs_time,
                                            initial_orbital_phase,
                                            mass_companion,
                                            eccentricity = 0.0,
                                            longitude_periastron = 0.0,
                                            mass_pulsar = 1.4,
                                            n_samples = 2**8,
                                            ):
    '''
    inclination: inclination angle in degrees
    orbital_period: orbital period in hours
    obs_time: observation time in hours
    initial_orbital_phase: orbital phase at the beginning of the observation (0 to 1)
    mass_companion: mass of the companion in solar masses
    eccentricity: eccentricity of the orbit
    mass_pulsar: mass of the pulsar in solar masses
    longitude_periastron: longitude of periastron in degrees
    n_samples: number of samples in the observation
    '''
    fake_inclination = inclination
    fake_orbital_period =  orbital_period * 3600
    fake_initial_orbital_phase = initial_orbital_phase
    fake_eccentricity = eccentricity
    fake_longitude_periastron = longitude_periastron * np.pi/180
    mass_companion = mass_companion
    mass_pulsar = mass_pulsar


    observation_time = obs_time * 3600
    #observation_time = fake_orbital_period

    n_samples = n_samples
    time = np.linspace(0, observation_time, n_samples)

    incl = fake_inclination * (np.pi/180)
    omegaB = 2.0 * np.pi/fake_orbital_period
    t0 = fake_initial_orbital_phase * fake_orbital_period
    massFunction = math.pow((mass_companion * np.sin(incl)), 3)/math.pow((mass_companion + mass_pulsar), 2)
    asini = math.pow(( M_SUN * massFunction * G * fake_orbital_period * \
                    fake_orbital_period / (4.0 * np.pi * np.pi)), 0.333333333333)

    meanAnomaly = omegaB * (time - t0)
    eccentricAnomaly = meanAnomaly + fake_eccentricity * np.sin(meanAnomaly) * \
    (1.0 + fake_eccentricity * np.cos(meanAnomaly))

    du = np.ones(n_samples)
    for i in range(len(du)):
        while(abs(du[i]) > 1.0e-13):
        
            du[i] = (meanAnomaly[i] - (eccentricAnomaly[i] - fake_eccentricity * \
                                np.sin(eccentricAnomaly[i])))/(1.0 - fake_eccentricity * np.cos(eccentricAnomaly[i]))

            eccentricAnomaly[i] += du[i]


    trueAnomaly = 2.0 * np.arctan(math.sqrt((1.0 + fake_eccentricity)/(1.0 - fake_eccentricity)) \
                                * np.tan(eccentricAnomaly/2.0))

    los_velocity = omegaB * (asini / (np.sqrt(1.0 - math.pow(fake_eccentricity, 2)))) * \
            (np.cos(fake_longitude_periastron + trueAnomaly) + fake_eccentricity * np.cos(fake_longitude_periastron))

    los_acceleration = (-1*(omegaB*omegaB)) * (asini / math.pow(1 - math.pow(fake_eccentricity, 2), 2)) * \
    np.power((1 + (fake_eccentricity * np.cos(trueAnomaly))), 2) * np.sin(fake_longitude_periastron + trueAnomaly)

    # los_jerk = (-1*(omegaB*omegaB*omegaB)) * (asini / math.pow(1 - math.pow(fake_eccentricity, 2), 3.5)) * \
    #     np.power((1 + (fake_eccentricity * np.cos(trueAnomaly))), 3) * \
    #     (np.cos(fake_longitude_periastron + trueAnomaly) + fake_eccentricity * np.cos(fake_longitude_periastron) - 3 * \
    #     fake_eccentricity * np.sin(fake_longitude_periastron + trueAnomaly) * np.sin(trueAnomaly))

    return time, los_velocity, los_acceleration

def mean_acceleration_from_los(los_acceleration):
    return np.mean(los_acceleration)

def middle_period(pApp):
    return pApp[int(len(pApp)//2)]

def return_spin_period_array(pRest,los_velocity):
    return pRest/(1.0 - (los_velocity / c))

def mean_acceleration_from_los(los_acceleration):
    return np.mean(los_acceleration)

def middle_period(pApp):
    return pApp[int(len(pApp)//2)]

def a_to_z(T_obs,a,h,P_s):
    T_obs = T_obs*3600
    return T_obs**2*a*h/(P_s*c) 

def P_s_from_z(z,T_obs,a,h):
    T_obs = T_obs*3600
    return (T_obs**2*a*h)/(z*c) 

def return_params(i,P_orb,m_c,P_s,phase,T_obs,h,e,lp):
    time, los_velocity, los_acceleration = return_los_time_velocity_acceleration(
                                            inclination = i,
                                            orbital_period = P_orb,
                                            obs_time = T_obs/60,
                                            initial_orbital_phase = phase,
                                            mass_companion = m_c,
                                            eccentricity = e,
                                            longitude_periastron = lp,
                                            n_samples = 2**10,
                                            )
    pApp = return_spin_period_array(P_s,los_velocity)
    a = mean_acceleration_from_los(los_acceleration)
    p_middle = middle_period(pApp)
    z = a_to_z(T_obs/60,a,h,p_middle)
 
    return p_middle, z
#,los_acceleration,los_velocity,pApp,time
fft_size = 16777216
time_res = 64e-6 # in seconds
T_obs = (fft_size*time_res)/60 # in minutes is equal to 17.895 minutes
freq_axis = np.fft.rfftfreq(fft_size, d=64e-6)
freq_res = 1/(T_obs*60)

np.random.seed(42)
#size = 2000

#period_array = np.random.uniform(0.001, 0.02, sims) #Uniform period in ms
snr_array = np.random.uniform(0.3, 0.8, sims) #signal to noise ratio
width_array = np.random.uniform(10, 30, sims) #width of the pulse in ms
bper_array = np.random.uniform(3, 30, sims) #binary period in hours
binc_array = np.random.uniform(0, 90, sims) #binary inclination angle in degrees
bcmass_array = np.random.uniform(0.1, 1.4, sims) #companion mass in solar masses
bphase_array = np.random.uniform(0, 1, sims) #binary phase
e_array = np.random.uniform(0,1,sims)*0.0 #eccentricity
lp_array = np.random.uniform(0,1,sims)*0.0 #longitude of periastron
z_array = np.zeros_like(bphase_array) 
period_array = np.zeros_like(bphase_array)

def final_z_calc(i):
    time, los_velocity, los_acceleration = return_los_time_velocity_acceleration(
                                            inclination = binc_array[i],
                                            orbital_period = bper_array[i],
                                            obs_time = T_obs/60,
                                            initial_orbital_phase = bphase_array[i],
                                            mass_companion = bcmass_array[i],
                                            eccentricity = e_array[i],
                                            longitude_periastron = lp_array[i],
                                            n_samples = 2**10,
                                            )
    a = mean_acceleration_from_los(los_acceleration)
    p = 0
    while (p < 0.001) or (p > 0.02): #period should be in the range of 1-20 ms
        if a < 0:
            z_temp = np.random.uniform(-5,0)
        else:
            z_temp = np.random.uniform(0,5)
        p = P_s_from_z(z_temp,T_obs/60,a,1)
    # if i%10000 == 0:
    #     myexecute(f'echo "{i}"')    
    myexecute(f'echo "{i},{p},{snr_array[i]},{width_array[i]},{bper_array[i]},{binc_array[i]},{bcmass_array[i]},{bphase_array[i]},{z_temp},{e_array[i]},{lp_array[i]}"')
    return i, p, z_temp

results = Parallel(n_jobs=-1)(delayed(final_z_calc)(i) for i in range(sims))
for i, p, z in results:
    period_array[i] = p
    z_array[i] = z 

meta_data = np.zeros((sims,11),np.float32)
meta_data[:,0] = np.arange(0,sims,1)
meta_data[:,1] = period_array
meta_data[:,2] = snr_array
meta_data[:,3] = width_array
meta_data[:,4] = bper_array
meta_data[:,5] = binc_array
meta_data[:,6] = bcmass_array
meta_data[:,7] = bphase_array
meta_data[:,8] = z_array
meta_data[:,9] = e_array
meta_data[:,10] = lp_array

cur_dir = f'/tmp/Abhinav_DATA{job_id}/'
root_dir = '/hercules/scratch/atya/BinaryML/'

myexecute(f'mkdir -p {cur_dir}')

#save numpy array as csv file
np.savetxt(f'{cur_dir}uniformZ{output_version}.csv',meta_data,delimiter=',',header='ind,period,snr,width,bper,binc,bcmass,bphase,z,e,lp')
#rsync the csv file to the hercules scratch directory
myexecute(f'rsync -avz {cur_dir} {root_dir}meta_data/')