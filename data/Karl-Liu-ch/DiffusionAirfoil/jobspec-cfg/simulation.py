import numpy as np
import sys
import os
from xfoil import XFoil
from xfoil.model import Airfoil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from option import opt
from utils import *
import gc
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_id
def evaluate(airfoil, mass = 0.22, diameter = 0.135, area = 0.194, Re2 = 4e5, lamda = 5, thickness = 0.058, return_CL_CD=False, check_thickness = True, modify_thickness = False):
    if detect_intersect(airfoil):
        # print('Unsuccessful: Self-intersecting!')
        perf = np.nan
        R = np.nan
        CD = np.nan
        
    elif (cal_thickness(airfoil) < 0.06 or cal_thickness(airfoil) > 0.09) and check_thickness:
        # print('Unsuccessful: Too thin!')
        perf = np.nan
        R = np.nan
        CD = np.nan
    
    elif np.abs(airfoil[0,0]-airfoil[-1,0]) > 0.01 or np.abs(airfoil[0,1]-airfoil[-1,1]) > 0.01:
        # print('Unsuccessful:', (airfoil[0,0],airfoil[-1,0]), (airfoil[0,1],airfoil[-1,1]))
        perf = np.nan
        R = np.nan
        CD = np.nan
        
    else:
        af = np.copy(airfoil)
        airfoil = setupflap(airfoil, theta=-2)
        if modify_thickness:
            airfoil[:,1] = airfoil[:,1] * thickness / cal_thickness(airfoil)
        airfoil = interpolate(airfoil, 400, 3)
        CD, _ = evalpreset(airfoil, Re=Re2)
        i = 0
        while CD < 0.004 and (not np.isnan(CD)) and i < 2:
            i += 1
            print(not np.isnan(CD), CD)
            airfoil = interpolate(airfoil, 400 + i * 10, 3)
            CD, _ = evalpreset(airfoil, Re=Re2 + i * 100)
            print(CD)
        if i >= 2:
            CD = np.nan
            
        airfoil = setflap(airfoil, theta=2)
        airfoil = interpolate(airfoil, 400, 3)
        perf = type2_simu(airfoil, mass, diameter, area)
        cd = 0.65 / perf
        R = cd + CD * lamda
        if R < 0.015 + 0.004852138459682465 * lamda and perf < 37:
            print(f'R: {R}, perf: {perf}, change reynolds')
            re2 = Re2 + 1000
            perf, CD, airfoil, R = evaluate(af, mass = mass, diameter = diameter, area = area, Re2 = re2, lamda = lamda, return_CL_CD=return_CL_CD, check_thickness = check_thickness)
        if perf < -100 or perf > 300 or cd < 1e-3:
            perf = np.nan
        elif not np.isnan(perf):
            print('Successful: CL/CD={:.4f}, R={}, launch cd: {}'.format(perf, R, CD))
            
    if return_CL_CD:
        return perf, cl.max(), cd.min()
    else:
        return perf, CD, airfoil, R

if __name__ == "__main__":
    mass = 0.22
    diameter = 0.135
    area = 0.194
    LAMBDA = 5
    perf_BL, R_BL = cal_baseline(lamda=LAMBDA, mass=mass, diameter = diameter, area = area)
    CD_BL = 0.004852138459682465
    cl = 0.65
    print(perf_BL, R_BL)
    best_perf=perf_BL
    best_airfoil = None
    if opt.method == '2d':
        name = 'Airfoils2D'
        airfoilpath = '/work3/s212645/DiffusionAirfoil/Airfoils/'
    elif opt.method == '1d':
        name = 'Airfoils1D'
        airfoilpath = '/work3/s212645/DiffusionAirfoil/'+name+'/'
    elif opt.method == 'bezier':
        name = 'Airfoilsbezier'
        airfoilpath = '/work3/s212645/BezierGANPytorch/Airfoils/'
    else:
        airfoilpath = opt.path
        name = opt.path.split('/')[3]

    try:
        log = np.loadtxt(f'results/{name}_simlog.txt')
        i = int(log[0])
        k = int(log[1])
        m = int(log[2])
    except:
        m = 0
        i = 0
        k = 0

    print(f'i: {i}, k: {k}, m: {m}')
    while i < 1000:
        f = open(f'results/{name}_simperf.log', 'a')
        f.write(f'files: {i}\n')
        f.close()
        del f
        num = str(i).zfill(3)
        airfoils = np.load(airfoilpath+num+'.npy')
        airfoils = delete_intersect(airfoils)
        while k < airfoils.shape[0]:
            airfoil = airfoils[k,:,:]
            airfoil = derotate(airfoil)
            airfoil = Normalize(airfoil)
            # if opt.method == '1d' or opt.method == '2d':
            xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
            airfoil[:,0] = xhat
            airfoil[:,1] = yhat
            if cal_thickness(airfoil) > 0.07 or cal_thickness(airfoil) < 0.06:
                airfoil[:,1] = airfoil[:,1] * 0.06 / cal_thickness(airfoil)
            successful = False
            while not successful:
                try:
                    perf, CD, af, R = evaluate(airfoil, mass=mass, diameter=diameter, area=area, lamda=LAMBDA, check_thickness=False, modify_thickness=True)
                    successful = True
                except Exception as e:
                    print(e)
                    break
            if np.isnan(perf):
                pass
            elif R < R_BL:
                mm = str(m).zfill(3)
                np.savetxt(f'samples/{name}_{mm}.dat', airfoil, header=f'{name}_{mm}', comments="")
                np.savetxt(f'samples/{name}_{mm}F.dat', af, header=f'{name}_{mm}F', comments="")
                f = open(f'results/{name}_simperf.log', 'a')
                f.write(f'perf: {perf}, R: {R}, launch cd: {CD}, m: {mm}, path: samples/{name}_{mm}.dat\n')
                f.close()
                m += 1
                del f
            k += 1
            log = np.array([i, k, m])
            np.savetxt(f'results/{name}_simlog.txt', log)
            del airfoil
            gc.collect()
        k = 0
        i += 1
        del airfoils
        gc.collect()