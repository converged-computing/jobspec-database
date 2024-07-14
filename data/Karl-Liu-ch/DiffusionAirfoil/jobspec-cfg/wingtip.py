import numpy as np
import sys
import os
from xfoil import XFoil
from xfoil.model import Airfoil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import argparse
parser = argparse.ArgumentParser(description="DiffusionAirfoil")
parser.add_argument('--method', type=str, default='1d')
opt = parser.parse_args()
from utils import *
import gc

def evaluate(airfoil, cl = 0.65, Re1 = 5.8e4, Re2 = 4e5, lamda = 3, return_CL_CD=False, check_thickness = True):
        
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
        airfoil = setupflap(airfoil, theta=-2)
        airfoil = interpolate(airfoil, 300, 3)
        CD, _ = evalpreset(airfoil, Re=Re2)
        i = 0
        while CD < 0.004 and (not np.isnan(CD)) and i < 2:
            i += 1
            print(not np.isnan(CD), CD)
            airfoil = interpolate(airfoil, 200 + i * 100, 3)
            CD, _ = evalpreset(airfoil, Re=Re2 + i * 100)
            print(CD)
        if i >= 2:
            CD = np.nan
            
        airfoil = setflap(airfoil, theta=2)
        perf, _, cd = evalperf(airfoil, cl = cl, Re = Re1)
        R = cd + CD * lamda
        if perf < -100 or perf > 300 or cd < 1e-3:
            perf = np.nan
        elif not np.isnan(perf):
            print('Successful: CL/CD={:.4f}, R={}'.format(perf, R))
            
    if return_CL_CD:
        return perf, cl.max(), cd.min()
    else:
        return perf, CD, airfoil, R

if __name__ == "__main__":
    LAMBDA = 5
    cd_BL = 0.0078
    cd_best = cd_BL
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

    try:
        log = np.loadtxt(f'results/{name}_tipsimlog.txt')
        i = int(log[0])
        k = int(log[1])
        m = int(log[2])
    except:
        m = 0
        i = 0
        k = 0

    print(f'i: {i}, k: {k}, m: {m}')
    while i < 1000:
        f = open(f'results/{name}_tipsimperf.log', 'a')
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
            xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
            airfoil[:,0] = xhat
            airfoil[:,1] = yhat
            if cal_thickness(airfoil) > 0.055 or cal_thickness(airfoil) < 0.45:
                airfoil[:,1] = airfoil[:,1] * 0.055 / cal_thickness(airfoil)
            af = setupflap(airfoil, -3, 0.6)
            if cal_thickness_percent(af) > 24.1:
                airfoil = set_thickness_pose(airfoil, cal_thickness_percent(airfoil)*(1-(cal_thickness_percent(af)-23)/cal_thickness_percent(af)))
                af = setupflap(airfoil, -3, 0.6)
            cd, _ = evalpreset(af, Re=1.4e5)
            if cd < 0.006:
                cd, _ = evalpreset(af, Re=1.41e5)
            thickness = cal_thickness(af)
            print('CD: ', cd, 'Thickness: ', thickness, 'Max thickness pose: ', cal_thickness_percent(af))
            if cd == np.nan or check_backpoint(af) != 0 or cal_thickness_percent(af) > 24.1:
                pass
            elif cd < cd_BL:
                if cd < cd_best:
                    np.savetxt(f'results/tip{name}F.dat', af, header=f'{name}F', comments="")
                    np.savetxt(f'results/tip{name}.dat', airfoil, header=f'{name}', comments="")
                mm = str(m).zfill(3)
                np.savetxt(f'samples/tip{name}_{mm}.dat', airfoil, header=f'{name}_{mm}', comments="")
                np.savetxt(f'samples/tip{name}_{mm}F.dat', af, header=f'{name}_{mm}F', comments="")
                f = open(f'results/{name}_tipsimperf.log', 'a')
                f.write(f'cd: {cd}, m: {mm}, thickness: {thickness}, path: samples/tip{name}_{mm}.dat\n')
                f.close()
                m += 1
                del f
            k += 1
            log = np.array([i, k, m])
            np.savetxt(f'results/{name}_tipsimlog.txt', log)
            del airfoil
            gc.collect()
        k = 0
        i += 1
        del airfoils
        gc.collect()