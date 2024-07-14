from __future__ import print_function
import numpy as np
from presto import presto
import presto.ppgplot as ppgplot
import time
from presto.Pgplot import pgpalette
import sys

p_middle = float(sys.argv[1])
file_name = sys.argv[2]
out = sys.argv[3]

N = 16777216 #2**14
dat_size = N
time_res = 64e-6 # in seconds
T_obs = (dat_size*time_res)/60 # in minutes is equal to 17.895 minutes
freq_axis = np.fft.rfftfreq(dat_size, d=64e-6)
freq_res = 1/(T_obs*60)

f_middle = 1/p_middle
r = f_middle/freq_res
rint = np.floor(r)

numbetween = 256
dr = 1.0/numbetween
dz = 4.0/numbetween
zlow = -100.0
zhigh = 100.0
rlow_rel = -5.0
rhigh_rel = 5.0
nz = int((zhigh-zlow)/dz)
nr = int((rhigh_rel-rlow_rel)/dr)

with open(file_name, 'rb') as f:
    dat = np.frombuffer(f.read(), dtype=np.float32)
    dat = dat[:N]

ft = presto.rfft(dat)

a = time.clock()
vol = presto.ffdot_plane(ft, rint-nr/2*dr, dr, nr,
                         zlow, dz, nz)

print("First vol took %.3f s" % (time.clock()-a))
# a = time.clock()
# vol = presto.fdotdot_vol(ft, rint-np/2*dr, dr, np,
#                          0.0-np/2*dz, dz, np,
#                          0.0-np/2*dw, dw, np)
# print("Second jerk vol took %.3f s" % (time.clock()-a))
# zarray = np.arange(zlow, zhigh, dz)
# rarray = np.arange(rint-nr/2*dr, rint+nr/2*dr, dr)
pvol = presto.spectralpower(vol.flat)
#theo_max_pow = N**2.0/4.0
#frp = max(pvol) / theo_max_pow # Fraction of recovered power
#print("Fraction of recovered signal power = %f" % frp)
# [maxpow, rmax, zmax, rd] = presto.maximize_rz(ft, r+np.random.standard_normal(1)[0]/5.0,
#                                        z+np.random.standard_normal(1)[0], norm=1.0)
# print(r, rmax, z, zmax, maxpow)
pvol.shape = (nz, nr)

np.save(out+'pvol.npy', pvol)
# np.save('zarray.npy', zarray)
# np.save('rarray.npy', rarray)