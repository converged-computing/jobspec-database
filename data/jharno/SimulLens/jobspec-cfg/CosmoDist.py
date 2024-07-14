from astropy.cosmology import w0waCDM
from astropy.cosmology import z_at_value
from numpy import savetxt, loadtxt

#print("*** USING PYTHON COSMODIST ***")
cosmo_par_array = loadtxt('cosmo_par.tmp', usecols=(0,1,2,3,4), unpack=True)

Om_m = cosmo_par_array[0]
Om_l = cosmo_par_array[1]
w_de = cosmo_par_array[2]
h_0 = cosmo_par_array[3]
z = cosmo_par_array[4]

cosmo = w0waCDM(H0=h_0*100.0, Om0=Om_m, Ode0=Om_l, w0=w_de, wa=0.0)

#print("***")
#print('cosmo:')
#print(cosmo)
#print("***")

chi_Mpc_over_h = cosmo.comoving_distance(z)*cosmo.h
#print(chi_Mpc_over_h)

#print("*** DONE PYTHON COSMODIST ***")

file_out_str= "./chi_from_python.tmp"
data = [chi_Mpc_over_h.value, Om_m, Om_l, w_de, h_0, z]
#print('data:',data)
savetxt(file_out_str,data, fmt='%.8e')
