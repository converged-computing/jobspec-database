'''
Script to calculate accretion rate: Gamma

for z1=0: delta_t ~ 5Gyrs so z2=0.5, snap=0067

for z1=0.5: delta_t ~4.05Gyrs so z2=1.35, snap=0055

for z1=0.95,  delta_t ~3Gyr so z2=2.25, snap=0032  

for z1=1: delta_t ~3Gyr so z2=2.25, snap=0032

for z1=1.5
'''

import numpy as np
import cat_reader as cat
import unyt
import h5py as h5
import matplotlib.pyplot as plt
import sys
from unyt.array import uconcatenate
import yt
from yt.funcs import get_pbar
from utils import snapshot, paths, data_bin
from snap_reader import snap_data
from collections import defaultdict
from unyt import cm, g, kg, m, G

group_path = "/cosma8/data/dp004/flamingo/Runs/"
res = '1800'
box = "L1000N" + res
run = "HYDRO_FIDUCIAL"
data_path = group_path + box + "/" + run
catalogue = "/SOAP"
#snapshot(5, data_path, catalogue, res)
#breakpoint() 

if res == '5040':
    merger_tree_path = "/cosma8/data/dp004/jch/FLAMINGO/MergerTrees/ScienceRuns/" + box + "/" + run + "/trees_f0.1_min10_max100/vr_trees.hdf5"
else:
    merger_tree_path = "/cosma8/data/dp004/jch/FLAMINGO/MergerTrees/ScienceRuns/" + box + "/" + run + "/trees_f0.1_min10_max100_vpeak/vr_trees.hdf5"

h5file = h5.File(merger_tree_path, 'r')

h5dset = h5file['/MergerTree/GalaxyID']
Galaxy_ID = h5dset[...] #ID of each halo                                                                                                                                                             

#h5dset = h5file['MergerTree/DescendantID']
#Desc_ID =  h5dset[...] #ID of last prog of each halo

h5dset = h5file['Subhalo/ID'] #Galaxy ID given in Velociraptor                                                                                                                                   
mrg_tree_VR = h5dset[...]

h5dset = h5file['Subhalo/SnapNum']
Snap_ID = h5dset[...]
h5file.close()
del h5file

def get_cat_data(z):
    snap = snapshot(z, data_path, catalogue, res)
    halo_props_name = "/halo_properties_" + str(snap).zfill(4) + ".hdf5"
    catalogue_path = data_path + catalogue + halo_props_name
    cc = cat.catalogue(catalogue_path, apert='50')
    
    M = cc.M200m
    z = cc.redshift
    del cc
    return M, z, (1/(1+z))

def calc_delta_a(z, rhoCrit,OmegaMatter):
    rho = 200*OmegaMatter*rhoCrit
    tdyn = unyt.unyt_quantity(1/np.sqrt(G * rho),'s')
    
    snap_ID = snapshot(z, data_path, catalogue, res)
    snapshot_name = "/flamingo_" + str(snap_ID).zfill(4)
    snap = snap_data(data_path + "/snapshots" + snapshot_name + snapshot_name + ".hdf5")
    print(snap.time - tdyn.to('Gyr'))
    print('tdyn:%s' % tdyn.to('Gyr'))
    #manually find the snapshot where the present time - tdyn is found
    #return the redshift for this
    return 
    
def IU_conv():
    halo_props_name = "/halo_properties_0018.hdf5"
    catalogue_path = data_path + catalogue + halo_props_name
    h5file = h5.File(catalogue_path, 'r')
    #load the highest redshift to reduce time
    groupName="/SWIFT/InternalCodeUnits"
    h5group=h5file[groupName]
    #convert to cgs
    attrName = "Unit mass in cgs (U_M)"
    IU_m = h5group.attrs[attrName]
    attrName =  "Unit time in cgs (U_t)"
    IU_s = h5group.attrs[attrName] 
    attrName = "Unit length in cgs (U_L)"
    IU_cm = h5group.attrs[attrName]
    
    return IU_m, IU_s, IU_cm

def save_files(z_pres):
    
    #z_pres = present redshift
    #zi = z at which we have to move back to to calculate Gamma
    my_dict = {'VR_ID':[], 'Gamma':[]}
    snap_z0 = snapshot(z_pres, data_path, catalogue, res)
    
    #find cosmology constants
    halo_props_name = "/halo_properties_" + str(snap_z0).zfill(4) + ".hdf5"
    catalogue_path = data_path + catalogue + halo_props_name
    h5file = h5.File(catalogue_path, 'r')
    groupName="/SWIFT/Cosmology/"
    h5group=h5file[groupName]
    attrName="Omega_lambda"
    OmegaLambda=h5group.attrs[attrName]
    attrName="H0 [internal units]"
    H0=h5group.attrs[attrName]
    #attrName="Omega_b"
    #OmegaBaryon=h5group.attrs[attrName]
    attrName="Omega_m"
    OmegaMatter=h5group.attrs[attrName]
    attrName="Critical density [internal units]"
    rhoCrit=h5group.attrs[attrName]
    h5file.close()
    del h5file
    #IU_m, IU_s, IU_cm = IU_conv()
    #calc_delta_a(z_pres, unyt.unyt_quantity(rhoCrit * IU_m * IU_cm**-3, 'g*cm**-3') , OmegaMatter)
    #breakpoint()
    zi = 0.5
    snap_zi = snapshot(zi, data_path, catalogue, res)

    #get indecs of clusters at z=0 and VR_IDs from merger tree
    snap_idx_z0 = np.where(Snap_ID == snap_z0)[0]
    mrg_tree_VR_z0 = mrg_tree_VR[snap_idx_z0]

    #VR ids for catalogue and merger tree must be identical
    #M500c index must be the same
    
    #Get mass data from SOAP cat.
    M_z0, z0,a0 = get_cat_data(z_pres)
    M_zi, zi, ai = get_cat_data(zi)
    
    if res == '5040' or res == '1800':
        mass_cut = np.where(M_z0 > 1e14)[0]
    else:
        mass_cut = np.where(M_z0 > 1e13)[0]

    #connect z=0 and z=0.5 by sorting in order of VR(z=0) list
    #use the fact that halos are stored in snapshot order to get VRs for z=0.5
    #both VR(z=0) and VR(z=0.5) lists should be a fixed number of indexes apart
    
    #recall VR_ID = index + 1
    idx_z0 = mrg_tree_VR_z0 - 1 #change to indexes to use in datasets to get mass     
    sort_idx = idx_z0.argsort()[mass_cut] #sort so only looking at clusters >1e13 at z=0 and their progenitors                                                                                     
    idx_z0 = idx_z0[sort_idx]    

    #get main prog at zi, e.g if we are looking at accretion rates for z=0
    #then zi will be z=0.5 and treelen will be 10
    #for zi=1.35, treelen will be 27
    
    snap_diff = 0.05
    tree_len = round((zi[0] - z0[0]) / snap_diff)
    if (res == '3600') & (zi > 1) & (z0 <= 0.95):
        #missing redhisft 1 in 3600 files
        tree_len = int(tree_len - 1)

    idx_zi = mrg_tree_VR[snap_idx_z0[sort_idx]+tree_len] - 1
    redshift_check = np.where(Snap_ID[snap_idx_z0[sort_idx]+tree_len] == snap_zi)[0] #make sure that there is a progenitor at z=0.5                                                          
    
    #this will create lists with 0s so we keep index consisten with other data
    M_sort_z0 = np.zeros(len(idx_z0))
    M_sort_zi = np.zeros(len(idx_zi))
    
    M_sort_z0[redshift_check] = M_z0[idx_z0[redshift_check]]
    M_sort_zi[redshift_check] = M_zi[idx_zi[redshift_check]]
    
    my_dict['VR_ID'].append(idx_z0 +1)
    gamma = (np.log10(M_sort_z0) - np.log10(M_sort_zi)) / (np.log10(a0) - np.log10(ai))
    my_dict['Gamma'].append(gamma)
    my_units = {'VR_ID':'','Gamma':''}

    for field in my_dict:
        my_dict[field] = unyt.unyt_array(my_dict[field],my_units[field])

    fake_ds = {'H0': H0, 'om_L':OmegaLambda, 'om_M':OmegaMatter}
    yt.save_as_dataset(fake_ds, 'saved_data/L1000N/1800/m9_acc_200m_z0_1e14_50kpc.h5', data=my_dict)

if __name__ == "__main__":
    save_files(0)
    breakpoint()
    
    #z_list = [0,0.5,1]    
    z_list = [0,0.5,0.95]
    #mass_range = [(14.1,14.45),(14.45,14.7),(14.7,15.15)]
    mass_range = [(13.5,13.9),(13.9,14.4),(14.4,14.8)]  
    fig, ax = plt.subplots(1,3, figsize=(14,6), sharey = True)
    #for res in [('3600','L1000N','m8'), ('1800','L1000N','m9'), ('5040','L2800N','m9')]:
    
    for j in range(len(mass_range)):
        for z in z_list:
            print('...loading data...')
            ds_acc = yt.load("saved_data/L1000N/3600/m8_acc_200m_z"+ str(z) +"_1e13_50kpc.h5")
            ds_mag = yt.load("saved_data/L1000N/3600/m8_mag_200m_z"+ str(z) +"_1e13_50kpc.h5")
            
            data_acc = ds_acc.data
            data_mag = ds_mag.data
    
            VR_acc = data_acc['VR_ID'][0]
            VR_mag = data_mag['host_id']
            m14_list = data_mag['M14']
            gamma = data_acc['Gamma'][0]
            mass =  data_mag['M200m']
            #breakpoint()
            acc = []
            m14 = []
            m200 = []
            pbar = yt.get_pbar('Comparing lists...', len(VR_mag))
            for i, vr in enumerate(VR_mag):
                pbar.update(i)
                idx = np.where(vr == VR_acc)[0]
                if len(idx) != 0:
                    if (np.log10(mass[i]) >= mass_range[j][0]) and (np.log10(mass[i]) <= mass_range[j][1]):
                        if np.isfinite(gamma[idx[0]]):
                            acc.append(gamma[idx[0]])
                            m14.append(m14_list[i])
                            m200.append(mass[i])
                        else: 
                            continue
                    else:
                        continue
                
            median,_,bins = data_bin(acc, m14, 15, stats=False)
            x = np.linspace(np.min(acc),np.max(acc),14)
            ax[j].plot(x,median, label=z)
            ax[j].set_xlim(-0.5,7.5)
            ax[j].set_title(mass_range[j])
            ax[j].set_xlabel('mag_gap')
            #tx = ax.twiny()
            #tx.xaxis.tick_top()
            #tx.xaxis.set_ticks(median_mass)
            #tx.set_xlim(-0.5,7.5)
            
    plt.legend()
    plt.ylabel('mag gap')
    plt.xlabel('gamma')
    plt.savefig('acc.png')
    breakpoint()


