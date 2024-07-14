import pdb
from math import log
import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans


step = 0.25;
n_dim = 2;
initial_partition = 1;
exponents = np.arange(1, 4, step);
partitions = 10**(exponents);
#partitions = [10, 100, 1000, 10000, 177.8279, 1778.2794, 17.7828, 3162.2777, 316.2278, 31.6228, 56.2341, 562.3413, 5623.4133]
partitions = [10, 17.7828, 31.6228, 56.2341, 100, 177.8279, 316.2278, 562.3413, 1000, 1778.2794, 3162.2777, 5623.4133]
entropy_rate = [0]*len(partitions);
coarse_graining_factor = 20 # For training kmeans
tau = 400
for idx in range(0,len(partitions)):
    print(str(partitions[idx]) + ' of ' + str(partitions[-1]),flush=True)
    n_partitions = int(np.round(partitions[idx]));
    ###########
    #n_partitions = 10
    ###########
    x = [];
    transition_matrix = np.zeros((n_partitions,n_partitions))
    
    for i in range(0,20):
        dat = sio.loadmat('coordinates_overdamped_0.8/X/' + str(10) + '/x_' + str(i+1) + '.mat')
        x.append(dat['X'])
    ############
    partitions[idx] = 10
    ############
    x = np.hstack(x)

    
    # Train kmeans on a subset of samples
    idx_array = np.arange(0,x.shape[1])
    idx_array = np.random.permutation(idx_array)
    idx_array = idx_array[np.arange(0,x.shape[1],coarse_graining_factor)]
    training_set = x[0][idx_array]
    training_set = training_set.reshape(len(training_set),1)
    kmeans_object = KMeans(n_clusters=n_partitions, init='k-means++', random_state=0).fit(training_set);
    print('Finished clustering')
    # Predict labels and centers of cluster
    c = kmeans_object.cluster_centers_
    #c_idx = kmeans_object.labels_
    c_idx = kmeans_object.predict(x.T)
    C = {}; C_idx = {}; C['center'] = c; C_idx['indices'] = c_idx;
    #sio.savemat('data/centers.mat', C)
    #sio.savemat('data/cluster_indices.mat',C_idx)
    coordinates = {}
    coordinates['X'] = x
    #sio.savemat('data/coordinates.mat',coordinates)
    for i in range(1,x.shape[1]-tau):
        #if ((i+1)%500000 == 0):
        #    continue
        transition_matrix[c_idx[i-1],c_idx[i+tau]] = transition_matrix[c_idx[i-1],c_idx[i+tau]] + 1;
    normalization_matrix = np.reshape(np.sum(transition_matrix,1),(n_partitions,1))
    transition_matrix = transition_matrix/np.tile(normalization_matrix,(1,n_partitions))
    print('Transition matrix evaluated')
    
    transition = {}
    #equilibrium_probability = {}
    transition['matrix']  = transition_matrix
    #equilibrium_probability['eq_prob'] = eq_probability
    w,v = np.linalg.eig(transition_matrix)
    eigen = {}
    eigen['eigenvalues'] = w
    sio.savemat('data/eigenvalues_'+str(idx)+'.mat',eigen)
    sio.savemat('data/transition_matrix'+str(idx)+'.mat',transition)
    #sio.savemat('data/eq_probability'+str(idx)+'.mat',equilibrium_probability)
entropy = {}
entropy['entropy_rate'] = entropy_rate
sio.savemat('entropy.mat',entropy)
