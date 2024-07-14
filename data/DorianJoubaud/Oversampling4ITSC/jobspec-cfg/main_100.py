from iterativeOS import *
import random
import sys
import os
import ast
import tensorflow as tf 

# ======= Input settings =======
dataset = sys.argv[1]
clfs = sys.argv[2]
tech = sys.argv[3]
Ma = list()
mi = list()
pourcentage = int(sys.argv[6])
for x in sys.argv[4]:
    Ma.append(int(x))
    
for x in sys.argv[5]:
    mi.append(int(x))
trig = False
nb_iter = 5


# ======= Input data =======
x_train = np.array(pd.read_csv(f'data/{dataset}/{dataset}_TRAIN.tsv', delimiter='\t',header=None))
x_test = np.array(pd.read_csv(f'data/{dataset}/{dataset}_TEST.tsv', delimiter='\t',header=None))

# ======= Creating folders for results ======= 
out = f'results/{dataset}/{sys.argv[4]}_{sys.argv[5]}/{clfs}_{tech}'
if not os.path.exists(out):
    os.makedirs(out)

np.savetxt(out+'/'+'accI.txt', [0 for i in range(nb_iter)])
np.savetxt(out+'/'+'mccI.txt',[0 for i in range(nb_iter)])
np.savetxt(out+'/'+'f1I.txt',[0 for i in range(nb_iter)])
np.savetxt(out+'/'+'gI.txt',[0 for i in range(nb_iter)])

np.savetxt(out+'/'+'accB.txt', [0 for i in range(nb_iter)])
np.savetxt(out+'/'+'mccB.txt',[0 for i in range(nb_iter)])
np.savetxt(out+'/'+'f1B.txt',[0 for i in range(nb_iter)])
np.savetxt(out+'/'+'gB.txt',[0 for i in range(nb_iter)])



# ======= Preprocessing data =======
y_train = x_train[:,0]
x_train = x_train[:,1:]

y_test = x_test[:,0]
x_test = x_test[:,1:]

y_train = class_offset(y_train, dataset)
y_test = class_offset(y_test, dataset)

# If <11 data per class percentage = 8

    
nb_timesteps = int(x_train.shape[1] / 1)
input_shape = (nb_timesteps , 1)
    
x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))

_, dist = np.unique(y_train, return_counts = True)
if dist[0] <11:
    pourcentage = 8
bal_str = {i:dist[i] for i in range(len(dist))}

y_train = tf.keras.utils.to_categorical(y_train, num_classes=None, dtype="float32")
y_test = tf.keras.utils.to_categorical(y_test, num_classes=None, dtype="float32")

# ======= Create all distribution from balance to imbalance =======

nb = dist.sum() 

dist_id = list()
all_dist = np.array(getAlldistMulti(len(dist),np.max(dist),Ma, mi)).astype(int)
dist_id = list()   
    
for dist_i in all_dist:
        
    dist_id.append([dist_i, ID(dist_i, 'HE'),dist_i.sum()])
            
dist_id.sort(key=lambda x: x[1])


id_trunk = list()
dist_trunk = list()

for i in np.linspace(0,99,pourcentage):
    
    idx = int(i*(len(dist_id)/100))
    
    
    
    id_trunk.append(dist_id[idx][1])
    dist_trunk.append(dist_id[idx][0])

pd.DataFrame(list(zip(dist_trunk, id_trunk))).to_csv(out+'/'+'id.txt')


# ======= Classifier initialization =======
clf = Classif(clfs)

# ======= Warning if initial data is not balanced =======
if list(dist).count(dist[0]) == len(dist):
    print('==== Imbalanced initial dataset ====')
    

    

# ======= Let s go =======





for i in range(len(dist_trunk)):
    print(f'======= {i/len(dist_trunk)*100} % =======')
    # ======= Create the sampling strategy =======
    sp_str = {j:dist_trunk[i][j] for j in range(len(dist))}
    print(sp_str)
    # ======= Temporary list to store the results
    tmp_ai = list()
    tmp_mi = list()
    tmp_fi = list()
    tmp_gi = list()
    
    tmp_ab = list()
    tmp_mb = list()
    tmp_fb = list()
    tmp_gb = list()
    
    # ======= Loop for how many time do we make the classification for one case =======
    for ite in range(nb_iter):
        # ======= Get the undersampled imbalanced data =======
        x,y = RandomUnderSampler(sampling_strategy=sp_str).fit_resample(x_train[:,:,0],np.argmax(y_train, axis = 1))
        
        x = x.reshape((-1, input_shape[0], 1))
        y = tf.keras.utils.to_categorical(y, num_classes=None, dtype="float32")

        # ====print(x.shape)=== Classif on imbalanced data =======
        
        
        if not os.path.exists(out+'/I_history'+f'/I_{i}'):
            os.makedirs(out+'/I_history'+f'/I_{i}')
            
        clf = Classif(clfs)
        clf.fit(x,y,x_test, y_test, dataset, f'I_{i}',out = out+'I_history', iters=ite) 
        
        a, m, f, g = clf.getPerf(x_test, y_test, average = True)
        
        tmp_ai.append(a)
        tmp_mi.append(m)
        tmp_fi.append(f)
        tmp_gi.append(g)
        
        
        # ======= Balancing phase using Data augmentation techniques choosen =======
        sampler = Sampler(tech, bal_str)
        x_os, y_os = sampler.sampleData(x[:,:,0],np.argmax(y, axis=1))
        x_os = np.array(x_os).reshape((-1, input_shape[0], input_shape[1]))
        y_os = tf.keras.utils.to_categorical(y_os, num_classes=None, dtype="float32")
        
        # ======= Classif on balanced data with synthetic data =======
        if not os.path.exists(out+'/B_history'+f'/B_{i}'):
            os.makedirs(out+'/B_history'+f'/B_{i}')
        clf.fit(x_os, y_os, x_test, y_test, dataset, f'B_{i}', out=out+'B_history',iters = ite) 
        
        
        a, m, f, g = clf.getPerf(x_test, y_test, average = True)
        
        tmp_ab.append(a)
        tmp_mb.append(m)
        tmp_fb.append(f)
        tmp_gb.append(g)
        
        
    # ======= Saving the nb_iter time we made the exp =======
    # Each line of the output file will represent a classification made with different imbalance
    # Each column of the output file will represent the i th of the nb_iter classif we made for a given imbalance
    
    # ======= Imbalance data results =======
    a = list(np.loadtxt(out+'/'+'accI.txt'))
    m = list(np.loadtxt(out+'/'+'mccI.txt'))
    f = list(np.loadtxt(out+'/'+'f1I.txt'))
    g = list(np.loadtxt(out+'/'+'gI.txt'))
  
        
    a = np.vstack([a, tmp_ai])
    m = np.vstack([m, tmp_mi])
    f = np.vstack([f, tmp_fi])
    g = np.vstack([g, tmp_gi])
    
    
    
        
    np.savetxt(out+'/'+'accI.txt', a)
    np.savetxt(out+'/'+'mccI.txt' ,m)
    np.savetxt(out+'/'+'f1I.txt', f)
    np.savetxt(out+'/'+'gI.txt', g)
    
    # ======= Balanced data with synthetic results =======
        
    a = list(np.loadtxt(out+'/'+'accB.txt'))
    m = list(np.loadtxt(out+'/'+'mccB.txt'))
    f = list(np.loadtxt(out+'/'+'f1B.txt'))
    g = list(np.loadtxt(out+'/'+'gB.txt'))
        
    a = np.vstack([a, tmp_ab])
    m = np.vstack([m, tmp_mb])
    f = np.vstack([f, tmp_fb])
    g = np.vstack([g, tmp_gb])
        
    np.savetxt(out+'/'+'accB.txt', a)
    np.savetxt(out+'/'+'mccB.txt',m)
    np.savetxt(out+'/'+'f1B.txt',f)
    np.savetxt(out+'/'+'gB.txt',g)
        
        
    
    
# for i in range(step):
    
    
#     tmp_ai = list()
#     tmp_mi = list()
#     tmp_fi = list()
#     tmp_gi = list()
    
#     tmp_ab = list()
#     tmp_mb = list()
#     tmp_fb = list()
#     tmp_gb = list()
    
#     # We realise 10 time the classification 
#     print(f'============== {i/step *100} % ================')
#     for ite in range(nb_iter):
        
#         correct_dist = False
#         # Select % random data belonging to the other classes
#         while not correct_dist:
#             y_red = random.choices(y_reduction, k = int(len(y_reduction) - i/step*len(y_reduction)))
#             _, new_dist = np.unique(y_red, return_counts = True)
#             correct_dist = True
#             for yi in range(len(new_dist)):
#                 if new_dist[yi] > red_dist[yi]:
#                     correct_dist = False
        
        
        
        
#         print(new_dist)
        
#         if np.sum(new_dist) <= 2*len(new_dist):
#             print('No more :)')
#             trig = True
#             break
#         for idx in range(len(new_dist)):
            
#             if (new_dist[idx]< 3):
#                 new_dist[np.argmax(new_dist)] -= 3 - new_dist[idx]    
#                 new_dist[idx] = 3
                    
#         # Create dict of the new distribution
#         sp_str = {0 : list(y_train).count(class_to_keep)} 
        
#         bal_str = {idx : list(y_train).count(class_to_keep) for idx in range(len(new_dist)+1)}
        
#         for j in range(len(new_dist)):
#             sp_str[j+1] = new_dist[j]
        
#         print(sp_str)
#         x,y = RandomUnderSampler(sampling_strategy=sp_str).fit_resample(x_train,y_train) # x and y
        
#         clf.fit(x,y) # Imbalanced classif
        
#         a, m, f, g = clf.getPerf(x_test, y_test, average = True)
        
#         tmp_ai.append(a)
#         tmp_mi.append(m)
#         tmp_fi.append(f)
#         tmp_gi.append(g)
        
#         #Balancing phase
#         sampler = Sampler(tech, bal_str)
#         x_os, y_os = sampler.sampleData(x,y)
        
        
#         clf.fit(x_os, y_os) #Balanced classif
        
#         a, m, f, g = clf.getPerf(x_test, y_test, average = True)
        
#         tmp_ab.append(a)
#         tmp_mb.append(m)
#         tmp_fb.append(f)
#         tmp_gb.append(g)
        
#     if not trig:
#         a = list(np.loadtxt(out+'/'+'accI.txt'))
#         m = list(np.loadtxt(out+'/'+'mccI.txt'))
#         f = list(np.loadtxt(out+'/'+'f1I.txt'))
#         g = list(np.loadtxt(out+'/'+'gI.txt'))
        
    
#         a = np.vstack([a, tmp_ai])
#         m = np.vstack([m, tmp_mi])
#         f = np.vstack([f, tmp_fi])
#         g = np.vstack([g, tmp_gi])
        
    
        
        
        
    
#         np.savetxt(out+'/'+'accI.txt', a)
#         np.savetxt(out+'/'+'mccI.txt' ,m)
#         np.savetxt(out+'/'+'f1I.txt', f)
#         np.savetxt(out+'/'+'gI.txt', g)
    
        
#         a = list(np.loadtxt(out+'/'+'accB.txt'))
#         m = list(np.loadtxt(out+'/'+'mccB.txt'))
#         f = list(np.loadtxt(out+'/'+'f1B.txt'))
#         g = list(np.loadtxt(out+'/'+'gB.txt'))
        
#         a = np.vstack([a, tmp_ab])
#         m = np.vstack([m, tmp_mb])
#         f = np.vstack([f, tmp_fb])
#         g = np.vstack([g, tmp_gb])
        
    
#         np.savetxt(out+'/'+'accB.txt', a)
#         np.savetxt(out+'/'+'mccB.txt',m)
#         np.savetxt(out+'/'+'f1B.txt',f)
#         np.savetxt(out+'/'+'gB.txt',g)
    
        
        
                 
        
    
print('========= END =========')