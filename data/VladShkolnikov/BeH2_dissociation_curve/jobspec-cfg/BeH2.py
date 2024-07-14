
import vqe_methods_add_by_one_Harper_truncation
import pickle
import sys
import numpy as np



min_dist = float(sys.argv[1])
max_dist = float(sys.argv[2])
num_step = int(sys.argv[3])
id_step = int(sys.argv[4])

dist_list = list(np.linspace(min_dist, max_dist, num_step))
dist_list = list(map(lambda x: round(x, 5), dist_list))


r=dist_list[id_step]

print(f'The list of distances to compute is: {dist_list}')
print(f'I am the distance: {r}')

PoolOfHope=['ZYXIZZZZZYYI', 'YXIIZZIIYYII', 'ZIXYZZZIYYII', 'XXIZZZYXIIII', 'XYZIZIYYZIII', 'IIYXYYZZZZII', 'ZZYXIZYYIIII', 'YZIXZZZIIYYI', 'IXXZIIIZZXYI', 'XIIIZZXYYIXY', 'XXXZYXXXYXYI', 'ZXIIIZZZZYII', 'XIZZIZXYZXII', 'YZXZZIZZYZYI', 'YZYXZIZIXZXY', 'ZZZIZIIZXXXY', 'IZZZYYYXYXXY']
Resultat=[]


geometry = "H 0 0 0; Be 0 0 {}; H 0 0 {}".format(r,2*r)
print(geometry)
vqe_methods_add_by_one_Harper_truncation.adapt_vqe(geometry,
	                  adapt_thresh    = 1e-7,                        #gradient threshold
                      adapt_maxiter   = 400,                       #maximum number of ops                   
                      Pool            = PoolOfHope,
                      Resultat        = Resultat,
                      bond_legth     = r
                      ) 
    
with open('Bond_length_dependence.BeH2_dissociation_curve_pickle_min_pool_{}'.format(r) , 'wb') as handle:
    pickle.dump(Resultat, handle, protocol=pickle.HIGHEST_PROTOCOL)                        





