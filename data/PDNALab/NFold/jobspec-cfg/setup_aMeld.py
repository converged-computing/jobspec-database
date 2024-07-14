#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from meld.remd import ladder, adaptor, leader
from meld import comm, vault
from meld import system
from meld import parse
import meld.system.montecarlo as mc
from meld.system.restraints import LinearRamp,ConstantRamp
from collections import namedtuple
import glob as glob


N_REPLICAS = 30    #number of replicas, which is equal to the number of GPUs should be used
N_STEPS = 20000    #simulation length of each replica, usually long enough for sampling with sufficient data.
BLOCK_SIZE = 100   #number of frames saved to Data/Blocks as each checkpoint 


def gen_state_templates(index, templates):                                                                                                                                                                              
    n_templates = len(templates)
    print((index,n_templates,index%n_templates))
    a = system.ProteinMoleculeFromPdbFile(templates[index%n_templates])
    #Note that it does not matter which forcefield we use here to build
    #as that information is not passed on, it is used for all the same as
    #in the setup part of the script
    b = system.SystemBuilder(forcefield="ff14sbside")
    c = b.build_system_from_molecules([a])
    pos = c._coordinates
    c._box_vectors=np.array([0.,0.,0.])
    vel = np.zeros_like(pos)
    alpha = index / (N_REPLICAS - 1.0)
    energy = 0 
    return system.SystemState(pos, vel, alpha, energy,c._box_vectors)
    

def get_distogram(filename, s, scaler,tight=False):
    dists = []
    rest_group = []
    lines = open(filename).read().splitlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            dists.append(s.restraints.create_restraint_group(rest_group, 1))
            rest_group = []
        else:
            cols = line.split()
            i = int(cols[0])
            name_i = cols[1]
            j = int(cols[2])
            name_j = cols[3]
            dist = float(cols[4]) / 10.
     
            r1 = dist -0.2
            if r1 < 0:
                r1 = 0.0
            if not tight:
                rest = s.restraints.create_restraint('distance', scaler,LinearRamp(0,100,0,1),
                                                      r1=r1, r2=dist, r3=dist+0.5, r4=dist+0.7, k=700,
                                                      atom_1_res_index=i, atom_2_res_index=j,                      
                                                      atom_1_name=name_i, atom_2_name=name_j)
            else:
                rest = s.restraints.create_restraint('distance', scaler,LinearRamp(0,100,0,1),
                                                      r1=r1, r2=dist, r3=dist+0.3, r4=dist+0.5, k=700,
                                                      atom_1_res_index=i, atom_2_res_index=j,           
                                                      atom_1_name=name_i, atom_2_name=name_j)
 
            rest_group.append(rest)
    return dists


def gen_state(s, index):
    pos = s._coordinates
    pos = pos - np.mean(pos, axis=0)
    s._box_vectors=np.array([0.,0.,0.])
    vel = np.zeros_like(pos)
    alpha = index / (N_REPLICAS - 1.0)
    energy = 0
    return system.SystemState(pos, vel, alpha, energy, s._box_vectors)


def setup_system():
    # load the sequence
    sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
    n_res = len(sequence.split())
    #print "!!!Starting from Templates!!!"
    templates = glob.glob('TEMPLATES/*.pdb')
    
    # build the system
    p = system.ProteinMoleculeFromPdbFile(templates[0])
    #p = system.ProteinMoleculeFromSequence(sequence)
    b = system.SystemBuilder(forcefield="ff14sbside")
    s = b.build_system_from_molecules([p])
    s.temperature_scaler = system.GeometricTemperatureScaler(0, 0.4, 400., 550.)
    n_res = s.residue_numbers[-1]
    #
    # Distance Restraints scaler
    #
    dist_scaler = s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)

    # Torsion Restraints. We trust 90% and 80% of tight and broad range
    #
    torsion_rests = []
    for line in open('phi.dat','r'):
        cols = line.split()
        res = int(cols[0])
        phi_avg = float(cols[1])
        phi_sd = 50
        phi_rest = s.restraints.create_restraint('torsion', dist_scaler,
                                                 phi=phi_avg, delta_phi=phi_sd, k=0.1,
                                                 atom_1_res_index=res-1, atom_1_name='C',
                                                 atom_2_res_index=res, atom_2_name='N',
                                                 atom_3_res_index=res, atom_3_name='CA',
                                                 atom_4_res_index=res, atom_4_name='C')
        torsion_rests.append(phi_rest)

    for line in open('psi.dat','r'):
        cols = line.split()
        res = int(cols[0])
        psi_avg = float(cols[1])
        psi_sd = 50
        psi_rest = s.restraints.create_restraint('torsion', dist_scaler,
                                                 phi=psi_avg, delta_phi=psi_sd, k=0.1,
                                                 atom_1_res_index=res, atom_1_name='N',
                                                 atom_2_res_index=res, atom_2_name='CA',
                                                 atom_3_res_index=res, atom_3_name='C',
                                                 atom_4_res_index=res+1, atom_4_name='N')
        torsion_rests.append(psi_rest)
    n_tors_keep = int(0.8 * len(torsion_rests))
    s.restraints.add_selectively_active_collection(torsion_rests, n_tors_keep)

    torsion_rests = []
    for line in open('tight_phi.dat','r'):
        cols = line.split()
        res = int(cols[0])
        phi_avg = float(cols[1])
        phi_sd = 30
        phi_rest = s.restraints.create_restraint('torsion', dist_scaler,
                                                 phi=phi_avg, delta_phi=phi_sd, k=0.1,
                                                 atom_1_res_index=res-1, atom_1_name='C',
                                                 atom_2_res_index=res, atom_2_name='N',
                                                 atom_3_res_index=res, atom_3_name='CA',
                                                 atom_4_res_index=res, atom_4_name='C')
        torsion_rests.append(phi_rest)
        
    for line in open('tight_psi.dat','r'):
        cols = line.split()
        res = int(cols[0])
        psi_avg = float(cols[1])
        psi_sd = 30
        psi_rest = s.restraints.create_restraint('torsion', dist_scaler,
                                                 phi=psi_avg, delta_phi=psi_sd, k=0.1,
                                                 atom_1_res_index=res, atom_1_name='N',
                                                 atom_2_res_index=res, atom_2_name='CA',
                                                 atom_3_res_index=res, atom_3_name='C',
                                                 atom_4_res_index=res+1, atom_4_name='N')
        torsion_rests.append(psi_rest)
    n_tors_keep = int(0.9 * len(torsion_rests))
    s.restraints.add_selectively_active_collection(torsion_rests, n_tors_keep)


    # Distance restraint. We trust 90% and 80% of tight and broad range 
    alphaFold = get_tight_distogram('tight_contacts.dat',s,scaler=dist_scaler,tight=True)
    s.restraints.add_selectively_active_collection(alphaFold, int(len(alphaFold)*0.90))   # trust 90% of the data
    alphaFold = get_distogram('contacts.dat',s,scaler=dist_scaler)
    s.restraints.add_selectively_active_collection(alphaFold, int(len(alphaFold)*0.80))   # trust 80% of the data

    
    # setup mc min at startup
    movers= []
    n_atoms = s.n_atoms
    for i in range(1, n_res +1):
        n=s.index_of_atom(i, 'N') -1
        ca = s.index_of_atom(i, 'CA') -1
        c= s.index_of_atom(i, 'C') -1
        mover= mc.DoubleTorsionMover(n, ca, list(range(ca,n_atoms)), 
                                     ca, c, list(range(c, n_atoms)))

        movers.append((mover,1))
    sched= mc.MonteCarloScheduler(movers, n_res *60)
 
    # create the options
    options = system.RunOptions()
    options.implicit_solvent_model = 'gbNeck2'
    options.use_big_timestep = False
    options.use_bigger_timestep = True
    options.cutoff = 1.8

    options.use_amap = False
    options.amap_alpha_bias = 1.0
    options.amap_beta_bias = 1.0
    options.timesteps = 11111
    options.minimize_steps = 20000
    options.min_mc = None
    options.run_mc = None

    # create a store
    store = vault.DataStore(s.n_atoms, N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE)
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)

    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=100)
    policy = adaptor.AdaptationPolicy(2.0, 50, 50)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy)

    remd_runner = leader.LeaderReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a)
    store.save_remd_runner(remd_runner)

    # create and store the communicator
    c = comm.MPICommunicator(s.n_atoms, N_REPLICAS)
    store.save_communicator(c)

    # create and save the initial states
    #states = [gen_state(s, i) for i in range(N_REPLICAS)]
    states = [gen_state_templates(i,templates) for i in range(N_REPLICAS)]
    store.save_states(states, 0)

    # save data_store
    store.save_data_store()

    return s.n_atoms


setup_system()

