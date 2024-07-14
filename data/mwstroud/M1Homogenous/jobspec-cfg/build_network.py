from bmtk.builder import NetworkBuilder
import numpy as np
from bmtk.builder.auxi.node_params import positions_cuboid, positions_list, xiter_random
from bmtk.utils.sim_setup import build_env_bionet
from bmtools.cli.plugins.util import util
import synapses
import math
import random
import os

from connectors import (one_to_one, one_to_one_offset, syn_dist_delay_feng_section, syn_uniform_delay_section,
                        syn_percent_o2a,recurrent_connector,one_to_one_thal,one_to_one_intbase,syn_percent, recurrent_connector_o2a,perc_conn,sphere_dropoff,section_id_placement,cs2cp_gaussian_cylinder)

np.random.seed(123412)

network_dir = 'network'
t_sim = 11500.0
dt = 0.05
scale = 1

x_start, x_end = 0,1000
y_start, y_end = 0,1000
z_start, z_end = 0,1000

# When enabled, a shell of virtual cells will be created around the core network.
edge_effects = False

# Number of cells in each population. Following 80/20 E/I equal on CP-CS and 60% FSI to 40% LTS for Interneurons
# Densities by cell proportion unless otherwise specified: CP: 20%  CS: 20% CTH: 20% CC: 20% FSI: 12% LTS: 8%
num_cells = 1000 * scale
num_CP = int(num_cells * 0.4)    # Corticopontine
num_CS = int(num_cells * 0.4)   # Corticostriatal
#num_CTH = int(num_cells * 0.2)  # Corticothalamic
#num_CC = int(num_cells * 0.2)   # Corticocortical
num_FSI = int(num_cells * 0.12)   # Fast Spiking Interneuron
num_LTS = int(num_cells * 0.08)    # Low Threshold Spiker

###################################################################################
####################### Cell Proportions and Positions ############################

# arrays for cell location csv
cell_name = []
cell_x = []
cell_y = []
cell_z = []
# amount of cells per layer

numCP_in5A = int(round(num_CP* 0.05))
numCP_in5B = int(round(num_CP* 0.95))  #CP cells are basically only in layer 5B and nowhere else. 


numCS_in5A = int(round(num_CS*0.95))    #CS cells span top of 5B to middle of 2/3 
numCS_in5B = int(round(num_CS*0.05))


numFSI_in5A = int(round(num_FSI*0.5))  # Even distribution of FSI cells between Layers 5A and 5B
numFSI_in5B = int(round(num_FSI*0.5))


numLTS_in5A = int(round(num_LTS*0.5))  # Even distribution of LTS cells between Layers 5A and 5B
numLTS_in5B = int(round(num_LTS*0.5))


min_conn_dist = 0.0 
max_conn_dist = 300.0 #300.0 #9999.9# Distance constraint for all cells

# total 400x400x1820 (ignoring layer 1)
# Order from top to bottom is 2/3, 4, 5A, 5B, 6
# Layer 2/3 (420 um thick) 23.1%
num_cells_5A=numCP_in5A+numCS_in5A+numFSI_in5A+numLTS_in5A
num_cells_5B=numCP_in5B+numCS_in5B+numFSI_in5B+numLTS_in5B

# Layer 5A (250 um thick) 13.7% (z is 250 to 499)
pos_list_5A = np.random.rand(num_cells_5A,3)
pos_list_5A[:,0]= pos_list_5A[:,0]*(x_end - x_start) + x_start
pos_list_5A[:,1]= pos_list_5A[:,1]*(y_end - y_start) + y_start
pos_list_5A[:,2] = pos_list_5A[:,2]*(1000 - 500) + 500

# Layer 5B (250 um thick) 13.7%  (z is 0 to 249)
pos_list_5B = np.random.rand(num_cells_5B,3)
pos_list_5B[:,0]= pos_list_5B[:,0]*(x_end - x_start) + x_start
pos_list_5B[:,1]= pos_list_5B[:,1]*(y_end - y_start) + y_start
pos_list_5B[:,2] = pos_list_5B[:,2]*(500-0) + 0




def build_networks(network_definitions: list) -> dict:
    # network_definitions should be a list of dictionaries eg:[{}]
    # Keys should include an arbitrary 'network_name', a positions_list (if any),
    # And 'cells'. 'cells' should contain a list of dictionaries, and the dictionary 
    # should corrospond with any valid input for BMTK's NetworkBuilder.add_nodes method 
    # A dictionary of NetworkBuilder BMTK objects will be returned, reference by individual network_name
    for net_def in network_definitions:
        network_name = net_def['network_name']
        if not networks.get(network_name):
            networks[network_name] = NetworkBuilder(network_name)  # This is changed
        pos_list = net_def.get('positions_list',None)

        # Add cells to the network
        for cell in net_def['cells']:
            num_cells = cell['N']
            extra_kwargs = {}
            if pos_list is not None:
                inds = np.random.choice(np.arange(0,np.size(pos_list,0)),num_cells,replace=False)
                pos = pos_list[inds,:]
                # Get rid of coordinates already used
                pos_list = np.delete(pos_list,inds,0)
                extra_kwargs['positions'] = positions_list(positions=pos)
            networks[network_name].add_nodes(**cell,**extra_kwargs)
    
    return networks

def build_edges(networks,edge_definitions,edge_params,edge_add_properties,syn=None):
    # Builds the edges for each network given a set of 'edge_definitions'
    # edge_definitions examples shown later in the code
    for edge in edge_definitions:
        network_name = edge['network']
        edge_src_trg = edge['edge']
        edge_params_val  = edge_params[edge['param']]
        dynamics_file = edge_params_val['dynamics_params']
        model_template = syn[dynamics_file]['level_of_detail']
        
        model_template_kwarg = {'model_template':model_template}

        net = networks[network_name]

        conn = net.add_edges(**edge_src_trg,**edge_params_val,**model_template_kwarg)
        
        if edge.get('add_properties'):
            edge_add_properties_val = edge_add_properties[edge['add_properties']]
            conn.add_properties(**edge_add_properties_val)

def save_networks(networks,network_dir):
    # Remove the existing network_dir directory
    for f in os.listdir(network_dir):
        os.remove(os.path.join(network_dir, f))

    # Run through each network and save their nodes/edges
    for i, (network_name, network) in enumerate(networks.items()):
        print('Building ' + network_name)
        network.build()
        network.save_nodes(output_dir=network_dir)
        network.save_edges(output_dir=network_dir)

networks = {} #Place to store NetworkBuilder objects referenced by name
network_definitions = [
    {  # Start Layer 5A
        'network_name':'cortex',
        'positions_list':pos_list_5A,
        'cells':[
            {   # CP
                'N':numCP_in5A,
                'pop_name':'CP',
                'rotation_angle_zaxis':xiter_random(N=numCP_in5A, min_x=0.33*np.pi, max_x=0.5*np.pi),  # want cells to point upward within 60degree cone
                'rotation_angle_yaxis':xiter_random(N=numCP_in5A, min_x=0.0, max_x=2*np.pi),
                'model_type':'biophysical',
                'model_template':'hoc:CP_Cell',
                'morphology': 'blank.swc'
            },
            {   # CS
                'N':numCS_in5A,
                'pop_name':'CS',
                'rotation_angle_zaxis':xiter_random(N=numCS_in5A, min_x=0.33*np.pi, max_x=0.5*np.pi),  # want cells to point upward within 60degree cone
                'rotation_angle_yaxis':xiter_random(N=numCS_in5A, min_x=0.0, max_x=2*np.pi),
                'model_type':'biophysical',
                'model_template':'hoc:CS_Cell',
                'morphology': 'blank.swc'
            },
            {   # FSI
                'N':numFSI_in5A,
                'pop_name':'FSI',
                'rotation_angle_zaxis':xiter_random(N=numFSI_in5A, min_x=0.0, max_x=2*np.pi),  # want cells to be randomly oriented
                'rotation_angle_yaxis':xiter_random(N=numFSI_in5A, min_x=0.0, max_x=2*np.pi),
                'model_type':'biophysical',
                'model_template':'hoc:FSI_Cell',
                'morphology': 'blank.swc'
            },
            {   # LTS
                'N':numLTS_in5A,
                'pop_name':'LTS',
                'rotation_angle_zaxis':xiter_random(N=numLTS_in5A, min_x=0.0, max_x=2*np.pi),  # want cells to be randomly oriented
                'rotation_angle_yaxis':xiter_random(N=numLTS_in5A, min_x=0.0, max_x=2*np.pi),
                'model_type':'biophysical',
                'model_template':'hoc:LTS_Cell',
                'morphology': 'blank.swc'
            }
        ]
    }, # End Layer 5A
    {  # Start Layer 5B
        'network_name':'cortex',
        'positions_list':pos_list_5B,
        'cells':[
            {   # CP
                'N':numCP_in5B,
                'pop_name':'CP',
                'rotation_angle_zaxis':xiter_random(N=numCP_in5B, min_x=0.33*np.pi, max_x=0.5*np.pi),  # want cells to point upward within 60degree cone
                'rotation_angle_yaxis':xiter_random(N=numCP_in5B, min_x=0.0, max_x=2*np.pi),
                'model_type':'biophysical',
                'model_template':'hoc:CP_Cell',
                'morphology': 'blank.swc'
            },
            {   # CS
                'N':numCS_in5B,
                'pop_name':'CS',
                'rotation_angle_zaxis':xiter_random(N=numCS_in5B, min_x=0.33*np.pi, max_x=0.5*np.pi),  # want cells to point upward within 60degree cone
                'rotation_angle_yaxis':xiter_random(N=numCS_in5B, min_x=0.0, max_x=2*np.pi),
                'model_type':'biophysical',
                'model_template':'hoc:CS_Cell',
                'morphology': 'blank.swc'
            },
            {   # FSI
                'N':numFSI_in5B,
                'pop_name':'FSI',
                'rotation_angle_zaxis':xiter_random(N=numFSI_in5B, min_x=0.0, max_x=2*np.pi),  # want cells to be randomly oriented
                'rotation_angle_yaxis':xiter_random(N=numFSI_in5B, min_x=0.0, max_x=2*np.pi),
                'model_type':'biophysical',
                'model_template':'hoc:FSI_Cell',
                'morphology': 'blank.swc'
            },
            {   # LTS
                'N':numLTS_in5B,
                'pop_name':'LTS',
                'rotation_angle_zaxis':xiter_random(N=numLTS_in5B, min_x=0.0, max_x=2*np.pi),  # want cells to be randomly oriented
                'rotation_angle_yaxis':xiter_random(N=numLTS_in5B, min_x=0.0, max_x=2*np.pi),
                'model_type':'biophysical',
                'model_template':'hoc:LTS_Cell',
                'morphology': 'blank.swc'
            }
        ]
    }, # End Layer 5B
    {  # Extrinsic Thalamic Inputs
        'network_name':'thalamus',
        'positions_list':None,
        'cells':[
            {   # Virtual Cells
                'N':num_CP+num_CS,
                'pop_name':'thal',
                'potential':'exc',
                'model_type':'virtual'
            }
        ]
    },
    {  # Extrinsic Intbase Inputs
        'network_name':'Intbase',
        'positions_list':None,
        'cells':[
            {   # Virtual Cells
                'N':num_FSI+num_LTS,
                'pop_name':'Int',
                'potential':'exc',
                'model_type':'virtual'
            }
        ]
    }
]

##########################################################################
############################  EDGE EFFECTS  ##############################

if edge_effects: # When enabled, a shell of virtual cells will be created around the core network.
    
    # compute the core volume
    core_x,core_y,core_z = (x_end-x_start),(y_end-y_start),(z_end-z_start)
    core_volume =  core_x * core_y * core_z

    # compute the outer shell volume. The absolute max_conn_dist will extend each dimension of the core by 2*max_conn_dist
    shell_x_start,shell_y_start,shell_z_start = x_start - max_conn_dist, x_start - max_conn_dist, z_start - max_conn_dist
    shell_x_end,shell_y_end,shell_z_end = x_end + max_conn_dist, y_end + max_conn_dist, z_end + max_conn_dist
    shell_x,shell_y,shell_z = (shell_x_end-shell_x_start),(shell_y_end-shell_y_start),(shell_z_end-shell_z_start)
    shell_volume = shell_x * shell_y * shell_z

    # Determine the size difference between core and shell
    shell_multiplier = (shell_volume/core_volume) 

    # Increase the number of original cells based on the shell_multiplier
    #Layer 5A
    virt_numCP_in5A = int(numCP_in5A * shell_multiplier)
    virt_numCS_in5A = int(numCS_in5A * shell_multiplier)
    virt_numFSI_in5A = int(numFSI_in5A * shell_multiplier)
    virt_numLTS_in5A  = int(numLTS_in5A * shell_multiplier)

    virt_num_cells_5A = virt_numCP_in5A +virt_numCS_in5A + virt_numFSI_in5A + virt_numLTS_in5A
    #Layer 5B
    virt_numCP_in5B = int(numCP_in5B * shell_multiplier)
    virt_numCS_in5B = int(numCS_in5B * shell_multiplier)
    virt_numFSI_in5B = int(numFSI_in5B * shell_multiplier)
    virt_numLTS_in5B  = int(numLTS_in5B * shell_multiplier)

    virt_num_cells_5B = virt_numCP_in5B +virt_numCS_in5B + virt_numFSI_in5B + virt_numLTS_in5B


    virt_num_cells =  virt_numCP_in5A + virt_numCS_in5A  + virt_numFSI_in5A + virt_numLTS_in5A + virt_numCP_in5B + virt_numCS_in5B  + virt_numFSI_in5B + virt_numLTS_in5B 
    
    # Create a positions list for each cell in the shell, this includes positions in the core
    virt_pos_list = np.random.rand(virt_num_cells,3)
    virt_pos_list[:,0] = virt_pos_list[:,0]*(shell_x_end - shell_x_start) + shell_x_start
    virt_pos_list[:,1] = virt_pos_list[:,1]*(shell_y_end - shell_y_start) + shell_y_start
    virt_pos_list[:,2] = virt_pos_list[:,2]*(shell_z_end - shell_z_start) + shell_z_start
    

    # Layer 5A (250 um thick) 13.7% (z is 250 to 499)
    virt_pos_list_5A = np.random.rand(virt_num_cells_5A,3)
    virt_pos_list_5A[:,0]= virt_pos_list_5A[:,0]*(shell_x_end - shell_x_start) + shell_x_start
    virt_pos_list_5A[:,1]= virt_pos_list_5A[:,1]*(shell_y_end - shell_y_start) + shell_y_start
    virt_pos_list_5A[:,2] = virt_pos_list_5A[:,2]*(800) + 500

    # Layer 5B (250 um thick) 13.7%  (z is 0 to 249)
    virt_pos_list_5B = np.random.rand(virt_num_cells_5B,3)
    virt_pos_list_5B[:,0]= virt_pos_list_5B[:,0]*(shell_x_end - shell_x_start) + shell_x_start
    virt_pos_list_5B[:,1]= virt_pos_list_5B[:,1]*(shell_y_end - shell_y_start) + shell_y_start
    virt_pos_list_5B[:,2] = virt_pos_list_5B[:,2]*(800) - 300


    # EXCLUDE POSITIONS IN THE CORE - We remove all virtual cells located in the core (accounting for no -1 on shell_multiplier)
    in_core = np.where(((virt_pos_list[:,0] < x_start) | (virt_pos_list[:,0] > x_end)) & 
                       ((virt_pos_list[:,1] < y_start) | (virt_pos_list[:,1] > y_end)) & 
                       ((virt_pos_list[:,2] < z_start) | (virt_pos_list[:,2] > z_end)))
    virt_pos_list = np.delete(virt_pos_list,in_core,0)

    # Bring down the number of shell cells to create by scaling
    # This ensures we have enough positions in virt_pos_list for all of our cells
    # Old density multiplied by new number of cells
    new_virt_num_cells = len(virt_pos_list)
    #Layer 5A
    virt_numCP_in5A = int(virt_numCP_in5A/virt_num_cells*new_virt_num_cells)
    virt_numCS_in5A = int(virt_numCS_in5A/virt_num_cells*new_virt_num_cells)
    virt_numFSI_in5A = int(virt_numFSI_in5A/virt_num_cells*new_virt_num_cells)
    virt_numLTS_in5A = int(virt_numLTS_in5A/virt_num_cells*new_virt_num_cells)
    #Layer 5B
    virt_numCP_in5B = int(virt_numCP_in5B/virt_num_cells*new_virt_num_cells)
    virt_numCS_in5B = int(virt_numCS_in5B/virt_num_cells*new_virt_num_cells)
    virt_numFSI_in5B = int(virt_numFSI_in5B/virt_num_cells*new_virt_num_cells)
    virt_numLTS_in5B = int(virt_numLTS_in5B/virt_num_cells*new_virt_num_cells)

    
    virt_num_cells = virt_numCP_in5A + virt_numCS_in5A  + virt_numFSI_in5A + virt_numLTS_in5A + virt_numCP_in5B + virt_numCS_in5B  + virt_numFSI_in5B + virt_numLTS_in5B 
    #print("The number of Virtual Cells in the M1 Cortex:")
    #print(virt_num_cells)
    # This should always be true, virt_num_cells is now equal to a scaled down number
    # While new_virt_num_cells is the length of the available cells
    assert(virt_num_cells <= new_virt_num_cells)    

    # This network should contain all the same properties as the original network, except
    # the cell should be virtual. For connectivity, you should name the cells the same as
    # the original network because connection rules defined later will require it
    shell_network = [
    {  # Start Layer 5A
        'network_name':'shell',
        'positions_list':virt_pos_list_5A,
        'cells':[
            {   # CP
                'N':virt_numCP_in5A,
                'pop_name':'CP',
                'rotation_angle_zaxis':xiter_random(N=virt_numCP_in5A, min_x=0.33*np.pi, max_x=0.5*np.pi),  # want cells to point upward within 60degree cone
                'rotation_angle_yaxis':xiter_random(N=virt_numCP_in5A, min_x=0.0, max_x=2*np.pi),
                'model_type':'virtual'
            },
            {   # CS
                'N':virt_numCS_in5A,
                'pop_name':'CS',
                'rotation_angle_zaxis':xiter_random(N=virt_numCS_in5A, min_x=0.33*np.pi, max_x=0.5*np.pi),  # want cells to point upward within 60degree cone
                'rotation_angle_yaxis':xiter_random(N=virt_numCS_in5A, min_x=0.0, max_x=2*np.pi),
                'model_type':'virtual'
            },
            {   # FSI
                'N':virt_numFSI_in5A,
                'pop_name':'FSI',
                'rotation_angle_zaxis':xiter_random(N=virt_numFSI_in5A, min_x=0.0, max_x=2*np.pi),  # want cells to be randomly oriented
                'rotation_angle_yaxis':xiter_random(N=virt_numFSI_in5A, min_x=0.0, max_x=2*np.pi),
                'model_type':'virtual'
            },
            {   # LTS
                'N':virt_numLTS_in5A,
                'pop_name':'LTS',
                'rotation_angle_zaxis':xiter_random(N=virt_numLTS_in5A, min_x=0.0, max_x=2*np.pi),  # want cells to be randomly oriented
                'rotation_angle_yaxis':xiter_random(N=virt_numLTS_in5A, min_x=0.0, max_x=2*np.pi),
                'model_type':'virtual'
            }
        ]
    }, # End Layer 5A
    {  # Start Layer 5B
        'network_name':'shell',
        'positions_list':virt_pos_list_5B,
        'cells':[
            {   # CP
                'N':virt_numCP_in5B,
                'pop_name':'CP',
                'rotation_angle_zaxis':xiter_random(N=virt_numCP_in5B, min_x=0.33*np.pi, max_x=0.5*np.pi),  # want cells to point upward within 60degree cone
                'rotation_angle_yaxis':xiter_random(N=virt_numCP_in5B, min_x=0.0, max_x=2*np.pi),
                'model_type':'virtual'
            },
            {   # CS
                'N':virt_numCS_in5B,
                'pop_name':'CS',
                'rotation_angle_zaxis':xiter_random(N=virt_numCS_in5B, min_x=0.33*np.pi, max_x=0.5*np.pi),  # want cells to point upward within 60degree cone
                'rotation_angle_yaxis':xiter_random(N=virt_numCS_in5B, min_x=0.0, max_x=2*np.pi),
                'model_type':'virtual'
            },
            {   # FSI
                'N':virt_numFSI_in5B,
                'pop_name':'FSI',
                'rotation_angle_zaxis':xiter_random(N=virt_numFSI_in5B, min_x=0.0, max_x=2*np.pi),  # want cells to be randomly oriented
                'rotation_angle_yaxis':xiter_random(N=virt_numFSI_in5B, min_x=0.0, max_x=2*np.pi),
                'model_type':'virtual'
            },
            {   # LTS
                'N':virt_numLTS_in5B,
                'pop_name':'LTS',
                'rotation_angle_zaxis':xiter_random(N=virt_numLTS_in5B, min_x=0.0, max_x=2*np.pi),  # want cells to be randomly oriented
                'rotation_angle_yaxis':xiter_random(N=virt_numLTS_in5B, min_x=0.0, max_x=2*np.pi),
                'model_type':'virtual'
            }
        ]
    } # End Layer 5B
    ]
    # Add the shell to our network definitions
    network_definitions.extend(shell_network)

##########################################################################
##########################################################################

# Build and save our NetworkBuilder dictionary
networks = build_networks(network_definitions)




# Whole reason for restructuring network building lies here, by separating out the
# source and target params from the remaining parameters in NetworkBuilder's 
# add_edges function we can reuse connectivity rules for the virtual shell
# or elsewhere
# [
#    {
#       'network':'network_name', # => The name of the network that these edges should be added to (networks['network_name'])
#       'edge': {
#                    'source': {},
#                    'target': {}
#               }, # should contain source and target only, any valid add_edges param works
#       'param': 'name_of_edge_parameter' # to be coupled with when add_edges is called
#       'add_properties': 'prop_name' # name of edge_add_properties for adding additional connection props, like delay
#    }
# ]





# Reciprocal connections are defined weird in literature: 
# I'm defining group a and b for FSI cells just to clarify the explanation here:
# FSIa  -> FSIb 
# 1)   then 34% unidirectional FSIa -> FSIb
# 2)   and 43% reciprocal FSIa -> FSIb        (this means total 34+43 = 77% FSIa -> FSIb)
# 3)   the reciprocal implies FSIb -> FSIa (backwards) so that ALSO means 43% FSIb -> FSIa (OR 100% of the connections made in 2)
# I define 3 edge definitions to accomplish this: 34% unidirectional only (suffix "_uni"), 43% unidirectional with a track list (suffix "_uniback") ,
# and 100% backwards applied to the 43% track list I made (suffix "_rec" for reciprocal).

# Edges that have this logic ^^^: FSI-FSI, FSI-CP (implied CP-FSI), FSI-CS (implied CS-FSI), LTS-CP (implied CP-LTS), LTS-CS (implied CS-LTS)
# LTS-FSI, CS-LTS,

# A track list just puts all of the edges that were actually created into a python list and is used as reference for other edge creations.

# If there is no mention of uniback: This means that the literature mentions reciprocal connections in a different logic:
# I'm defining group a and b for CP cells just to clarify the explanation here:
# CPa  -> CPb
# 1) then 10% unidirectional CPa -> CPb with a track list
# 2) and 30% backwards means 30% of the connections already made in the 10% unidirectional edge creation (which really means 10% * 30%)
# I define 2 edge definitions to accomplish this: 10% unidirectional only (suffix "_uni") ,
# and 30% backwards applied to the 10% track list I made (suffix "_rec" for reciprocal).

# Edges that have this logic ^^^: CP-CP, CS-CS, CS-CP, CP-CS

edge_definitions = [
    {   # FSI -> FSI Unidirectional
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> FSI Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> FSI Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CP Unidirectional
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'FSI2CP_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CP Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'FSI2CP_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CP Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CP']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'FSI2CP_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CS Unidirectional 
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'FSI2CS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CS Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'FSI2CS_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CS Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CS']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'FSI2CS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> LTS Unidirectional 
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['LTS']}
        },
        'param': 'FSI2LTS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> LTS Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['LTS']}
        },
        'param': 'FSI2LTS_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> LTS Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'FSI2LTS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CP Unidirectional
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'LTS2CP_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CP Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'LTS2CP_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CP Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CP']}, 
            'target':{'pop_name': ['LTS']}
        },
        'param': 'LTS2CP_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CS Unidirectional 
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'LTS2CS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CS Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'LTS2CS_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CS Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CS']}, 
            'target':{'pop_name': ['LTS']}
        },
        'param': 'LTS2CS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> FSI Unidirectional
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'LTS2FSI_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> FSI Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'LTS2FSI_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> FSI Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['LTS']}
        },
        'param': 'LTS2FSI_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> CS
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CP']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'CP2CS',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> CP
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CS']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'CS2CP',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> CP Unidirectional
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CP']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'CP2CP_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> CP Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CP']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'CP2CP_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> CS Unidirectional
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CS']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'CS2CS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> CS Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CS']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'CS2CS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    }, 

        ################### THALAMIC INPUT ################

    {   # Thalamus Excitation to CP
        'network':'cortex',
        'edge': {
            'source':networks['thalamus'].nodes(),
            'target':networks['cortex'].nodes(pop_name=['CP'])
        },
        'param': 'Thal2CP'
    },
    {   # Thalamus Excitation to CS
        'network':'cortex',
        'edge': {
            'source':networks['thalamus'].nodes(),
            'target':networks['cortex'].nodes(pop_name=['CS'])
        },
        'param': 'Thal2CS'
    },

        ##################### Interneuron baseline INPUT #####################

    {    # To all FSI
        'network':'cortex',
        'edge': {
            'source':networks['Intbase'].nodes(),
            'target':networks['cortex'].nodes(pop_name=['FSI'])
        },
        'param': 'Intbase2FSI'        
    },
    {    # To all LTS
        'network':'cortex',
        'edge': {
            'source':networks['Intbase'].nodes(),
            'target':networks['cortex'].nodes(pop_name=['LTS'])
        },
        'param': 'Intbase2LTS'        
    },
]
# A few connectors require a list for tracking synapses that are recurrent, declare them here 
FSI_FSI_list = []
FSI_CP_list = []
FSI_CS_list = []
CP_CP_list = []
CS_CS_list = []
LTS_LTS_list = []
LTS_CP_list = []
CP_LTS_list = []
CS_LTS_list = []
FSI_LTS_list = []
LTS_FSI_list = []
#CS_CP_list = []
#CP_CS_list = []
# edge_params should contain additional parameters to be added to add_edges calls

# I copied this down to help code the parameters too.
# Reciprocal connections are defined weird in literature: 
# I'm defining group a and b for FSI cells just to clarify the explanation here:
# FSIa  -> FSIb 
# 1)   then 34% unidirectional FSIa -> FSIb
# 2)   and 43% reciprocal FSIa -> FSIb        (this means total 34+43 = 77% FSIa -> FSIb)
# 3)   the reciprocal implies FSIb -> FSIa (backwards) so that ALSO means 43% FSIb -> FSIa (OR 100% of the connections made in 2)
# I define 3 edge definitions to accomplish this: 34% unidirectional only (suffix "_uni"), 43% unidirectional with a track list (suffix "_uniback") ,
# and 100% backwards applied to the 43% track list I made (suffix "_rec" for reciprocal).

# Edges that have this logic ^^^: FSI-FSI, FSI-CP (implied CP-FSI), FSI-CS (implied CS-FSI), LTS-CP (implied CP-LTS), LTS-CS (implied CS-LTS)
# LTS-FSI, CS-LTS,

# A track list just puts all of the edges that were actually created into a python list and is used as reference for other edge creations.

# If there is no mention of uniback: This means that the literature mentions reciprocal connections in a different logic:
# I'm defining group a and b for CP cells just to clarify the explanation here:
# CPa  -> CPb
# 1) then 10% unidirectional CPa -> CPb with a track list
# 2) and 30% backwards means 30% of the connections already made in the 10% unidirectional edge creation (which really means 10% * 30%)
# I define 2 edge definitions to accomplish this: 10% unidirectional only (suffix "_uni") ,
# and 30% backwards applied to the 10% track list I made (suffix "_rec" for reciprocal).

# Edges that have this logic ^^^: CP-CP, CS-CS, CS-CP, CP-CS

edge_params = {
    'FSI2FSI_uni': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.34},
        'syn_weight':1,
        'dynamics_params':'FSI2FSI.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2FSI_uniback': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.43, 'track_list':FSI_FSI_list},
        'syn_weight':1,
        'dynamics_params':'FSI2FSI.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2FSI_rec': {
        'iterator':'one_to_all',
        'connection_rule':recurrent_connector_o2a,
        'connection_params':{'p':1, 'all_edges':FSI_FSI_list},
        'syn_weight':1,
        'dynamics_params':'FSI2FSI.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2CP_uni': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.20},
        'syn_weight':1,
        'dynamics_params':'FSI2CP.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2CP_uniback': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.28,'track_list':FSI_CP_list},
        'syn_weight':1,
        'dynamics_params':'FSI2CP.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2CP_rec': {
        'iterator':'one_to_all',
        'connection_rule':recurrent_connector_o2a,
        'connection_params':{'p':1, 'all_edges':FSI_CP_list},
        'syn_weight':1,
        'dynamics_params':'CP2FSI.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2CS_uni': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.17},
        'syn_weight':1,
        'dynamics_params':'FSI2CS.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2CS_uniback': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.20,'track_list':FSI_CP_list},
        'syn_weight':1,
        'dynamics_params':'FSI2CS.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2CS_rec': {
        'iterator':'one_to_all',
        'connection_rule':recurrent_connector_o2a,
        'connection_params':{'p':1, 'all_edges':FSI_CP_list},
        'syn_weight':1,
        'dynamics_params':'CS2FSI.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2LTS_uni': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.34},
        'syn_weight':1,
        'dynamics_params':'FSI2LTS.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2LTS_uniback': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.21,'track_list':FSI_LTS_list},
        'syn_weight':1,
        'dynamics_params':'FSI2LTS.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'FSI2LTS_rec': {
        'iterator':'one_to_all',
        'connection_rule':recurrent_connector_o2a,
        'connection_params':{'p':1, 'all_edges':FSI_LTS_list},
        'syn_weight':1,
        'dynamics_params':'LTS2FSI.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2CP_uni': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.35},
        'syn_weight':1,
        'dynamics_params':'LTS2PN.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2CP_uniback': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.175,'track_list':LTS_CP_list},
        'syn_weight':1,
        'dynamics_params':'LTS2PN.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2CP_rec': {
        'iterator':'one_to_all',
        'connection_rule':recurrent_connector_o2a,
        'connection_params':{'p':1, 'all_edges':LTS_CP_list},
        'syn_weight':1,
        'dynamics_params':'PN2LTS.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2CS_uni': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.35},
        'syn_weight':1,
        'dynamics_params':'LTS2PN.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2CS_uniback': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.175,'track_list':LTS_CP_list},
        'syn_weight':1,
        'dynamics_params':'LTS2PN.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2CS_rec': {
        'iterator':'one_to_all',
        'connection_rule':recurrent_connector_o2a,
        'connection_params':{'p':1, 'all_edges':LTS_CP_list},
        'syn_weight':1,
        'dynamics_params':'PN2LTS.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2FSI_uni': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.53},
        'syn_weight':1,
        'dynamics_params':'LTS2FSI.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2FSI_uniback': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.21,'track_list':LTS_FSI_list},
        'syn_weight':1,
        'dynamics_params':'LTS2FSI.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'LTS2FSI_rec': {
        'iterator':'one_to_all',
        'connection_rule':recurrent_connector_o2a,
        'connection_params':{'p':1, 'all_edges':LTS_FSI_list},
        'syn_weight':1,
        'dynamics_params':'FSI2LTS.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'CP2CS': {
        'connection_rule':perc_conn,
        'connection_params':{'p':0.01},
        'syn_weight':1,
        'dynamics_params':'CP2CS.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'CS2CP': {
        'connection_rule':perc_conn,
        'connection_params':{'p':0.1},
        'syn_weight':1,
        'dynamics_params':'CS2CS.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'CP2CP_uni': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.10, 'track_list':CP_CP_list},
        'syn_weight':1,
        'dynamics_params':'CP2CP.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'CP2CP_rec': {
        'iterator':'one_to_all',
        'connection_rule':recurrent_connector_o2a,
        'connection_params':{'p':0.3, 'all_edges':CP_CP_list},
        'syn_weight':1,
        'dynamics_params':'CP2CP.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'CS2CS_uni': {
        'iterator':'one_to_all',
        'connection_rule':syn_percent_o2a,
        'connection_params':{'p':0.1, 'track_list':CS_CS_list},
        'syn_weight':1,
        'dynamics_params':'CS2CS.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'CS2CS_rec': {
        'iterator':'one_to_all',
        'connection_rule':recurrent_connector_o2a,
        'connection_params':{'p':0.3, 'all_edges':CS_CS_list},
        'syn_weight':1,
        'dynamics_params':'CS2CS.json',
        'distance_range':[min_conn_dist,max_conn_dist],
        'target_sections': ['dend']
    },
    'Thal2CP': {
        'connection_rule':one_to_one_thal,
        'connection_params':{'offset': 100},
        'syn_weight':1,
        'dynamics_params':'Thal2CP.json',
        'distance_range':[min_conn_dist, max_conn_dist],
        'delay':0.0,
        'target_sections': ['dend']
    },
    'Thal2CS': {
        'connection_rule':one_to_one_thal,
        'connection_params':{'offset': 100},
        'syn_weight':1,
        'dynamics_params':'Thal2CS.json',
        'distance_range':[min_conn_dist, max_conn_dist],
        'delay':0.0,
        'target_sections': ['dend']
    },
    'Intbase2FSI': {
        'connection_rule':one_to_one_intbase,
        'connection_params':{'offset1': 400, 'offset2': 800},
        'syn_weight':1,
        'dynamics_params':'Intbase2FSI.json',
        'distance_range':[min_conn_dist, max_conn_dist],
        'delay':0.0,
        'target_sections': ['dend']
    },
    'Intbase2LTS': {
        'connection_rule':one_to_one_intbase,
        'connection_params':{'offset1': 400,'offset2': 800},
        'syn_weight':1,
        'dynamics_params':'Intbase2LTS.json',
        'distance_range':[min_conn_dist, max_conn_dist],
        'delay':0.0,
        'target_sections': ['dend']
    }
} # edges referenced by name

# Will be called by conn.add_properties for the associated connection
edge_add_properties = {
    'syn_dist_delay_feng_section_default': {
        'names':['delay','sec_id','sec_x'],
        'rule':syn_dist_delay_feng_section,
        'rule_params':{'sec_x':0.9},
        'dtypes':[np.float, np.int32, np.float]
    },
    'syn_uniform_delay_section_default': {
        'names':['delay','sec_id','sec_x'],
        'rule':syn_uniform_delay_section,
        'rule_params':{'sec_x':0.9},
        'dtypes':[np.float, np.int32, np.float]
    },
    'section_id_placement': {
        'names':['sec_id','sec_x'],
        'rule':section_id_placement,
        'dtypes':[np.int32, np.float]
    }
}

################################################################################
############################  EDGE EFFECTS EDGES  ##############################

if edge_effects:
    # These rules are for edge effect edges. They should directly mimic the connections
    # created previously, re-use the params set above. This keeps our code DRY
    virt_edges = [
    {   # FSI -> FSI Unidirectional
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> FSI Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> FSI Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'FSI2FSI_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CP Unidirectional
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'FSI2CP_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CP Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'FSI2CP_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CP Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CP']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'FSI2CP_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CS Unidirectional 
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'FSI2CS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CS Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'FSI2CS_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> CS Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CS']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'FSI2CS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> LTS Unidirectional 
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['LTS']}
        },
        'param': 'FSI2LTS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> LTS Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['LTS']}
        },
        'param': 'FSI2LTS_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # FSI -> LTS Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'FSI2LTS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CP Unidirectional
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'LTS2CP_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CP Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'LTS2CP_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CP Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CP']}, 
            'target':{'pop_name': ['LTS']}
        },
        'param': 'LTS2CP_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CS Unidirectional 
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'LTS2CS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CS Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'LTS2CS_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> CS Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CS']}, 
            'target':{'pop_name': ['LTS']}
        },
        'param': 'LTS2CS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> FSI Unidirectional
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'LTS2FSI_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> FSI Uniback
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['LTS']}, 
            'target':{'pop_name': ['FSI']}
        },
        'param': 'LTS2FSI_uniback',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # LTS -> FSI Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['FSI']}, 
            'target':{'pop_name': ['LTS']}
        },
        'param': 'LTS2FSI_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> CS
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CP']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'CP2CS',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> CP
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CS']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'CS2CP',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> CP Unidirectional
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CP']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'CP2CP_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CP -> CP Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CP']}, 
            'target':{'pop_name': ['CP']}
        },
        'param': 'CP2CP_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> CS Unidirectional
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CS']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'CS2CS_uni',
        'add_properties': 'syn_dist_delay_feng_section_default'
    },
    {   # CS -> CS Reciprocal
        'network':'cortex',
        'edge': {
            'source':{'pop_name': ['CS']}, 
            'target':{'pop_name': ['CS']}
        },
        'param': 'CS2CS_rec',
        'add_properties': 'syn_dist_delay_feng_section_default'
    }, 

        ################### THALAMIC INPUT ################

    {   # Thalamus Excitation to CP
        'network':'cortex',
        'edge': {
            'source':networks['thalamus'].nodes(),
            'target':networks['cortex'].nodes(pop_name=['CP'])
        },
        'param': 'Thal2CP'
    },
    {   # Thalamus Excitation to CS
        'network':'cortex',
        'edge': {
            'source':networks['thalamus'].nodes(),
            'target':networks['cortex'].nodes(pop_name=['CS'])
        },
        'param': 'Thal2CS'
    },

        ##################### Interneuron baseline INPUT #####################

    {    # To all FSI
        'network':'cortex',
        'edge': {
            'source':networks['Intbase'].nodes(),
            'target':networks['cortex'].nodes(pop_name=['FSI'])
        },
        'param': 'Intbase2FSI'        
    },
    {    # To all LTS
        'network':'cortex',
        'edge': {
            'source':networks['Intbase'].nodes(),
            'target':networks['cortex'].nodes(pop_name=['LTS'])
        },
        'param': 'Intbase2LTS'        
    },
] 
edge_definitions = edge_definitions 
##########################################################################
########################## END EDGE EFFECTS ##############################

############################ GAP JUNCTIONS ###############################
net = NetworkBuilder("cortex")
conn = net.add_gap_junctions(source={'pop_name': 'FSI'}, target={'pop_name': 'FSI'},
            resistance = 1500, target_sections=['somatic'], 
            connection_rule=perc_conn,
            connection_params={'p':0.4})
conn._edge_type_properties['sec_id'] = 0
conn._edge_type_properties['sec_x'] = 0.9

net = NetworkBuilder("cortex")
conn = net.add_gap_junctions(source={'pop_name': 'LTS'}, target={'pop_name': 'LTS'},
            resistance = 1500, target_sections=['somatic'], 
            connection_rule=perc_conn,
            connection_params={'p':0.3})
conn._edge_type_properties['sec_id'] = 0
conn._edge_type_properties['sec_x'] = 0.9


##########################################################################
###############################  BUILD  ##################################

# Load synapse dictionaries
# see synapses.py - loads each json's in components/synaptic_models into a 
# dictionary so the properties can be referenced in the files eg: syn['file.json'].get('property')
synapses.load()
syn = synapses.syn_params_dicts()

# Build your edges into the networks
build_edges(networks, edge_definitions,edge_params,edge_add_properties,syn)

# Save the network into the appropriate network dir
save_networks(networks,network_dir)

# Usually not necessary if you've already built your simulation config
build_env_bionet(base_dir='./',
		network_dir=network_dir,
		tstop=t_sim, dt = dt,
		report_vars = ['v'],
        celsius = 31.0,
		spikes_inputs=[('thalamus','./input/thalamus_base.h5'),('thalamus','./input/thalamus_short.h5'),('thalamus','./input/thalamus_long.h5'),  # Name of population which spikes will be generated for, file
                       ('Intbase','./input/Intbase.h5')],
		components_dir='components',
        config_file='config.json',
		compile_mechanisms=False)
