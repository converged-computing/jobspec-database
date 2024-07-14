import os, sys
from bmtk.simulator import bionet
import numpy as np
import synapses
import warnings
#from bmtools.cli.plugins.util import util
from neuron import h

def populations(config):
    nodes = util.load_nodes_from_config(config)
    i=0
    j=0
    CP_nodes = []
    CS_nodes = []
    FSI_nodes = []
    LTS_nodes = []
    for j in nodes:
        node=nodes[j]
        num_cells=len(node['node_type_id'])
        if node['model_type'][0]=='biophysical':
            #print(node)
            #print(node['node_id'])
            num_cells=len(node['node_type_id'])
            for i in range(num_cells-1):
                if(node['pop_name'][i]=='CP'):
                    CP_nodes.append(i)
                if(node['pop_name'][i]=='CS'):
                    CS_nodes.append(i)
                if(node['pop_name'][i]=='FSI'):
                    FSI_nodes.append(i)
                if(node['pop_name'][i]=='LTS'):
                    LTS_nodes.append(i)
    return CP_nodes, CS_nodes, FSI_nodes, LTS_nodes 

def run(config_file):

    warnings.simplefilter(action='ignore', category=FutureWarning)
    synapses.load()
    from bmtk.simulator.bionet.pyfunction_cache import add_weight_function

    def gaussianBL(edge_props, source, target):
        w0 = edge_props["syn_weight"]
        sigma = edge_props["weight_sigma"]
        return np.random.normal(w0, sigma, 1)
	
    def lognormal(edge_props, source, target):
        m = edge_props["syn_weight"]
        s = edge_props["weight_sigma"]
        mean = np.log(m) - 0.5 * np.log((s/m)**2+1)
        std = np.sqrt(np.log((s/m)**2 + 1))
        return np.random.lognormal(mean, std, 1)

    add_weight_function(lognormal)
    add_weight_function(gaussianBL)


    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)

    


    # This fixes the morphology error in LFP calculation
    pop = graph._node_populations['cortex']
    for node in pop.get_nodes():
        node._node._node_type_props['morphology'] = node.model_template[1]

    sim = bionet.BioSimulator.from_config(conf, network=graph)
    
    # This is my attempt to record the point conductance current only
    #CP_nodes, CS_nodes, FSI_nodes, LTS_nodes=populations(config_file)
    #pick_index = FSI_nodes [10]  # Picks the eleventh (zero index) cell arbitrarily in the list of FSI cell indices.
    cells = graph.get_local_cells()
    #if pick_index in list(cells.keys()):
        #print(cells.get(pick_index))
        #cell = cells.get(pick_index)

        #soma = cell.hobj.soma[0](0.5) # Getting the soma out of the FSI cell
        #print(dir(soma.na_ion))
        #soma_i_exc = h.Vector()
        #soma_i_inh = h.Vector()
        #soma_i_exc.record(soma.noise_exc._ref_i_exc)

    # This calls insert_mechs() on each cell to use its gid as a seed
    # to the random number generator, so that each cell gets a different
    # random seed for the point-conductance noise
    cells = graph.get_local_cells()
    for cell in cells:
        cells[cell].hobj.insert_mechs(cells[cell].gid)
        pass
    
    sim.run()

    #i_exc=soma_i_exc.to_python()
    #print(i_exc)
    bionet.nrn.quit_execution()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config.json')
