from neuron import h
import os, sys
from bmtk.simulator import bionet
import numpy as np
import synapses
import warnings
from bmtk.simulator.core import simulation_config 
import h5py
from bmtk.simulator.bionet import synaptic_weight

def run(config_file):
    #warnings.simplefilter(action='ignore', category=FutureWarning)
    synapses.load()
    #from bmtk.simulator.bionet.pyfunction_cache import add_weight_function
    conf = bionet.Config.from_json(config_file, validate=True)



    conf.copy_to_output()
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)

    

    # This fixes the morphology error in LFP calculation
    pop = graph._node_populations['biophysical']
    for node in pop.get_nodes():
         node._node._node_type_props['morphology'] = node.model_template[1]

    sim = bionet.BioSimulator.from_config(conf, network=graph)
    

    #sim.add_mod(bionet.modules.save_synapses.SaveSynapses('updated_conns'))


    # This calls insert_mechs() on each cell to use its gid as a seed
    # to the random number generator, so that each cell gets a different
    # random seed for the point-conductance noise
    cells = graph.get_local_cells()
    for cell in cells:
        cells[cell].hobj.insert_mechs(cells[cell].gid)
        pass


    sim.run()
    

    
    bionet.nrn.quit_execution()

run('simulation_configLFP.json')
"""
if __name__ == '__main__':
    run('simulation_config.json')
    if __file__ != str(os.path.join(os.getcwd(),sys.argv[-1])):
        run(sys.argv[-1])
    else:
        run('simulation_config.json')
"""