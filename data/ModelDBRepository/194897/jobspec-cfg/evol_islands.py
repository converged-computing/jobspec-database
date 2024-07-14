#! /usr/bin/env python
# runpbs_evol.py
# runs evolutionary algorithm on arm2dms model using PBS Torque in HPC

import os, sys
from numpy import mean
import csv
import copy
from inspyred.ec.variators import mutator
from random import Random
from time import time, sleep
import inspyred
import logging
from popen2 import popen2
import pickle
import multiprocessing
import queue
import subprocess

ngen = -1 #global variable keeping number of generations

###############################################################################
### Simulation options
###############################################################################
evolAlgorithm = 'evolutionStrategyCross' #'diffEvolution' # 'evolutionStrategy' #'krichmarCustom' #'genetic'#'particleSwarm100'#'estimationDist' #'diffEvolution' # 'evolutionStrategy' # 'krichmarCustom', 'genetic'
simdatadir = '../data/16jul26_'+evolAlgorithm # folder to save sim results

num_islands = 6 # number of islands
numproc = 16 # number of cores per job
musculoskeletalArm = 1 # need to know cause requires extra core
max_migrants = 1 #
migration_interval = 5
pop_size = 10 # population size per island
num_elites = pop_size/10 # num of top individuals kept each generation - maybe set to pop_size/10?
max_generations = 2000
max_evaluations = max_generations *  num_islands * pop_size
targets_eval = [0] # center-out reaching target to evaluate

# parameter names and ranges
pNames = []
pRanges = []

pNames.append('trainTime'); pRanges.append([30*1e3,180*1e3]) # int (round to nearest 1000)
pNames.append('stdpwin'); pRanges.append([10,50])
pNames.append('eligwin'); pRanges.append([50,150])
pNames.append('RLfactor'); pRanges.append([0.01,0.1])
pNames.append('RLinterval'); pRanges.append([50,100])
pNames.append('backgroundrate'); pRanges.append([50,150])
pNames.append('explorMovsFactor'); pRanges.append([0.1,5])
pNames.append('cmdmaxrate'); pRanges.append([500,2000])
pNames.append('PMdconnweight'); pRanges.append([0.5,4])
pNames.append('PMdconnprob'); pRanges.append([1,8])

num_inputs = len(pNames)

# evol specific params
mutation_rate = 0.4 # only for custom EC
crossover_rate = 0.2 # % of children with crossover
ux_bias = round(0.1*num_inputs)


# Set bounds and allowed ranges for params
def bound_params(candidate, args = []):
    cBound = []
    for i,p in enumerate(candidate):
        cBound.append(max(min(p, max(pRanges[i])), min(pRanges[i])))

    # need to be integer
    cBound[0] = round(max(min(candidate[0], max(pRanges[0])), min(pRanges[0]))/1000.0)*1000.0 # round to 1000.0
    #cBound[1] = round(max(min(candidate[1], max(pRanges[1])), min(pRanges[1])))
    #cBound[10] = round(max(min(candidate[10], max(pRanges[10])), min(pRanges[10])))

    # fixed values from list
    #param14 = min(param14_range, key=lambda x:abs(x-c[13]))

    candidate = cBound
    return candidate


###############################################################################
### Generate new set of random values for params
###############################################################################
def generate_rastrigin(random, args):
    size = args.get('num_inputs', 10)
    paramsRand = []
    for iparam in range(len(pNames)):
        paramsRand.append(random.uniform(min(pRanges[iparam]),max(pRanges[iparam])))

    # need to be integer
    paramsRand[0] = round(paramsRand[0]/1000.0)*1000.0
    #paramsRand[1] = round(paramsRand[1])
    #paramsRand[10] = round(paramsRand[10])

    # fixed values from list
    #param[14] = min(param14_range, key=lambda x:abs(x-param14))

    return paramsRand


###############################################################################
### Observer
###############################################################################
def my_observer(population, num_generations, num_evaluations, args):
    #ngen=num_generations
    best = max(population)
    print(('{0:6} -- {1} : {2}'.format(num_generations,
                                      best.fitness,
                                      str(best.candidate))))

###############################################################################
### Custom mutator (nonuniform taking into account bounds)
###############################################################################
@mutator
def nonuniform_bounds_mutation(random, candidate, args):
    """Return the mutants produced by nonuniform mutation on the candidates.
    .. Arguments:
       random -- the random number generator object
       candidate -- the candidate solution
       args -- a dictionary of keyword arguments
    Required keyword arguments in args:
    Optional keyword arguments in args:
    - *mutation_strength* -- the strength of the mutation, where higher
      values correspond to greater variation (default 1)

    """
    #bounder = args['_ec'].bounder
    #num_gens = args['_ec'].num_generations
    lower_bound = [x[0] for x in pRanges]
    upper_bound = [x[1] for x in pRanges]
    strength = args.setdefault('mutation_strength', 1)
    exponent = strength
    mutant = copy.copy(candidate)
    for i, (c, lo, hi) in enumerate(zip(candidate, lower_bound, upper_bound)):
        if random.random() <= 0.5:
            new_value = c + (hi - c) * (1.0 - random.random() ** exponent)
        else:
            new_value = c - (c - lo) * (1.0 - random.random() ** exponent)
        mutant[i] = new_value
    mutant_bounded = bound_params(mutant)
    return mutant_bounded

###############################################################################
### Parallel evaluation
###############################################################################
def parallel_evaluation_pbs(candidates, args):
    global ngen, targets_eval
    simdatadir = args.get('simdatadir') # load params
    ngen += 1 # increase number of generations
    maxiter_wait=args.get('maxiter_wait',2000) #
    default_error=args.get('default_error',0.3)

    #run pbs jobs
    total_jobs = 0
    commandList = []
    for i, c in enumerate(candidates):

        outfilestem=simdatadir+"/gen_"+str(ngen)+"_cand_"+str(i) # set filename
        for itarget in targets_eval:
            with open('%s_params'% (outfilestem), 'w') as f: # save current candidate params to file
                pickle.dump(c, f)
            command = 'mpirun -machinefile %s/nodes%d -np %d nrniv -python -mpi main.py outfilestem="%s" targetid=%d'%(simdatadir, i+1, numproc, outfilestem, itarget) # set command to run


            for iparam, param in enumerate(c): # add all param names and values dynamically
                paramstring = ' %s=%r' % (pNames[iparam], param)
                command += paramstring

            command += ' > %s.run &' % (outfilestem)  # to save to file and run in background

            commandList.append(command)

            total_jobs+=1  # increase jobs

    job_name = simdatadir+"/gen_"+str(ngen) # set job name (for each gen)
    walltime = '00:20:00'
    nodes = pop_size
    coresPerNode = 24 #numproc+musculoskeletalArm
    email = 'salvadordura@gmail.com'
    mailType = 'END,FAIL' if (ngen==0 or ngen%5 == 0) else 'FAIL' # only send email for 1st individual and every 5 generations; or if fail
    project = 'csd403'

    job_string = """#!/bin/sh
#SBATCH -e %s.err# Name of stderr output file
#SBATCH --partition=compute    # submit to the 'large' queue for jobs > 256 nodes
#SBATCH -J %s       # Job name
#SBATCH -t %s            # Run time (hh:mm:ss)
#SBATCH --mail-user=%s
#SBATCH --mail-type=%s
#SBATCH -A %s              # Allocation name to charge job against
#SBATCH --nodes=%d              # Total number of nodes requested (24 cores/node)
#SBATCH --ntasks-per-node=%d    # Total (?) number of mpi tasks requested; see also below: --npernode; CIPRES_THREADSPP; CIPRES_NP
#SBATCH --export=ALL
#SBATCH --switches=1
##SBATCH --res=nsguser_350
#SBATCH --res=salvadord_371

##SBATCH --qos=nsg

module purge
module load intel
export MODULEPATH=/share/apps/compute/modulefiles/mpi:$MODULEPATH
module load openmpi_ib/1.8.4npmi
module load python
module load gsl
module load scipy
module load gnu
module load mkl

export PATH=~nsguser/applications/neuron7.4/installdir/x86_64/bin:~nsguser/.local/bin:$PATH
export LD_LIBRARY_PATH=~nsguser/applications/neuron7.4/installdir/x86_64/lib:$LD_LIBRARY_PATH

export SLURM_NODEFILE=`generate_pbs_nodefile`
cat $SLURM_NODEFILE | uniq > %s/nodeslist
awk 'NR==1 {print $0" slots=17"}' %s/nodeslist > %s/nodes1
awk 'NR==2 {print $0" slots=17"}' %s/nodeslist > %s/nodes2
awk 'NR==3 {print $0" slots=17"}' %s/nodeslist > %s/nodes3
awk 'NR==4 {print $0" slots=17"}' %s/nodeslist > %s/nodes4
awk 'NR==5 {print $0" slots=17"}' %s/nodeslist > %s/nodes5
awk 'NR==6 {print $0" slots=17"}' %s/nodeslist > %s/nodes6
awk 'NR==7 {print $0" slots=17"}' %s/nodeslist > %s/nodes7
awk 'NR==8 {print $0" slots=17"}' %s/nodeslist > %s/nodes8
awk 'NR==9 {print $0" slots=17"}' %s/nodeslist > %s/nodes9
awk 'NR==10 {print $0" slots=17"}' %s/nodeslist > %s/nodes10


cd '/home/salvadord/m1ms/sim/'

""" % (job_name, job_name, walltime, email, mailType, project, nodes, coresPerNode, simdatadir,
    simdatadir, simdatadir, simdatadir, simdatadir, simdatadir,
    simdatadir, simdatadir, simdatadir, simdatadir, simdatadir,
    simdatadir, simdatadir, simdatadir, simdatadir, simdatadir,
    simdatadir, simdatadir, simdatadir, simdatadir, simdatadir)


    print(job_string) # print sbatch script

    batchfile = '%s/gen_%d.sbatch'%(simdatadir, ngen)
    with open(batchfile, 'w') as text_file:
        text_file.write("%s" % job_string)
        for comm in commandList:
            text_file.write("\n%s\n" % comm)
        text_file.write("\nwait \n")

    #subprocess.call
    output, pinput = popen2('sbatch '+batchfile) # Open a pipe to the qsub command.
    pinput.close()


    #read results from file
    targetFitness = [[None for j in targets_eval] for i in range(len(candidates))]
    num_iters = 0
    jobs_completed=0
    while jobs_completed < total_jobs:
        #print outfilestem
        print(str(jobs_completed)+" / "+str(total_jobs)+" jobs completed")
        unfinished = [[(i,j) for j,y in enumerate(x) if y is None] for i, x in enumerate(targetFitness)]
        unfinished = [item for sublist in unfinished for item in sublist]
        print("unfinished:"+str(unfinished))
        for (icand,itarget) in unfinished:
            # load error from file
            try:
                outfilestem=simdatadir + "/gen_" + str(ngen) + "_cand_" + str(icand) + "_target_" + str(itarget) # set filename
                with open('%s_error'% (outfilestem)) as f:
                    errorDic=pickle.load(f)
                    targetFitness[icand][itarget] = errorDic['errorFitness']
                    jobs_completed+=1
                    print("icand:",icand," itarget:",itarget," error: "+str(errorDic['errorFitness']))
            except:
                pass
            #print "Waiting for job: "+str(i)+" ... iteration:"+str(num_iters[i])
        num_iters+=1
        if num_iters>=maxiter_wait: #or (num_iters>maxiter_wait/2 and jobs_completed>(0.95*total_jobs)):
            print("max iterations reached -- remaining jobs set to default error")
            for (icand,itarget) in unfinished:
                targetFitness[icand][itarget] = default_error
                jobs_completed+=1
        sleep(2) # sleep 2 seconds before checking agains
    print(targetFitness)
    try:
        fitness = [mean(x) for x in targetFitness]
    except:
        fitness = [default_error for x in range(len(candidates))]
    print('fitness:',fitness)
    return fitness


###############################################################################
### Multiprocessing Migration
###############################################################################
class MultiprocessingMigratorNoBlock(object):
    """Migrate among processes on the same machine.
      remove lock
    """
    def __init__(self, max_migrants=1, migration_interval=10):
        self.max_migrants = max_migrants
        self.migration_interval = migration_interval
        self.migrants = multiprocessing.Queue(self.max_migrants)
        self.__name__ = self.__class__.__name__

    def __call__(self, random, population, args):
        # only migrate every migrationInterval generations
        if (args["_ec"].num_generations % self.migration_interval)==0:
            evaluate_migrant = args.setdefault('evaluate_migrant', False)
            migrant_index = random.randint(0, len(population) - 1)
            old_migrant = population[migrant_index]
            try:
                migrant = self.migrants.get(block=False)
                if evaluate_migrant:
                    fit = args["_ec"].evaluator([migrant.candidate], args)
                    migrant.fitness = fit[0]
                    args["_ec"].num_evaluations += 1
            except queue.Empty:
                pass
            try:
                self.migrants.put(old_migrant, block=False)
            except queue.Full:
                pass
        return population


###############################################################################
### Set initial conditions (in case have to restart)
###############################################################################

def setInitial(simdatadir):
    global ngen
    # load individuals.csv file and set last population as initial_cs
    ind_gens=[]
    ind_cands=[]
    ind_fits=[]
    ind_cs=[]
    with open('%s/individuals.csv' % (simdatadir)) as f:
        reader=csv.reader(f)
        for row in reader:
            ind_gens.append(int(row[0]))
            ind_cands.append(int(row[1]))
            ind_fits.append(float(row[2]))
            cs = [float(row[i].replace("[","").replace("]","")) for i in range(3,len(row))]
            ind_cs.append(cs)

    initial_gen = max(max(ind_gens) - 1, 0)
    initial_cs = [ind_cs[i] for i in range(len(ind_gens)) if ind_gens[i]==initial_gen]
    initial_fit = [ind_fits[i] for i in range(len(ind_gens)) if ind_gens[i]==initial_gen]

    # set global variable to track number of gens to initial_gen
    ngen = initial_gen

    print(initial_gen, initial_cs, initial_fit)
    return initial_gen, initial_cs, initial_fit


###############################################################################
### Create islands
###############################################################################
def create_island(rand_seed, island_number, mp_migrator, simdatadir, max_evaluations, max_generations, \
    num_inputs, mutation_rate, crossover_rate, ux_bias, pop_size, num_elites):
    global num_islands

    # create folder
    if num_islands > 1:
        simdatadir = simdatadir+'_island_'+str(i)
    mdir_str='mkdir %s' % (simdatadir)
    os.system(mdir_str)

    # if individuals.csv already exists, continue from last generation
    if os.path.isfile(simdatadir+'/individuals.csv'): # disabled by adding '!!'
        initial_gen, initial_cs, initial_fit = setInitial(simdatadir)
    else:
        initial_gen=0
        initial_cs=[]
        initial_fit=[]

    statfile = open(simdatadir+'/statistics.csv', 'a')
    indifile = open(simdatadir+'/individuals.csv', 'a')

    #random nums and save seed
    my_seed = rand_seed #int(time())
    seedfile = open(simdatadir+'/randomseed.txt', 'a')
    seedfile.write('{0}'.format(my_seed))
    seedfile.close()
    prng = Random()
    prng.seed(my_seed)


    # Custom algorithm based on Krichmar's params
    if evolAlgorithm == 'customEvol':
        # a real-valued optimization algo- rithm called Evolution Strategies (De Jong, 2002)
        # was used with deterministic tournament selection, weak-elitism replacement, 40% Gaussian mutation and 50%
        # crossover. Weak-elitism ensures the overall fitness monotonically increases each generation by replacing the
        # worst fitness individual of the offspring population with the best fitness individual of the parent population.


        ea = inspyred.ec.EvolutionaryComputation(prng)
        ea.selector = inspyred.ec.selectors.tournament_selection
        ea.variator = [inspyred.ec.variators.uniform_crossover, nonuniform_bounds_mutation]
                       #inspyred.ec.variators.gaussian_mutation]
        ea.replacer = inspyred.ec.replacers.generational_replacement#inspyred.ec.replacers.plus_replacement
        #inspyred.ec.replacers.truncation_replacement (with num_selected=50)
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = [inspyred.ec.observers.stats_observer, inspyred.ec.observers.file_observer]
        final_pop = ea.evolve(generator=generate_rastrigin,
                              evaluator=parallel_evaluation_pbs,
                              pop_size=pop_size,
                              bounder=bound_params,
                              maximize=False,
                              max_evaluations=max_evaluations,
                              max_generations=max_generations,
                              num_inputs=num_inputs,
                              mutation_rate=mutation_rate,
                              crossover_rate=crossover_rate,
                              tournament_size=2,
                              num_selected=pop_size,
                              num_elites=num_elites,
                              simdatadir=simdatadir,
                              statistics_file=statfile,
                              individuals_file=indifile,
                              evaluate_migrant=False,
                              initial_gen=initial_gen,
                              initial_cs=initial_cs,
                              initial_fit=initial_fit)

    # Genetic
    elif evolAlgorithm == 'genetic':
        ea = inspyred.ec.GA(prng)
        if num_islands > 1: ea.migrator = mp_migrator
        ea.terminator = inspyred.ec.terminators.evaluation_termination
        ea.observer = [inspyred.ec.observers.stats_observer, inspyred.ec.observers.file_observer]
        final_pop = ea.evolve(generator=generate_rastrigin,
                            evaluator=parallel_evaluation_pbs,
                            pop_size=pop_size,
                            bounder=bound_params,
                            maximize=False,
                            max_evaluations=max_evaluations,
                            max_generations=max_generations,
                            num_inputs=num_inputs,
                            num_elites=num_elites,
                            simdatadir=simdatadir,
                            statistics_file=statfile,
                            individuals_file=indifile)

    # Evolution Strategy
    elif evolAlgorithm == 'evolutionStrategy':
        ea = inspyred.ec.ES(prng)
        if num_islands > 1: ea.migrator = mp_migrator
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = [inspyred.ec.observers.stats_observer, inspyred.ec.observers.file_observer]
        final_pop = ea.evolve(generator=generate_rastrigin,
                            evaluator=parallel_evaluation_pbs,
                            pop_size=pop_size,
                            bounder=bound_params,
                            maximize=False,
                            max_evaluations=max_evaluations,
                            max_generations=max_generations,
                            num_inputs=num_inputs,
                            num_elites=num_elites,
                            simdatadir=simdatadir,
                            statistics_file=statfile,
                            individuals_file=indifile,
                            initial_gen=initial_gen,
                            initial_cs=initial_cs,
                            initial_fit=initial_fit)

    # Evolution Strategy with crossover
    elif evolAlgorithm == 'evolutionStrategyCross':

        ea = inspyred.ec.ES(prng)
        if num_islands > 1: ea.migrator = mp_migrator
        ea.variator = [inspyred.ec.variators.uniform_crossover, ea._internal_variation]
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = [inspyred.ec.observers.stats_observer, inspyred.ec.observers.file_observer]
        final_pop = ea.evolve(generator=generate_rastrigin,
                            evaluator=parallel_evaluation_pbs,
                            pop_size=pop_size,
                            bounder=bound_params,
                            maximize=False,
                            max_evaluations=max_evaluations,
                            max_generations=max_generations,
                            num_inputs=num_inputs,
                            crossover_rate=crossover_rate,
                            ux_bias=ux_bias,
                            simdatadir=simdatadir,
                            statistics_file=statfile,
                            individuals_file=indifile,
                            initial_gen=initial_gen,
                            initial_cs=initial_cs,
                            initial_fit=initial_fit)



    # Simulated Annealing
    elif evolAlgorithm == 'simulatedAnnealing':
        ea = inspyred.ec.SA(prng)
        if num_islands > 1: ea.migrator = mp_migrator
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = [inspyred.ec.observers.stats_observer, inspyred.ec.observers.file_observer]
        final_pop = ea.evolve(generator=generate_rastrigin,
                            evaluator=parallel_evaluation_pbs,
                            pop_size=pop_size,
                            bounder=bound_params,
                            maximize=False,
                            max_evaluations=max_evaluations,
                            max_generations=max_generations,
                            num_inputs=num_inputs,
                            num_elites=num_elites,
                            simdatadir=simdatadir,
                            statistics_file=statfile,
                            individuals_file=indifile)

    # Differential Evolution
    elif evolAlgorithm == 'diffEvolution':
        ea = inspyred.ec.DEA(prng)
        if num_islands > 1: ea.migrator = mp_migrator
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = [inspyred.ec.observers.stats_observer, inspyred.ec.observers.file_observer]
        final_pop = ea.evolve(generator=generate_rastrigin,
                            evaluator=parallel_evaluation_pbs,
                            pop_size=pop_size,
                            bounder=bound_params,
                            maximize=False,
                            num_selected=pop_size,
                            max_evaluations=max_evaluations,
                            max_generations=max_generations,
                            num_inputs=num_inputs,
                            num_elites=num_elites,
                            simdatadir=simdatadir,
                            statistics_file=statfile,
                            individuals_file=indifile)

    # Estimation of Distribution
    elif evolAlgorithm == 'estimationDist':
        ea = inspyred.ec.EDA(prng)
        if num_islands > 1: ea.migrator = mp_migrator
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = [inspyred.ec.observers.stats_observer, inspyred.ec.observers.file_observer]
        final_pop = ea.evolve(generator=generate_rastrigin,
                            evaluator=parallel_evaluation_pbs,
                            pop_size=pop_size,
                            bounder=bound_params,
                            maximize=False,
                            max_evaluations=max_evaluations,
                            max_generations=max_generations,
                            num_inputs=num_inputs,
                            num_elites=num_elites,
                            num_offspring=pop_size,
                            simdatadir=simdatadir,
                            statistics_file=statfile,
                            individuals_file=indifile)


    # Particle Swarm optimization
    elif evolAlgorithm == 'particleSwarm':
        ea = inspyred.swarm.PSO(prng)
        if num_islands > 1: ea.migrator = mp_migrator
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = [inspyred.ec.observers.stats_observer, inspyred.ec.observers.file_observer]
        ea.topology = inspyred.swarm.topologies.ring_topology
        final_pop = ea.evolve(generator=generate_rastrigin,
                            evaluator=parallel_evaluation_pbs,
                            pop_size=pop_size,
                            num_offspring=pop_size,
                            num_selected=pop_size/2,
                            bounder=bound_params,
                            maximize=False,
                            max_evaluations=max_evaluations,
                            max_generations=max_generations,
                            num_inputs=num_inputs,
                            simdatadir=simdatadir,
                            statistics_file=statfile,
                            individuals_file=indifile,
                            neighborhood_size=5)

    # Ant colony optimization (requires components)
    elif evolAlgorithm == 'antColony':
        ea = inspyred.swarm.ACS(prng)
        if num_islands > 1: ea.migrator = mp_migrator
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = [inspyred.ec.observers.stats_observer, inspyred.ec.observers.file_observer]
        ea.topology = inspyred.swarm.topologies.ring_topology
        final_pop = ea.evolve(generator=generate_rastrigin,
                            evaluator=parallel_evaluation_pbs,
                            pop_size=pop_size,
                            bounder=bound_params,
                            maximize=False,
                            max_evaluations=max_evaluations,
                            max_generations=max_generations,
                            num_inputs=num_inputs,
                            simdatadir=simdatadir,
                            statistics_file=statfile,
                            individuals_file=indifile)



    best = max(final_pop)
    print(('Best Solution: \n{0}'.format(str(best))))

    return ea


###############################################################################
### Main - logging, island model params, launch multiprocessing
###############################################################################
if __name__ == '__main__':
    # create folder
    mdir_str='mkdir -p %s' % (simdatadir)
    os.system(mdir_str)

    # debug info
    logger = logging.getLogger('inspyred.ec')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(simdatadir+'/inspyred.log', mode='a')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # run single population or multiple islands
    rand_seed = int(time())
    if num_islands == 1:
        create_island(rand_seed, 1, [], simdatadir, max_evaluations, max_generations, num_inputs, mutation_rate, crossover_rate, ux_bias, pop_size, num_elites)
    else:
        mp_migrator = MultiprocessingMigratorNoBlock(max_migrants, migration_interval)
        jobs = []
        for i in range(num_islands):
            p = multiprocessing.Process(target=create_island, args=(rand_seed + i, i, mp_migrator, simdatadir, \
             max_evaluations, max_generations, num_inputs, mutation_rate, crossover_rate, ux_bias, pop_size, num_elites))
            p.start()
            jobs.append(p)
        for j in jobs:
            j.join()
