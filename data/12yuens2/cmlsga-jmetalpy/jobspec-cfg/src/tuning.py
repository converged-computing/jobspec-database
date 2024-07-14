import numpy as np
import os
import shutil
import time
import sys
import json

from cmlsga.algorithms.genetic_algorithms import *
from cmlsga.algorithms.particle_swarm_optimisation import *

from cmlsga.problems.wfg import *
from cmlsga.problems.uf import *
from cmlsga.problems.imb import *
from cmlsga.problems.dascmop import *
from cmlsga.problems.mop import *

from jmetal.core.quality_indicator import InvertedGenerationalDistance
from jmetal.lab.experiment import *
from jmetal.problem.multiobjective.zdt import *
from jmetal.problem.multiobjective.dtlz import *
from jmetal.problem.multiobjective.lz09 import *

from jmetal.util.evaluator import MapEvaluator

from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant, CategoricalHyperparameter, UniformFloatHyperparameter

def generate_summary_from_experiment(
    input_dir, quality_indicators, reference_fronts="", summary_filename="QualityIndicatorSummary.csv"
):

    if not quality_indicators:
        quality_indicators = []

    with open(summary_filename, "w+") as of:
        of.write("Algorithm,Problem,ExecutionId,IndicatorName,IndicatorValue\n")

    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            try:
                # Linux filesystem
                algorithm, problem = dirname.split("/")[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem = dirname.split("\\")[-2:]

            if "TIME" in filename:
                run_tag = [s for s in filename.split(".") if s.isdigit()].pop()

                with open(os.path.join(dirname, filename), "r") as content_file:
                    content = content_file.read()

                with open(summary_filename, "a+") as of:
                    of.write(",".join([algorithm, problem, run_tag, "Time", str(content)]))
                    of.write("\n")

            if "FUN" in filename:
                solutions = read_solutions(os.path.join(dirname, filename))
                run_tag = [s for s in filename.split(".") if s.isdigit()].pop()
                for indicator in quality_indicators:
                    reference_front_file = os.path.join(reference_fronts, problem + ".pf")

                    # Add reference front if any
                    if hasattr(indicator, "reference_front"):
                        if Path(reference_front_file).is_file():
                            reference_front = []
                            with open(reference_front_file) as file:
                                for line in file:
                                    reference_front.append([float(x) for x in line.split()])

                            indicator.reference_front = reference_front
                        else:
                            logger.warning("Reference front not found at", reference_front_file)

                    result = indicator.compute([solutions[i].objectives for i in range(len(solutions))])

                    # Save quality indicator value to file
                    with open(summary_filename, "a+") as of:
                        of.write(",".join([algorithm, problem, run_tag, indicator.get_short_name(), str(result)]))
                        of.write("\n")

def tae(cfg):
    max_evaluations = cfg["evaluations"]
    runs = cfg["runs"]
    output_path = cfg["outpath"]

    algorithm = globals()[cfg["algorithm"]]
    problem = parse_problem(cfg["problem"])
    algo_type = cfg["type"]

    population_size = cfg["population"]
    mutation_rate = cfg["mutation"]
    crossover_rate = cfg["crossover"]
    leaders = cfg["leaders"]
    e = MapEvaluator(processes=2)

    constructor = 0
    kwargs = 0

    if algo_type == "ga":
        constructor, kwargs = algorithm(problem, population_size, max_evaluations, e, mutation_rate, crossover_rate)
    elif algo_type == "pso": 
        constructor, kwargs = algorithm(problem, population_size, max_evaluations, e, mutation_rate, leaders)

    algorithm = constructor(**kwargs)


    jobs = [Job(algorithm, algorithm.get_name(), problem.get_name(), run)
            for run in range(runs)]
    experiment = Experiment(output_path, jobs)
    experiment.run()

    summary_filename = output_path + "/QualityIndicatorSummary.csv"
    generate_summary_from_experiment(output_path, [InvertedGenerationalDistance(None)],
                                     reference_fronts="resources/reference_front",
                                     summary_filename=summary_filename)

    igd = compute_mean_indicator(summary_filename, "IGD")

    # Cleanup
    os.remove(summary_filename)
    shutil.rmtree(output_path)

    # 1 - igd to minimise
    return 1 - igd


def create_configuration(algorithm, problem, algorithm_type, output_path, evaluations, runs):
    cs = ConfigurationSpace()

    # Constants
    cs.add_hyperparameter(Constant("problem", problem))
    cs.add_hyperparameter(Constant("algorithm", algorithm))
    cs.add_hyperparameter(Constant("type", algo_type))
    cs.add_hyperparameter(Constant("outpath", outpath))
    cs.add_hyperparameter(Constant("evaluations", evaluations))
    cs.add_hyperparameter(Constant("runs", runs))

    # Parameters to tune
    moead_pops = [100,300,400,500,600,800,1000]
    pops = [i for i in range(100, 1100, 100)]

    if algorithm == "moead":
        pops = moead_pops

    if algo_type == "ga":
        cs.add_hyperparameter(CategoricalHyperparameter("population", pops))
        cs.add_hyperparameter(UniformFloatHyperparameter("crossover", 0.1, 1.0))
        cs.add_hyperparameter(UniformFloatHyperparameter("mutation", 0.0, 0.2))
    elif algo_type == "pso":
        cs.add_hyperparameter(CategoricalHyperparameter("population", pops))
        cs.add_hyperparameter(UniformFloatHyperparameter("mutation", 0.0, 0.2))
        cs.add_hyperparameter(CategoricalHyperparameter("leaders", pops))

    return cs


def parse_problem(problem):
    problem_string = problem.split("(")

    if len(problem_string) == 1:
        return globals()[problem]()

    else: #DASCMOP case
        difficulty = int(problem_string[1][0])
        return globals()[problem_string[0]](difficulty)

    

# Usage:
# python tuning.py [algorithm] [problem] [algorithm_type] [evaluations] [runs]
if __name__ == "__main__":

    algorithm = sys.argv[1]
    problem = sys.argv[2]
    algo_type = sys.argv[3]
    evaluations = int(sys.argv[4])
    runs = int(sys.argv[5])

    mainpath = "/ssdfs/users/sy6u19/smac/{}/{}".format(algorithm, problem)
    outpath = mainpath + "/data-smac-100k"

    cs = create_configuration(algorithm, problem, algo_type, outpath, evaluations, runs)

    scenario = Scenario(
	{
	    "output_dir": mainpath,
	    "run_obj": "quality",
	    "runcount-limit": 100,
	    "cs": cs,
	    "shared_model": True,
	    "input_psmac_dirs": mainpath,
            "wallclock_limit": 43200
	}
    )

    smac = SMAC4AC(
	scenario=scenario,
	rng=np.random.RandomState(5),
	tae_runner=tae,
    )

    best = smac.optimize()

    with open("smac-parameters.csv", "a") as smac_output:
	# Algorithm, Problem, Population size, Crossover, Mutation, Leaders (for pso)
        smac_output.write("{},{},{},{},{},{}\n".format(
	    algorithm, problem, best["population"], best["crossover"], best["mutation"], best["leaders"]
        ))

    print("Algorithm: {}, Problem: {}".format(algorithm, problem))
    print("best: population: {}, crossover: {}, mutation: {}, leaders: {}".format(
	best["population"], best["crossover"], best["mutation"], best["leaders"]
    ))
