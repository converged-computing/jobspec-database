import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import os
import gc 
import sys
from decpomdp import DecPOMDP
from constant import Constants

gc.enable()

# how to run this file 
# example :
# python experiment_server.py problem=dectiger horizon=3 iterations=3 algo=tabular

#set problem 
if len(sys.argv) < 2:
    print("err0r : not enough arguments given")
    sys.exit(1)
if len(sys.argv)> 5 :
    file_name = str(sys.argv[1]).split("=")[1]
    planning_horizon = int(sys.argv[2].split("=")[1])
    num_iterations = int(sys.argv[3].split("=")[1])
    density = float(sys.argv[4].split("=")[1])
    algorithm = sys.argv[5].split("=")[1]
    problem = DecPOMDP(file_name,horizon=planning_horizon)

    #initialize problem
    Constants.initialize(problem)
    from experimentFunctions import Experiment
    experiment = Experiment(planning_horizon,problem,algorithm=algorithm)

    print(f"game size :\n\t|S| = {len(problem.states)}")
    print(f"\t|Z| = {problem.num_joint_observations}\n\t|U| = {problem.num_joint_actions} with |U_i| = {problem.num_actions[0]}")
    print(f"intiial_belief : {problem.actions}")

    database, matrix = experiment.run_experiments(num_iterations,density)

elif len(sys.argv)> 3 :
    file_name = str(sys.argv[1]).split("=")[1]
    planning_horizon = int(sys.argv[2].split("=")[1])
    num_iterations = int(sys.argv[3].split("=")[1])

    problem = DecPOMDP(file_name,horizon=planning_horizon)
    Constants.initialize(problem)
    from experimentFunctions import Experiment
    experiment = Experiment(planning_horizon,problem)

    print(f"game size :\n\t|S| = {len(problem.states)}")
    print(f"\t|Z| = {problem.num_joint_observations}\n\t|U| = {problem.num_joint_actions} with |U_i| = {problem.num_actions[0]}")
    print(f"intiial_belief : {problem.actions}")

    database, matrix = experiment.run_experiments(num_iterations,limit=1000)
    experiment.generate_summary_table()
    experiment.horizon_value_plot()
    experiment.plots()


else : 
    print("not enough arguments")
    sys.exit()

print("+++++    ++++++++++  +++++++++  end of experiment  ********  **********  ***")
