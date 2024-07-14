from helper import parallelize, saveComplexJson
from agent import  AgentWithSingleAlgo, AgentWithEnsemble
from mazes import *
from experimentConf import *
import sys

NUM_STEPS = 10000
NUM_TRIALS = 50

NUM_PROCESSES = 32

if len(sys.argv) < 3:
    print("Not enough arguments, pass either 'single' or 'ensemble and the experiment number'")
    exit(0)

mode = sys.argv[1]
if mode not in ['single', 'ensemble', 'full']:
    print("Invalid mode, pass either 'single', 'ensemble' or 'full'")
    exit(0)

expNum = int(sys.argv[2])
if expNum not in [1, 2, 3, 4, 5]:
    print("Invalid experiment, pass either 2, 3, 4 or 5")
    exit(0)

resultsFilename = "results/%s-%d.json" % (mode, expNum)

if expNum == 1:
    NUM_STEPS = 50000
    NUM_TRIALS = 500
    algoParams = algoParamsListExp1
    algos = algosExp1
    ensembles = ensemblesExp1
    mazeGenerator = createSimpleMaze
    getBeliefState = None

if expNum == 2:
    algoParams = algoParamsListExp2
    algos = algosExp2
    ensembles = ensemblesExp2
    mazeGenerator = createDynamicStartMaze
    getBeliefState = initBeliefState

elif expNum == 3:
    algoParams = algoParamsListExp3
    algos = algosExp3
    ensembles = ensemblesExp3
    mazeGenerator = createDynamicObstaclesMaze
    getBeliefState = None

elif expNum == 4:
    algoParams = algoParamsListExp4
    algos = algosExp4
    ensembles = ensemblesExp4
    mazeGenerator = createDynamicGoalMaze
    getBeliefState = None
    NUM_STEPS = 50000
    NUM_TRIALS = 250


elif expNum == 5:
    algoParams = algoParamsListExp5
    algos = algosExp5
    ensembles = ensemblesExp5
    mazeGenerator = createGeneralizedMaze
    getBeliefState = None
    NUM_STEPS = 50000
    NUM_TRIALS = 250

print("Saving results of experiment %d in %s mode in filename %s" % (expNum, mode, resultsFilename))



def runTrialSingleAlgorithm(algorithm, params, numSteps, maze, beliefState):
    agent = AgentWithSingleAlgo(maze, algorithm, params, beliefState)

    # reward intake = reward moyen par mouvement
    # Il mesure deux choses
    # 1) Dans 2500 derniers épisodes, fait la moyenne du reward intake
    # 2) Tous les 2500 épisodes, regarde quel est le reward intake, puis à la fin il fait la somme
    allRewardIntakes, numberOfSteps = agent.learn(numSteps)
    return allRewardIntakes, numberOfSteps

def runTrialEnsemble(ensemble, algoParams, temp, numSteps, maze, beliefState):
    if expNum in [1, 4, 5]:
        agent = AgentWithEnsemble(maze, ensemble, algoParams, temp, beliefState, neural=False)
    else:  
        agent = AgentWithEnsemble(maze, ensemble, algoParams, temp, beliefState, neural=True)
    allRewardIntakes, numberOfSteps = agent.learn(numSteps)
    return allRewardIntakes, numberOfSteps


def addJobsSingleAlgorithm(jobs, pool):
    numSteps = NUM_STEPS
    for i in range(NUM_TRIALS):
        maze = mazeGenerator()
        
        if getBeliefState is None:
            beliefState = None
        else:
            beliefState = initBeliefState(maze.obstacles)

        for algorithmName, algorithm, algoParams in algos:
            jobs[(algorithmName, i)] = pool.apply_async(runTrialSingleAlgorithm, (algorithm, algoParams, numSteps, maze, beliefState))

def addJobsEnsemble(jobs, pool):
    numSteps = NUM_STEPS
    for i in range(NUM_TRIALS):
        maze = mazeGenerator()

        if getBeliefState is None:
            beliefState = None
        else:
            beliefState = initBeliefState(maze.obstacles)

        for ensembleName, ensemble, algoParams, temp in ensembles:
            jobs[(ensembleName, i)] = pool.apply_async(runTrialEnsemble, (ensemble, algoParams, temp, numSteps, maze, beliefState))

def addJobsFull(jobs, pool):
    numSteps = NUM_STEPS
    for i in range(NUM_TRIALS):
        maze = mazeGenerator()

        if getBeliefState is None:
            beliefState = None
        else:
            beliefState = initBeliefState(maze.obstacles)

        for ensembleName, ensemble, algoParams, temp in ensembles:
            jobs[(ensembleName, i)] = pool.apply_async(runTrialEnsemble, (ensemble, algoParams, temp, numSteps, maze, beliefState))

        for algorithmName, algorithm, algoParams in algos:
            jobs[(algorithmName, i)] = pool.apply_async(runTrialSingleAlgorithm, (algorithm, algoParams, numSteps, maze, beliefState))


#def jobDone(key):
#    algorithmName, i  = key
#    print("Done for algorithm %s (iteration %d)" % (algorithmName, i))

if mode == 'single':
    results = parallelize(addJobsSingleAlgorithm, numProcesses=NUM_PROCESSES)
    saveComplexJson(resultsFilename, results)
elif mode == 'ensemble':
    results = parallelize(addJobsEnsemble, numProcesses=NUM_PROCESSES)
    saveComplexJson(resultsFilename, results)
elif mode == 'full':
    results = parallelize(addJobsFull, numProcesses=NUM_PROCESSES)
    saveComplexJson(resultsFilename, results)
