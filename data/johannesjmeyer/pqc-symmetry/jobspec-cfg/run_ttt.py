# %%
from tictactoe import *
from circuits_ttt import *
import circuits_ttt

import argparse # parser for command line options
import glob
from timeit import default_timer as timer
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.set_printoptions(precision = 2,linewidth = 200)
# %%

###############################################################
############## parse command line options
###############################################################

def str2bool(v): 
    # To deal with boolean arguments
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print(v)
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Set options for experiment')

# enforce symmetry
parser.add_argument('-s', "--symmetric", type=str, default = 'true',
                    help='if true, ttt symmetry is enforced')

# length of experiment                  
parser.add_argument('-n', "--num_steps", type=int, default = 50,
                    help='number of gradient descent steps') 

# file containing data points for this experiment                 
parser.add_argument('-d', "--data", type=str, default = 'None',
                    help='file of data points for training') 
                    # we feed the data file for reproducibility and facilitate comparison between symm/asymm

# number of data points infile to use             
parser.add_argument('-p', "--points", type=int, default = 6,
                    help='number of data points to use per step. If epochs are activated, the size of the training set is determined by the product of data points and number of steps.') 

# circuit design          
parser.add_argument('-l', "--layout", type=str, default = 'tcemoid',
                    help='string specifying order of encoding layers: \
                    \n t: encode game -- data encoding\
                    \n c: corners -- 1q trainable \
                    \n e: edges -- 1q trainable\
                    \n m: middle/center -- 1q trainable\
                    \n o: outer layer -- 2q trainable\
                    \n i: inner layer  -- 2q trainable\
                    \nd: diagonal layer -- 2q trainable') 
                
parser.add_argument('-f', "--foldername", type=str, default = 'output',
                    help='foldername to save') 

parser.add_argument('-ss', "--stepsize", type=float, default = 0.008,
                    help='specifies step size of gradient descent optimizer') 

parser.add_argument('-sr', "--samplerandom", type=str, default = 'false', # not implemented for epochs
                    help='if true, pick new random games for each step. Only relevant if no epochs are used') 

parser.add_argument('-w', "--wins", default = "012",
                    help='wins to include in dataset. Seperate numbers with a comma e.g. only including wins for -1 and 0 would look like -1,0. For completely random games set to "R"') 

parser.add_argument('-es', "--excludesymmetry", type=str, default = 'false',
                    help='if true, uses reduced dataset only keeps one game per symmetry group') 
                    
parser.add_argument('-re', "--repetitions", type=int, default = 7,
                    help='how many times to repeat layout') 

parser.add_argument('-ep', "--epochs", type=str, default = 'true',
                    help='uses Adam with epochs by default, when set to False lbfgs is used instead') 

parser.add_argument('-epn', "--epochssize", type=int, default = 10, # actually specifies how many epochs there will be
                    help='number of epochs') 

parser.add_argument('-lf', "--loss", type=str, default = 'mse',
                    help='specifies loss function use \
                        mse: mean squared error \
                        ce: cross entropy \
                        5q: 5 qubit mse') 
        
parser.add_argument('-cg', "--controlgate", type=str, default = 'x',
                    help='Use different gate: \
                        "x": CNOT \
                        "z": CZ \
                        "rx": CRX \
                        "rz": CRZ \
                        "ry": CRY') 

parser.add_argument('-sq', "--symmetryqubits", type=str, default = '024613578',
                    help='which qubits should be treated symmetrically when symmetric=True\
                        First 4, Second 4 qubits and last qubit are treated symetrically') 

args = parser.parse_args()
###############################################################
############## produce data (mostly useful for debugging)
###############################################################
if args.data == 'None':
    data_name = None
else:
    import os.path
    if not os.path.isfile(args.data): # if the specified data file does not exist
        if str2bool(args.epochs):
            gen_games_sample(args.points, output = args.data, truesize=True)
        else:
            gen_games_sample(args.points, output = args.data) # create data file with specified name and size (# of points)
    data_name = args.data

###############################################################
############## run experiment
###############################################################

filename = args.foldername + '/' + f'TIME{int(time.time())}' + '-'+str(round(np.random.uniform(), 3))

if 'R' in args.wins:
    wins=[]
else:
    wins = [int(i)-1 for i in args.wins]

start = timer()

# initialize simulation object
exp = tictactoe(symmetric=str2bool(args.symmetric), design=args.layout, \
    random_sample=str2bool(args.samplerandom), wins=wins, reduced=str2bool(args.excludesymmetry), loss_fn = args.loss, controlstring = args.controlgate, symmetrystring = args.symmetryqubits)

# create random initial parameters
exp.random_parameters(repetitions=args.repetitions)

# run optimization
if str2bool(args.epochs): # epochs are only implemented with Adam optimizer
    exp.run_epochs(epochs = args.epochssize, samplesize_per_step = args.points, steps_per_epoch = args.num_steps, stepsize = args.stepsize, data_file = data_name)
else: # use lbfgs if epochs is set to false
    exp.run_lbfgs(size = args.points, steps = args.num_steps, stepsize = args.stepsize, data_file = data_name)

end = timer()
exp.save(filename, end - start)

print('Total execution time: {} s'.format(end - start))
