from intersection import *
from pdg import *
import pdg

import argparse # parser for command line options
import glob
from timeit import default_timer as timer
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.set_printoptions(precision = 2,linewidth = 200)

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
                    help='filename to save') 

parser.add_argument('-ss', "--stepsize", type=float, default = 0.1,
                    help='specifies step size of gradient descent optimizer') 

parser.add_argument('-sr', "--samplerandom", type=str, default = 'false', # not implemented for epochs
                    help='if true, pick new random games for each step') 
                 
parser.add_argument('-re', "--repetitions", type=int, default = 7,
                    help='how many times to repeat layout') 

parser.add_argument('-ep', "--epochs", type=str, default = 'false',
                    help='uses epochs') 

parser.add_argument('-epn', "--epochssize", type=int, default = 10, # actually specifies how many epochs there will be
                    help='number of epochs') 
     
parser.add_argument('-cg', "--controlgate", type=str, default = 'x',
                    help='Use different gate: \
                        "x": CNOT \
                        "z": CZ \
                        "crx": CRX \
                        "crz": CRZ \
                        "cry": CRY') 

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
            gen_games_sample(args.points, output = args.data)
        else:
            gen_games_sample(args.points, output = args.data) # create data file with specified name and size (# of points)
    data_name = args.data

if args.controlgate == 'rx':
    pdg.rotation_2q = True
    pdg.gate_2q = qml.CRX

elif args.controlgate == 'rz':
    pdg.rotation_2q = True
    pdg.gate_2q = qml.CRZ

elif args.controlgate == 'ry':
    pdg.rotation_2q = True
    pdg.gate_2q = qml.CRY

elif args.controlgate == 'x':
    pdg.rotation_2q = False
    pdg.gate_2q = qml.CNOT

elif args.controlgate == 'z':
    pdg.rotation_2q = False
    pdg.gate_2q = qml.CZ

else:
    raise TypeError

if pdg.rotation_2q:
    pdg.args_symmetric = {'c': 2, 'e': 2, 'o': 2, 'm': 2, 'i': 1, 'd': 1, 'z': 3, 'x':2}
    pdg.args_asymmetric = {'c': 8, 'e': 8, 'o': 8, 'm': 2, 'i': 4, 'd': 4, 'z': 9, 'x':2}
else:
    pdg.args_symmetric = {'c': 2, 'e': 2, 'o': 0, 'm': 2, 'i': 0, 'd': 0, 'z': 0, 'x':2}
    pdg.args_asymmetric = {'c': 8, 'e': 8, 'o': 0, 'm': 2, 'i': 0, 'd': 0, 'z': 0, 'x':2}
#######################

#filename = args.foldername + '/' + '-'.join(f'{k}={v}' for k, v in vars(args).items()) + f'-TIME{int(time.time())}'
filename = args.foldername + '/' + f'TIME{int(time.time())}' + '-'+str(round(np.random.uniform(), 3))

start = timer()
exp = road_detection(symmetric=str2bool(args.symmetric), sample_size=args.points, data_file=data_name, design=args.layout)

exp.random_parameters(repetitions=args.repetitions) # select best of 20 random points as starting point
if str2bool(args.epochs):
    exp.run_epochs(args.epochssize, args.points, args.num_steps, args.stepsize, data_name)
else:
    exp.run_lbfgs(args.num_steps, args.stepsize)

exp.check_accuracy()
end = timer()
exp.save(filename, end - start)
print('Total execution time: {} s'.format(end - start))