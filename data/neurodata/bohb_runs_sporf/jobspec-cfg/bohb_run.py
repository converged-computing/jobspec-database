import os
import logging
logging.basicConfig(level=logging.WARNING)

import socket
import argparse
import pickle

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
import matplotlib.pyplot as plt
import hpbandster.visualization as hpvis
#from hpbandster.examples.commons import MyWorker

from Workers import allWorker as MyWorker
from Workers import getID

import numpy
import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

import openml

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from rerf.rerfClassifier import rerfClassifier

from slacker import Slacker

### Set up slacker for status updates
with open('neurodata-slackr.conf', 'r') as fp:
    slack_token = fp.readline().strip()
    slack_channel = fp.readline().strip()

slack = Slacker(slack_token)
host_name = socket.gethostname()




parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=int, help='Minimum budget used during the optimization.',    default=10)
parser.add_argument('--max_budget',   type=int, help='Maximum budget used during the optimization.',    default=500)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=2)
parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=1)
parser.add_argument('--n_jobs', type=int,   help='Number of threads given to training.', default = 1)
parser.add_argument('--openml_dataid', type=int, help='OpenML-DataID.', default = 15)
parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.', default = "en4")
parser.add_argument('--run_id', type=int,   help='run ID.', default=1)
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default="output")
#parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')

args=parser.parse_args()

# Every process has to lookup the hostname
host = hpns.nic_name_to_host(args.nic_name)

if host_name == "synaptomes1":
    prefix = "syn1"
else:
    prefix = "test_run"

outputName = f"{prefix}_openml_d_{str(args.openml_dataid)}"


# Step 1: Start a nameserver (see example_1)
# Start a nameserver:
# We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()



# Step 2: workers
# Most optimizers are so computationally inexpensive that we can affort to run a
# worker in parallel to it. Note that this one has to run in the background to
# not plock!
w = MyWorker(sleep_interval = 0.5, run_id=args.run_id, host=host,\
             nameserver=ns_host, nameserver_port=ns_port, n_jobs = args.n_jobs,\
             dataID = args.openml_dataid)
w.run(background=True)



#if True:
#    time.sleep(5)   # short artificial delay to make sure the nameserver is already running
#    w = MyWorker(sleep_interval = 0.5,run_id=args.run_id, host=host, dataID = args.openml_dataid)
#    w.load_nameserver_credentials(working_directory=args.shared_directory)
#    w.run(background=False)
#    exit(0)


# Run an optimizer
# We now have to specify the host, and the nameserver information
bohb = BOHB(  configspace = MyWorker.get_configspace(),
                      run_id = args.run_id,
                      host=host,
                      nameserver=ns_host,
                      nameserver_port=ns_port,
                      min_budget=args.min_budget, max_budget=args.max_budget
               )

res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)


d1 = res.get_pandas_dataframe()[0]
loss = res.get_pandas_dataframe()[1] 

d1['loss'] = loss

d1.to_csv(os.path.join(args.shared_directory, outputName + ".csv"))

# In a cluster environment, you usually want to store the results for later analysis.
# One option is to simply pickle the Result object
print("Saving:\n")
with open(os.path.join(args.shared_directory, f'bohb_openml_d_{args.openml_dataid}.pkl'), 'wb') as fh:
    pickle.dump(res, fh)



# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()


## Send slack message
slack.chat.post_message(slack_channel, f'`{host_name}`: Finished openml_d_{args.openml_dataid}')








