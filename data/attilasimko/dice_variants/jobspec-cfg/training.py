from comet_ml import Experiment
import argparse
import os
from keras import backend as K

# Set up argument parser for running code from terminal
parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--dataset", default="WMH", help="Select dataset. Options are 'acdc' and 'wmh'.")
parser.add_argument("--num_epochs", default=10, help="Number of epochs.")
parser.add_argument("--num_filters", default=24, help="Number of epochs.")
parser.add_argument("--dskip", default=0, help="Number of epochs.")
parser.add_argument("--optimizer", default="SGD", help="Optimizer to use during training.")
parser.add_argument("--batch_size", default=12, help="Batch size for training and validating.")
parser.add_argument("--learning_rate", default=5e-4, help="Learning rate for the optimizer used during training. (Adam, SGD, RMSprop)")
parser.add_argument("--loss", default="coin", help="Loss function to use during training.")
parser.add_argument("--skip_background", default="False", help="Skip the background class when computing the loss.")
parser.add_argument("--epsilon", default="1", help="Skip the background class when computing the loss.")
parser.add_argument("--alpha1", default="-", help="Alpha for coin loss.")
parser.add_argument("--beta1", default="-", help="Beta for coin loss.")
parser.add_argument("--alpha2", default="-", help="Alpha for coin loss.")
parser.add_argument("--beta2", default="-", help="Beta for coin loss.")
parser.add_argument("--alpha3", default="-", help="Alpha for coin loss.")
parser.add_argument("--beta3", default="-", help="Beta for coin loss.")
parser.add_argument("--alpha4", default="-", help="Alpha for coin loss.")
parser.add_argument("--beta4", default="-", help="Beta for coin loss.")
parser.add_argument("--base", default=None) # Name of my PC, used to differentiate between different paths.
parser.add_argument("--gpu", default=None) # If using gauss, you need to specify the GPU to use.
args = parser.parse_args()

if ((args.dataset == "ACDC") | (args.dataset == "WMH")):
    dataset = args.dataset
else:
    raise ValueError("Dataset not found.")
gpu = args.gpu
if args.base == "gauss":
    if gpu is not None: # GPUs are usually defined only when running from gauss, everywhere else there's only a single GPU.
        base = 'gauss'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        base_path = '/data/attila/' + str(dataset) + "_" + str(args.dskip) + '/'
        save_path = '/home/attila/out/'
    else:
        raise ValueError("You need to specify a GPU to use on gauss.")
elif args.base == "alvis":
    base = 'alvis'
    base_path = '/mimer/NOBACKUP/groups/naiss2023-6-64/' + str(dataset) + "_" + str(args.dskip) + '/'
    save_path = '/cephyr/users/attilas/Alvis/out/'
else:
    base_path = "/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/" + dataset + "_" + str(args.dskip) + '/'
    save_path = "/home/attilasimko/Documents/out/"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
from model import unet_2d
from data import DataGenerator
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import uuid
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import gc
from keras import backend as K
from utils import *

# Set seeds for reproducibility
set_seeds()
# All the comet_ml things are for online progress tracking, with this API key you get access to the MIQA project
experiment = Experiment(api_key="ro9UfCMFS2O73enclmXbXfJJj", project_name="segmentation-uncertainty")
batch_size = int(args.batch_size)
num_epochs = int(args.num_epochs)
if (dataset == "WMH"):
    labels = ["Background", "WMH", "Other"]
elif (dataset == "ACDC"):
    labels = ["Background", "LV", "RV", "Myo"]

# Custom data generator for efficiently loading the training data (stored as .npz files under base_path+"training/")
gen_train = DataGenerator(base_path + "train/",
                          dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True)

# Data generator for validation data
gen_val = DataGenerator(base_path + "val/",
                        dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False)

# Data generator for test data
gen_test = DataGenerator(base_path + "test/",
                         dataset=dataset,
                         batch_size=1,
                         shuffle=False)

# Allow gpu memory growth for tracking
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

# Log training parameters to the experiment
experiment.log_parameter("dataset", dataset) # The dataset used (MIQA or MIQAtoy)
experiment.log_parameter("save_path", save_path) # The loss function used
experiment.log_parameter("loss", args.loss) # The loss function used
experiment.log_parameter("skip_background", args.skip_background) # Whether to skip the background class when computing the loss
experiment.log_parameter("epsilon",  args.epsilon) # Smoothing for Dice and Coin loss
experiment.log_parameter("alpha1", 0.0 if (args.skip_background == "True") else args.alpha1) # Alpha for coin loss
experiment.log_parameter("beta1", 0.0 if (args.skip_background == "True") else args.beta1) # Beta for coin loss
experiment.log_parameter("alpha2", args.alpha2) # Alpha for coin loss
experiment.log_parameter("beta2", args.beta2) # Beta for coin loss
experiment.log_parameter("alpha3", args.alpha3) # Alpha for coin loss
experiment.log_parameter("beta3", args.beta3) # Beta for coin loss
experiment.log_parameter("alpha4", args.alpha4) # Alpha for coin loss
experiment.log_parameter("beta4", args.beta4) # Beta for coin loss
experiment.log_parameter("dskip", args.dskip) # Beta for coin loss
experiment.log_parameter("num_epochs", num_epochs) # The number of epochs
experiment.log_parameter("optimizer", args.optimizer)
experiment.log_parameter("learning_rate", float(args.learning_rate))
experiment.log_parameter("num_filters", int(args.num_filters))

# Set up the generators. This could have been done before, but this way the generator is the same as in another project, which is conventient.
gen_train.set_experiment(experiment)
gen_val.set_experiment(experiment)
gen_test.set_experiment(experiment)

if (dataset == "WMH"):
    if (experiment.get_parameter("alpha4") != "-"):
        raise ValueError("alpha4 is not used for WMH.")
    if (experiment.get_parameter("beta4") != "-"):
        raise ValueError("beta4 is not used for WMH.")
    
experiment_name = experiment.get_name()
if (experiment_name is None): #In some cases, comet_ml fails to provide a name for the experiment, in this case we generate a random UID
    experiment_name = str(uuid.uuid1())
print("Experiment started with name: " + str(experiment_name))

# Build model
model = unet_2d((256, 256, len(gen_train.inputs)), int(experiment.get_parameter("num_filters")), len(gen_train.outputs))
if (experiment.get_parameter('dataset') == "WMH"):
    alphas = [experiment.get_parameter('alpha1'),
                experiment.get_parameter('alpha2'),
                experiment.get_parameter('alpha3')]
    betas = [experiment.get_parameter('beta1'),
                experiment.get_parameter('beta2'),
                experiment.get_parameter('beta3')]
elif (experiment.get_parameter('dataset') == "ACDC"):
    alphas = [experiment.get_parameter('alpha1'),
                experiment.get_parameter('alpha2'),
                experiment.get_parameter('alpha3'),
                experiment.get_parameter('alpha4')]
    betas = [experiment.get_parameter('beta1'),
                experiment.get_parameter('beta2'),
                experiment.get_parameter('beta3'),
                experiment.get_parameter('beta4')]




print("Trainable model weights:")
print(int(np.sum([K.count_params(p) for p in model.trainable_weights])))

plot_idx = 0
x_val, y_val = gen_val.get_patient_data()
step = 0
learning_rate = float(experiment.get_parameter('learning_rate'))
patience_thr = 20
patience = 0
patience_dice = 0

grads_table = np.zeros((num_epochs, np.sum([x.shape[0] for x in x_val.values()]) * len(gen_val.outputs) * 2))
for epoch in range(num_epochs):
    compile(model, experiment.get_parameter('optimizer'), 
        learning_rate, 
        experiment.get_parameter('loss'), 
        experiment.get_parameter('skip_background') == "True",
        experiment.get_parameter('epsilon'), 
        alphas, betas)
    
    experiment.set_epoch(epoch)
    loss_total = []
    loss_a = []
    loss_b = []
    grads_min = []
    grads_max = []
    for _ in range(len(gen_train.outputs)):
        grads_min.append([])
        grads_max.append([])


    for i in range(int(len(gen_train))):
        x, y = gen_train.next_batch()
        plot_idx += 1

        inp = tf.Variable(x, dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            predictions = model(inp)
            loss_value = model.loss(tf.Variable(y, dtype=tf.float64), predictions)   
        gradients = tape.gradient(loss_value, predictions)

        for slc in range(gradients.shape[0]):
            for j in range(gradients.shape[-1]):
                if (np.min(gradients[slc, :, :, j]) < 0):
                    grads_min[j].append(np.min(gradients[slc, :, :, j]))
                if (np.max(gradients[slc, :, :, j]) > 0):
                    grads_max[j].append(np.max(gradients[slc, :, :, j]))

        loss_value = model.train_on_batch(x, y)
        loss_total.append(loss_value[0])


    gen_train.stop()
    experiment.log_metrics({'training_loss': np.mean(loss_total)}, epoch=epoch)
    for j in range(len(labels)):
        experiment.log_metrics({f'grad_min_{labels[j]}': np.sum(grads_min[j]),
                                f'grad_max_{labels[j]}': np.sum(grads_max[j])}, epoch=epoch)
    print(f"Training - Loss: {str(np.mean(loss_total))}")
    grads_table[epoch, :], _, _ = evaluate(experiment, (x_val, y_val), model, "val", labels, epoch)
    gen_val.stop()

    if (experiment.get_metric("val_avg_dice") > patience_dice):
        patience = 0
        patience_dice = experiment.get_metric("val_avg_dice")
    else:
        patience += 1
        if (patience > patience_thr):
            print("Patience limit reached, halving learning rate.")
            learning_rate /= 2
            patience = 0

    K.clear_session()
    gc.collect()

np.savetxt(save_path + "grads.csv", grads_table, delimiter=",")
experiment.log_table(save_path + "grads.csv")

x_test, y_test = gen_test.get_patient_data()
_ = evaluate(experiment, (x_test, y_test), model, "test", labels, epoch)
plot_results(gen_test, model, dataset, experiment, save_path)
experiment.end()