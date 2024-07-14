# in use imports

import utils.model_utils as model_utils
import utils.train_utils as train_utils
import utils.data_utils as data_utils

import jax
from jax import numpy as jnp
import optax
from flax import linen as nn

from typing import Tuple

#usual imports
import pandas as pd
import argparse
import pickle as pl

"""### Model definition and train state definition"""

def linear_warmup(step, c_init, c_trgt, sharpness_init, rate):
    "lr = lr_max * step / step_max"
    lr_init = c_init / sharpness_init
    lr_trgt = c_trgt / sharpness_init
    assert(lr_trgt >= lr_init), f'lr_max: {lr_trgt}, lr_min: {lr_init}'
    lr = lr_init + (lr_trgt - lr_init) * rate * (step)
    return min(lr_trgt, lr)

def create_train_state(config: argparse.ArgumentParser, batch: Tuple):
    x, y = batch

    x = x[:config.batch_size, ...]
    y = y[:config.batch_size, ...]

    # create model
    model = models[config.model](num_filters = config.width, widening_factor = config.widening_factor, num_classes = config.out_dim, act = config.act, varw = config.varw, scale = config.scale)
    # initialize using the init seed
    key = jax.random.PRNGKey(config.init_seed)
    init_params = model.init(key, x)['params']
    #debugging: check shapes and norms
    shapes = jax.tree_util.tree_map(lambda x: x.shape, init_params)
    #print(shapes)
    norms = jax.tree_util.tree_map(lambda x: config.width * jnp.var(x), init_params)
    #print(norms)
    # create an optimizer
    opt = optax.inject_hyperparams(optax.sgd)(learning_rate = 0.1, momentum = config.momentum)
    # create a train state
    state = train_utils.TrainState.create(apply_fn = model.apply, params = init_params, opt = opt)

    return state


def train_and_evaluate(config: argparse.ArgumentParser, train_ds: Tuple, test_ds: Tuple):
    "train model acording the config"
    
    # create a train state
    state = create_train_state(config, train_ds)
    
    state_fn = state.apply_fn

    # create train and test batches for measurements: measure batches are called train_batches and val_batches; training batches are called batches
    seed = config.sgd_seed
    rng = jax.random.PRNGKey(seed)
    train_batches = train_utils.data_stream(seed, train_ds, config.batch_size)
    test_batches = train_utils.data_stream(seed, test_ds, config.batch_size)
    
    
    # update learning rate
    state.update_learning_rate(learning_rate = config.lr)

    step_results = list()
    divergence = False
    
    ######### TRAINING PHASE ##############
    import time 
    # Start timing
    start_time = time.time()
    
    batches = train_utils.data_stream(seed, train_ds, config.batch_size, augment = config.use_augment)
    num_batches = config.num_batches # make it a non config variable
    
    # prepare an initial guess for the eigenvectors of the hessian
    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)

    key = jax.random.PRNGKey(24)
    vs_step = jax.random.normal(key, shape=flat_params.shape)

    for epoch in range(config.num_epochs):     
        if divergence: break
        
        train_loss = 0
        train_accuracy = 0

        for batch_ix in range(config.num_batches):

            #get the next batch and calculate the step
            batch = next(batches)
            imgs, targets = batch
            step = config.num_batches*epoch + batch_ix
            
            #state, logits_step, loss_step, sharpness_step, vs_step = train_utils.train_sharpness_power_step(state, batch, config.loss_fn, vs_step)
            #accuracy_step = train_utils.compute_accuracy(logits_step, targets)
            #print(f't: {step}, loss: {loss_step:0.6f}, sharpness: {sharpness_step:0.6f}, accuracy: {accuracy_step:0.6f}')
            
            # measure sharpness
            #sharpness_step, _ = train_utils.hessian_power_step(state, batch, config.loss_fn, vs_step)

            #train for one step
            state, logits_step, loss_step = train_utils.train_step(state, batch, config.loss_fn)
            # estimate accuracy from logits
            accuracy_step = train_utils.compute_accuracy(logits_step, targets)
            
            #print(f't: {step}, loss: {loss_step:0.6f}, sharpness: {sharpness_step:0.6f}, accuracy: {accuracy_step:0.6f}')
            
            # append them to the running metrics
            train_loss += loss_step
            train_accuracy += accuracy_step
            #print(f't: {step}, Loss: {loss_step:0.4f}, Accuracy: {accuracy_step:0.4f}')
            #check for divergence
            if (jnp.isnan(loss_step) or jnp.isinf(loss_step)): divergence = True; num_batches = batch_ix+1; break
        # Estimate running loss and accuracy at each epoch
        
        train_loss /= num_batches
        train_accuracy /= num_batches

        # Estimate test loss and accuracy at each epoch
        print(f'E: {epoch}')
        #test_loss, test_accuracy = train_utils.compute_metrics(state_fn, state.params, config.loss_fn, test_batches, config.num_test, config.batch_size)
        #print(f'Epoch: {epoch}, Train loss: {train_loss:0.4f}, Train accuracy: {train_accuracy:0.4f}, Test loss: {test_loss:0.4f}, Test accuracy: {test_accuracy:0.4f}')

    
    # end time
    end_time = time.time()
    
    # Calculate elapsed time in seconds
    elapsed_time = end_time - start_time

    # Convert to minutes and seconds
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60

    # Print out the time
    print(f"Time taken: {int(minutes)} minutes and {seconds:.2f} seconds")
    
    return divergence


# Add more models later on
#models = {'fcn_mup': model_utils.fcn_int, 'fcn_sp': model_utils.fcn_sp, 'myrtle_sp': model_utils.Myrtle, 'myrtle_mup': model_utils.Myrtle_int}
models = {'WideResNet16': model_utils.WideResNet16, 'WideResNet20': model_utils.WideResNet20, 'WideResNet28': model_utils.WideResNet28, 'WideResNet40': model_utils.WideResNet40}
loss_fns = {'mse': train_utils.mse_loss, 'xent': train_utils.cross_entropy_loss}
activations = {'relu': nn.relu, 'tanh': jnp.tanh, 'linear': lambda x: x}

parser = argparse.ArgumentParser(description = 'Experiment parameters')
# Dataset parameters
parser.add_argument('--dataset', type = str, default = 'cifar10')
parser.add_argument('--out_dim', type = int, default = 10)
parser.add_argument('--num_examples', type = int, default = 50000)
# Model parameters
parser.add_argument('--abc', type = str, default = 'sp')
parser.add_argument('--width', type = int, default = 16)
parser.add_argument('--widening_factor', type = int, default = 1)
parser.add_argument('--depth', type = int, default = 16)
parser.add_argument('--varw', type = float, default = 2.0)
parser.add_argument('--scale', type = float, default = 1.0)
parser.add_argument('--bias', type = str, default = 'True') # careful about the usage
parser.add_argument('--act_name', type = str, default = 'relu')
parser.add_argument('--init_seed', type = int, default = 1)
#Optimization parameters
parser.add_argument('--loss_name', type = str, default = 'xent')
parser.add_argument('--augment', type = str, default = 'False')
parser.add_argument('--opt', type = str, default = 'sgd')
parser.add_argument('--sgd_seed', type = int, default = 1)
parser.add_argument('--warm_steps', type = int, default = 128)
parser.add_argument('--num_epochs', type = int, default = 100)
parser.add_argument('--c_init', type = float, default = 0.0)
parser.add_argument('--lr_exp_start', type = float, default = 0.0)
parser.add_argument('--lr_step', type = float, default = 0.5)
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--momentum', type = float, default = 0.0)
parser.add_argument('--batch_size', type = int, default = 128)
# Sharpness estimation
parser.add_argument('--topk', type = int, default = 1)
parser.add_argument('--test', type = str, default = 'True')
parser.add_argument('--measure_batches', type = int, default = 10)


config = parser.parse_args()

# Model parameters
config.model = f'WideResNet{config.depth}'
config.use_bias = True if config.bias == 'True' else False
config.use_augment = True if config.augment == 'True' else False
config.act = activations[config.act_name]

config.loss_fn = loss_fns[config.loss_name]
# Optimization parameters
config.schedule_name = 'constant'
config.rate = 1.0 / config.warm_steps # This rate correspond to 1/T_warm and not lr_trgt / T_warm
config.measure_batches = int(4096.0 / config.batch_size)

save_dir = 'resnet_results'
data_dir = '/home/dayal'

(x_train, y_train), (x_test, y_test) = data_utils.load_image_data(data_dir, config.dataset, flatten = False, num_examples = config.num_examples)

config.num_train, config.num_test = x_train.shape[0], x_test.shape[0]

# #standardize the inputs
x_train = data_utils._standardize(x_train, abc = config.abc)
x_test = data_utils._standardize(x_test, abc = config.abc)

config.in_dim = int(jnp.prod(jnp.array(x_train.shape[1:])))

#get one hot encoding for the labels
y_train = data_utils._one_hot(y_train, config.out_dim)
y_test = data_utils._one_hot(y_test, config.out_dim)

config.num_batches = train_utils.estimate_num_batches(config.num_train, config.batch_size)
divergence = False

config.lr_exp = config.lr_exp_start

# train the model
divergence = train_and_evaluate(config, (x_train, y_train), (x_test, y_test))
