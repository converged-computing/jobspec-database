import pickle, re, os, time
import jax, optax, jax.random as rd, jax.numpy as jnp, flax.linen as nn
from functools import partial
from DataLoader import DataLoader,take_data,Generation_DataSet
from optical_flax.fiber_system import get_data
from optical_flax.layers import MetaSSFM, GRU_DBP, fdbp, NNSSFM,Dense_net
from optical_flax.models import Transformer, CNN, xi_config,H_config
from optical_flax.initializers import fdbp_init, near_zeros,ones
from optical_flax.utils import realize,show_tree
from commplax.module.core import Signal
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', help='choose the meta method of ["cnn","dense","nn"]', default='nn')
parser.add_argument('--step', help='choose the DBP steps', type=int, default=3)
parser.add_argument('--epochs', help='choose train epochs.', type=int, default=40)
parser.add_argument('--save_path', help='save path.', default='loading/')
args = parser.parse_args()

######################## STEP (0) Loading data ########################
# SETTINGS: Construct dataset.
Nlen = 10000
Nstep = 800
sps = 2
ch_key  = 'ch7'

data_file = '/home/xiaoxinyu/data/'
train_path =  data_file + '0628train/dataset_sps8/'
test_path =   data_file + '0628test/dataset_sps8/'
config_path = data_file + '0628test/dataset_sps8/data_ch7_power0_FO0_lw0'

train_path_list = [train_path + f for f in os.listdir(train_path) if ch_key in f]
test_path_list = [test_path + f for f in os.listdir(test_path) if ch_key in f]
x_data,y_data = Generation_DataSet(train_path_list, sps, Nlen=Nlen, Nstep=Nstep)
data_sml = get_data(config_path, sps)
y_example = jnp.ones([3,Nlen*sps, 2]) + 1j


######################## STEP (1) setting parameters ########################

# SETTINGS: Training parameters.
k = 2   # additional mimo symbols
assert (Nlen-Nstep)//2-k >= 0
steps = args.step
save_param = False
Epochs = args.epochs
batch_size = 100

d_init, n_init = fdbp_init(data_sml.a, xi=1.1, steps=steps, domain='frequency')
dtaps = Nlen*sps
ntaps = Nlen*sps
init_rng = rd.PRNGKey(1213)
dropout_rng = rd.PRNGKey(233)

######################## STEP (2) Define model. ########################
# SETTINGS: Construct Model.
method = args.method
nn_H = partial(Dense_net, nn_mode=True)
trans_H = partial(Transformer, config=H_config, nn_mode=False)
cnn_H = partial(CNN, nn_mode=False)
dense_H = partial(Dense_net, width=(5,5), nn_mode=False)

nn_xi = partial(Dense_net, nn_mode=True)
trans_xi = partial(Transformer, config=xi_config, nn_mode=False)
cnn_xi = partial(CNN,dtype=jnp.float32, param_dtype=jnp.float32, nn_mode=False)
dense_xi = partial(Dense_net, width=(20,20,20),dtype=jnp.float32, param_dtype=jnp.float32,nn_mode=False)

if method == 'cnn':
    Meta_H,Meta_xi = nn_H, cnn_xi
elif method == 'dense':
    Meta_H, Meta_xi = nn_H, dense_xi
elif method == 'nn':
    Meta_H,Meta_xi = nn_H,nn_xi
else:
    raise(ValueError)



Net = partial(MetaSSFM,
            steps=steps, 
            d_init=d_init, 
            n_init=n_init, 
            dtaps=dtaps, 
            ntaps=ntaps, 
            discard=sps*((Nlen-Nstep)//2-k),
            Meta_H=Meta_H,     # TODO: change the meta net
            Meta_xi=Meta_xi)   # TODO: change the meta net
#Net = partial(NNSSFM,steps=steps, d_init=d_init, n_init=n_init, dtaps=dtaps, ntaps=ntaps, discard=sps*((Nlen-Nstep)//2-k))
Net_vmap = nn.vmap(Net, variable_axes={'params':None}, split_rngs={'params':False,'dropout':False}, in_axes=(0,None),out_axes=(0))

class MyDBP(nn.Module):

    @nn.compact
    def __call__(self, signal, train):
        x, t = Net_vmap()(signal,train)
        # x [batch, N, 2]
        x = nn.Conv(features=2,kernel_size=((2*k+1)*sps,),strides=(sps,), param_dtype=jnp.complex64,dtype=jnp.complex64, padding='valid')(x)
        return Signal(x,t)
    
net = MyDBP()
LDBP = realize(net)
var0 = LDBP.init(init_rng, Signal(y_example), False)
print(show_tree(var0))

######################## STEP (3) Define optimizer. ########################

# SETTINGS: Construct optimizer.
schedule = optax.warmup_cosine_decay_schedule(
  init_value=0.0,
  peak_value=0.1,
  warmup_steps=10,
  decay_steps=100,
  end_value=0.001,
)

tx = optax.chain(
  optax.clip(1.0),
  optax.adamw(learning_rate=schedule,weight_decay=0.0001),
)
# tx = optax.adam(learning_rate=1e-3)

######################## STEP (4) define loss and update step.########################

def train_loss(var, xi, yi, key): 
    x = LDBP.apply(var, Signal(yi), True, rngs={'dropout':key})
    s = (Nlen - Nstep)//2
    e = s + Nstep
    return jnp.mean(jnp.abs(x.val - xi[:,s:e,:])**2)


def test_loss(var, xi, yi): 
    x = LDBP.apply(var, Signal(yi), False)
    s = (Nlen - Nstep)//2
    e = s + Nstep
    return jnp.mean(jnp.abs(x.val - xi[:,s:e,:])**2)


@jax.jit
def update_param(var, opt_state, xi, yi, key):
    loss_val, grads = jax.value_and_grad(train_loss)(var, xi,yi, key)
    updates, opt_state = tx.update(grads, opt_state, var)
    var = optax.apply_updates(var, updates)
    return var, opt_state, loss_val


######################## STEP (5) Training model. ########################

loss_dict = {}
loss_dict['train'] = []
for test_path in test_path_list:
    loss_dict[test_path] = []
opt_state0 = tx.init(var0)



for t in range(Epochs):
    
    DL = DataLoader(x_data,y_data,batch_size,jax.random.PRNGKey(0))
    for i,(xi,yi) in enumerate(DL):
        key = rd.fold_in(dropout_rng, t)
        var0, opt_state0, l = update_param(var0, opt_state0, xi, yi, key)
        loss_dict['train'].append(l)
        if i % 10 == 0:
            print(f'Epoch {t} -- batch {i} -- train loss {l}')
    
    for test_path in test_path_list:
        # FIXME: data OOM.
        yi,xi = take_data(test_path, sps, Nlen, Nlen)
        l = test_loss(var0, xi, yi)
        loss_dict[test_path].append(l)

        j = re.search('power',test_path).start(0)
        print(f'########### Epoch {t} -- Test loss on {test_path[j:j+8]}: {l}  ##############')
    if save_param:
        with open(f'loading/param_NN/state_epoch{t}', 'wb') as file:
            pickle.dump({'param':var0, 'loss_val':l}, file)
    


with open(args.save_path,'wb') as file:
    pickle.dump(loss_dict, file)