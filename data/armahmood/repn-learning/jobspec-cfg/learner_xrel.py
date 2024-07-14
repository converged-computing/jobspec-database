import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
import argparse
import os
import sys
import pickle
import math
import json
from tqdm import tqdm
from adam_gnt import adam_gnt

"""
Three sources of randomness.
1. Target network
2. Learning network
3. Data generation
"""

#####UTILITY FUNCTIONS

def update_config(path):
    with open(path, 'r') as json_file: 
      data = json.load(json_file)
    return data

def store_losses(losses, features, seed_num, search=False, stname =''):
  if search:
    path = stname + '/search/' + str(features) + '/'
  else:
    path = stname + '/fixed/' + str(features) + '/'
  try:
    assert os.path.exists(path)
  except:
    os.makedirs(path)
  fname = path + 'run_' + str(seed_num)
  dbfile = open(fname, 'ab')
  pickle.dump(losses, dbfile)
  dbfile.close()

#######

def calculate_threshold(weights):
  """Calculates LTU threshold according to weights"""
  m = weights.shape[1]
  beta = 0.6
  S = torch.sum(weights<0, dim=1)
  threshold = torch.ones(weights.shape[0])*m*beta - S.float()
  return threshold

def ltu(input, weights):
  """LTU logic"""
  tau = calculate_threshold(weights)
  input = input - tau
  input[input>=0] = 1.0
  input[input<0] =  0.0
  return input

class LTU(nn.Module):
  """LTU activation function"""
  def __init__(self, n_inp, n_tl1):
    super().__init__()
    self.weight = Parameter(torch.Tensor(n_tl1, n_inp))
  
  def forward(self, input):
    input = ltu(input, self.weight)
    return input

def update_lr(optimizer,lr):
    """Scheduler to update learning rate at every iteration"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

activations = {"LTU":LTU, "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh, "Relu": nn.ReLU, "leaky": nn.LeakyReLU}


def calbound_kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    gain = torch.nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return bound

def initialize_target_net(n_inp, n_tl1, tgen, seed_num_target, config):
  """Initializes target network"""
  act = config["activation_target"]
  if act=="LTU":
    activation_function = activations[act](n_inp, n_tl1)
  else:
    activation_function = activations[act]()
  tnet = nn.Sequential(nn.Linear(n_inp, n_tl1, bias=False), activation_function, nn.Linear(n_tl1, 1))
  with torch.no_grad():
    #Input weights initialized with +1/-1
    tgen.manual_seed(seed_num_target)
    tnet[0].weight.data = (torch.randint(0, 2, tnet[0].weight.data.shape, generator=tgen)*2-1).float()  ### 1
    if tnet[0].bias is not None:
      tnet[0].bias.data = torch.randn(tnet[0].bias.data.shape, generator=tgen)
    #Output layer weights initialized with N(0,1)
    tnet[2].weight.data = torch.randn(tnet[2].weight.data.shape, generator=tgen)  ### 1
    tnet[2].bias.data = torch.randn(tnet[2].bias.data.shape, generator=tgen)  ### 1
    if act=="LTU":
      tnet[1].weight = tnet[0].weight
  return tnet

def initialize_learning_net(n_inp, n_l1, lgen, seed_num, config):
  """Initializes learning network"""
  act = config["activation_learning"]
  if act=="LTU":
    activation_function_ = activations[act](n_inp, n_l1)
  else:
    activation_function_ = activations[act]()
  net = nn.Sequential(nn.Linear(n_inp, n_l1, bias=False), activation_function_, nn.Linear(n_l1, 1))
  with torch.no_grad():
    lgen.manual_seed(seed_num)
    bound = calbound_kaiming_uniform_(net[0].weight, a=math.sqrt(5))
    net[0].weight.uniform_(-bound, bound, generator=lgen)### 2
    #net[0].weight.data = (torch.randint(0, 2, net[0].weight.data.shape, generator=lgen)*2-1).float()  ### 2
    if net[0].bias is not None:
      net[0].bias.data = torch.randn(net[0].bias.data.shape, generator=lgen)
    if act=="LTU":
      net[1].weight = net[0].weight
    torch.nn.init.zeros_(net[2].weight)
    torch.nn.init.zeros_(net[2].bias)
  return net, bound

def run_experiment(n_inp, n_tl1, T, n_l1, seed_num, target_seed, config, search=False):
  tgen = torch.Generator()
  tnet = initialize_target_net(n_inp, n_tl1, tgen, target_seed, config)
  lossfunc = nn.MSELoss()
  lgen = torch.Generator()
  net, bound = initialize_learning_net(n_inp, n_l1, lgen, seed_num, config)
  step_size = config['step']
  adam = adam_gnt.AdamGNT(net[2:].parameters(), lr = step_size)
  dgen = torch.Generator().manual_seed(seed_num + 2000)
  lgen.manual_seed(seed_num + 3000)
  losses = []
    
  if search:
    util = torch.zeros(n_l1)
    tester_lr = config["tester_lr"] #0.01
    rr = config["replacement_rate"] #1/200# Replacement rate per time step per feature
    n_el = 0  # rr*n_l1  # Number of features eligible for replacement

  for _ in tqdm(range(T)):
    inp = torch.randint(0, 2, (n_inp,), generator=dgen, dtype=torch.float32)  ### 3
    target = tnet(inp) + torch.randn(1, generator=dgen)  ### 3
    neck = net[:2](inp)
    pred = net[2:](neck)
    loss = lossfunc(pred, target)
    losses.append(loss.item())
    net.zero_grad()
    loss.backward()
    adam.step()         
    if search:
      n_el += rr*n_l1
      with torch.no_grad():
        if config["tester"]==1:
          util_target = torch.abs(net[2].weight.data[0])
        if config["tester"]==2:
          wx = net[2].weight.data[0]*neck
          util_target = torch.abs(wx)

        util += tester_lr*(util_target - util)
        if n_el >= 1:
          weak_node_i = torch.topk(util, int(n_el), largest=False)[1]
          subs = torch.zeros(net[0].weight[weak_node_i].shape)
          net[0].weight[weak_node_i] = subs.uniform_(-bound, bound, generator=lgen) ### 2
          if net[0].bias is not None:
            net[0].bias[weak_node_i] = torch.randn(1, generator=lgen)
          net[2].weight[0][weak_node_i] = 0.0
          adam.state[net[2].weight]['exp_avg'][0][weak_node_i] = 0.0
          adam.state[net[2].weight]['exp_avg_sq'][0][weak_node_i] = 0.0
          adam.state[net[2].weight]['step'][0][weak_node_i] = 0.0
          util[weak_node_i] = torch.median(util)
          n_el = 0

  losses = np.array(losses)
  return losses


def main():
  parser = argparse.ArgumentParser(description="Generate and Test")
  parser.add_argument("--cfg", type=str, default='config.json',
                      help="config file name")
  parser.add_argument("-se", "--search", action='store_true',
                      help="run experiment with search")
  parser.add_argument("-o", "--save", action='store_true',
                      help="Saves the output graph")
  parser.add_argument("--save_losses", action='store_true',
                      help="Saves losses for individual runs(NOT TO BE USED WITHOUT BASH SCRIPT)")
  args = parser.parse_args()
  fname = args.cfg
  config = update_config(fname)
  T = config["examples"]
  n = config["runs"]
  nbin = 10000
  n_inp = config["input_size"]
  n_tl1 = 20
  n_feature = config["features"]
  n_seed = config["learner_seeds"]
  t_seed = config["target_seed"]

  #stname = "sigmoid001"
  #stname = "experiment_leaky/leakypaper"+"_rr" + "{:.6f}".format(config["replacement_rate"]) + "_step"+ "{:.5f}".format(config["step"])
  stname = "experiment_relu/ReLUpaper"+ "_rr" + "{:.6f}".format(config["replacement_rate"]) + "_dr" + str(config["tester_lr"])
  try:
    path = "output/out_" + str(t_seed)+".png"
    assert not os.path.exists(path)
  except:
    print("Experiment results already exist")
    sys.exit(1)

  try:
    assert t_seed not in n_seed
  except:
    print("Error: t_seed has to be different than n_seed")
    sys.exit(1)

  if (len(n_seed)!=n):
    print("Insuffcient number of seeds")
    return

  for nl_1 in n_feature:
    net_loss = 0
    print("No of Features:", nl_1)
    for l in range(n):
      print("Run:", l+1)
      if args.search:
        net_loss = net_loss + run_experiment(n_inp, n_tl1, T, nl_1, n_seed[l], t_seed, config, search=True)
        if args.save_losses:
          store_losses(net_loss, n_feature[0], n_seed[0], search=True, stname=stname)
          sys.exit(1)
      else:
        net_loss = net_loss + run_experiment(n_inp, n_tl1, T, nl_1, n_seed[l], t_seed, config)
        if args.save_losses:
          store_losses(net_loss, n_feature[0], n_seed[0], stname=stname)
          sys.exit(1)
    net_loss = net_loss/n
    bin_losses = net_loss.reshape(T//nbin, nbin).mean(1)
    plt.plot(range(0, T, nbin), bin_losses, label=nl_1)
  tgen = torch.Generator()
  tnet = initialize_target_net(n_inp, n_tl1, tgen, t_seed, config)
  norm_out = tnet[-1].weight.norm().data
  norm_out = format(float(norm_out), '.4f')
  title = "Output weight norm of t-net: " + str(norm_out) 
  plt.suptitle(title, fontsize=13)
  axes = plt.axes()
  axes.set_ylim([1.0, 3.5])
  plt.legend()

  if args.save:
    try:
      assert os.path.exists("output/")
    except:
      os.makedirs('output/') 
    filename = "output/out_" + str(t_seed)
    plt.savefig(filename)
  else:
    plt.show()


if __name__ == "__main__":
    main()
