## 测试进度条
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path',   help='choose the path')
parser.add_argument('--seed',   help='choose the random seed',         type=int,   default=2333)
parser.add_argument('--dz',     help='choose the dz: [km]',            type=float, default=1.0)
parser.add_argument('--batch',  help='choose the batch size.',         type=int,   default=10)
parser.add_argument('--Nch',    help='choose number of channels',      type=int,   nargs='+', default=[19])
parser.add_argument('--SpS',    help='choose samples per symbol',      type=int,   default=32)
parser.add_argument('--power',  help='choose power per channel,[dBm]', type=float, nargs='+', default=[0])
parser.add_argument('--Rs',     help='choose symbol rate [Hz]',        type=float,   default=190e9)
parser.add_argument('--freqspace',     help='choose freq space [Hz]',  type=float,   default=220e9)

parser.add_argument('--Nbits',  help='choose number of bits.',         type=int,   default=400000)
parser.add_argument('--precision',  help='choose precision.',          type=str,   default='float64')
parser.add_argument('--Nmodes', help='choose the mode: tx, rx',        type=int,   default=1)

args = parser.parse_args()

if args.precision == 'float64':
    from jax.config import config
    config.update("jax_enable_x64", True)     # float64 precision
    
from copyreg import pickle
from optical_flax.fiber_system import Tx_data, channel, Rx_data
from optical_flax.fiber_channel import ssfm, manakov_ssf
import jax
import jax.random as rd
import os
import pickle

batch = args.batch
dz = args.dz
Nbits = args.Nbits  # number of bits   
N = args.Nch        # number of channels
SpS = args.SpS        # samples per symbol
Power = args.power
if args.Nmodes == 1:
    module = ssfm
else:
    module = manakov_ssf


k1,k2,k3 = rd.split(rd.PRNGKey(args.seed), 3)
data_path = args.path
key_tx = rd.split(k1, len(N))
key_ch = rd.split(k2, len(N))
key_rx = rd.split(k3, len(N))




if not os.path.exists(data_path):
    os.mkdir(data_path)


for i in range(len(N)):
    n = N[i]
    for j in range(len(Power)):
        power = Power[j]
        print(f'     channel {n}  Power {power}      ')

        ## Step 1: Tx
        tx_path= data_path + f'/Tx_ch{n}_power{power}'
        if os.path.exists(tx_path):
            print('Tx data has been generated before !')
            with open(tx_path, 'rb') as file:
                sigWDM, symbWDM, param = pickle.load(file)
        else:
            sigWDM, symbWDM, param = Tx_data(key_tx[i], batch, Nch=N[i], Power=power, SpS=SpS, Nbits=Nbits, Nmodes=args.Nmodes, Rs=args.Rs, freq_space=args.freqspace)
            with open(tx_path, 'wb') as file:
                pickle.dump((sigWDM, symbWDM, param), file)


        ## Step 2: channel
        channel_path = data_path + f'/Channel_ch{n}_power{power}_dz{dz}'
        if os.path.exists(channel_path):
            print('channel data has been generated before !')
            with open(channel_path, 'rb') as file:
                sigWDM_rx, paramCh= pickle.load(file)
        else:
            Fs = param.Rs*param.SpS  # sample rates
            sigWDM_rx, paramCh = channel(key_ch[i], sigWDM, Fs, dz=dz, module=module)
            with open(channel_path, 'wb') as file:
                pickle.dump((sigWDM_rx, paramCh), file)
        # jax.profiler.save_device_memory_profile(f"ch.prof")
        

        # rx_sps = 8
        # FO = 0    # 64e6
        # lw = 0    # 100e3

        # ## Step 3:
        # if args.phase == 'rx':
        #     rx_path = data_path + f'/dataset_sps{rx_sps}/data_ch{n}_power{power}_FO{FO}_lw{lw}'
        #     if os.path.exists(rx_path):
        #         print('Rx data has been generated before !')
        #     else:
        #         data_sml, paramRx, noise = Rx_data(key_rx[i], sigWDM_rx, symbWDM, rx_sps, param=param, paramCh=paramCh, FO=FO, lw=lw)
        #         with open(rx_path, 'wb') as file:
        #             pickle.dump((data_sml, paramRx, noise), file)
