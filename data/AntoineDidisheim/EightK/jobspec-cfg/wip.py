import didipack as didi
import pandas as pd


import pandas as pd
import datetime
from matplotlib import pyplot as plt
from data import Data
from parameters import *
import seaborn as sns
from utils_local.plot import plot_ev
from utils_local.general import table_to_latex_complying_with_attila_totally_unreasonable_demands
import pandas as pd
import pytz
import pytz
import pytz

if __name__ == '__main__':
    args = didi.parse()
    par = Params()
    data = Data(par)
    save_dir = Constant.EMB_PAPER

    year = np.arange(1996, 2023)[args.a]  # len 27
    to_load = f'ref{year}.p'
    load_dir = data.p_to_vec_main_dir + '/single_stock_news_to_vec/'
    df = pd.read_pickle(load_dir + to_load)

    df = pd.read_pickle(data.p_eight_k_clean + f'press_{year}.p')
