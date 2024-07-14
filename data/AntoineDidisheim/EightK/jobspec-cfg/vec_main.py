import os

import pandas as pd
import pyperclip
import tqdm

from parameters import *
from data import Data
import didipack as didi
from matplotlib import pyplot as plt
import seaborn as sns
from didipack import PandasPlus, PlotPlus
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
from utils_local.zip import decompress_gz_file
import json
import glob
import itertools
import re
from utils_local.llm import EncodingModel
from utils_local.zip import decompress_gz_file, unzip_all
from utils_local.vec_functions import vectorise_in_batch


def load_and_process_eight_k_legal_or_pressed(par, args, press_or_legal='legal'):
    save_size = 1000
    batch_size = 2
    # load and pre process data (code specific)
    data = Data(par)
    # those are the events for which we have a match in crsp
    ev = data.load_list_by_date_time_permno_type()
    ev = ev[['cik', 'form_id']].drop_duplicates()
    year = np.arange(2004, 2023)[args.a]
    # load the raw text of the year
    df = pd.read_pickle(data.p_eight_k_clean + f'{press_or_legal}_{year}.p')
    # make the dtypes match for the merge
    df['cik'] = df['cik'].astype(int)
    sh = df.shape[0]
    # keep only the ones with a match
    df = df.merge(ev, how='inner')
    print('Droped', df.shape[0] / sh)
    print('Left', df.shape[0])
    if press_or_legal == 'legal':
        df['item'] = pd.to_numeric(df['item'], errors='coerce')
        ind = df['item'].isin(Constant.LIST_ITEMS_TO_USE)
        df = df.loc[ind, :]
        id_col = ['cik', 'form_id', 'item']
        # remove the text that have been extracted from multiple sources
        df['len'] = df['txt'].apply(lambda x: len(str(x)))
        ind = df.groupby(id_col)['len'].transform('max') == df['len']
        df = df.loc[ind, :].reset_index(drop=True)
        # remove the duplicates that have same lengths of text arbitrairly
        while df[id_col].duplicated().sum() != 0:
            ind = ~df[id_col].duplicated()
            df = df.loc[ind, :].reset_index(drop=True)
    else:
        id_col = ['cik', 'form_id', 'k']
    df = df.drop_duplicates()
    return id_col, save_size, batch_size, year, df


def load_and_process_wsj(par, args):
    save_size = 1000
    batch_size = 2
    year = np.arange(1996, 2023)[args.a]
    # load and pre process data (code specific)
    data = Data(par)
    if par.enc.news_source == NewsSource.WSJ_ONE_PER_STOCK:
        df = data.load_wsj_one_per_tickers().reset_index(drop=True)
        df = df.drop_duplicates()
        df['ids'] = df.index
        ind = df['date'].dt.year == year
        df = df.loc[ind, :]
        id_col = ['ids', 'date', 'ticker']
        df['txt'] = df['headline'] + ' \n \n ' + df['body']
        df = df.drop(columns=['body', 'headline'])
    return id_col, save_size, batch_size, year, df


def drop_already_process_text_from_df(df, par):
    save_dir = par.get_vec_process_dir()
    id_processed = []
    for f in tqdm.tqdm([x for x in os.listdir(save_dir) if str(year) in x], 'remove processed event'):
        id_processed.append(int(f.split('.p')[-2].split('_')[1]))
        temp = pd.read_pickle(save_dir + f)
        ind = ~df.set_index(id_col).index.isin(temp.index)
        df = df.loc[ind, :]
    if len(id_processed) > 0:
        max_id_processed = max(id_processed)
    else:
        max_id_processed = 0
    return df, max_id_processed


def load_and_process_news_one_stock_ref_or_third(par, args, ref_or_thrid_party='ref'):
    if ref_or_thrid_party =='ref':
        batch_size = 1
    else:
        batch_size = 1

    if par.enc.opt_model_type == OptModelType.BOW1:
        save_size = 10000
    else:
        save_size = 5000
    # load and pre process data (code specific)
    data = Data(par)
    load_dir = data.p_to_vec_main_dir + '/single_stock_news_to_vec/'
    # Keep this to check the lsit to run in debugging
    # list_todo = np.sort([x for x in os.listdir(load_dir) if ref_or_thrid_party in x])
    year = np.arange(1996, 2023)[args.a]  # len 27
    to_load = f'{ref_or_thrid_party}{year}.p'
    df = pd.read_pickle(load_dir + to_load)
    df = df.reset_index(drop=True).reset_index()
    df['txt'] = df['headline']
    ind = df['alert'] == False
    df.loc[ind, 'txt'] = df.loc[ind, 'txt'] + ' \n ' + df.loc[ind, 'body']
    id_col = ['id', 'index', 'timestamp', 'alert', 'ticker']
    df = df[id_col + ['txt']]
    print('Loaded year', year)
    print('With ', ref_or_thrid_party)
    print('Size', df.shape)
    print('Duplicated id', df[id_col].duplicated().sum())
    return id_col, save_size, batch_size, year, df


if __name__ == "__main__":
    args = didi.parse()  # --legal=0/1 --eight=0/1 |  19 variations for legal eight
    # BUILD THE MODEL AND DEFINE PARAMETERS
    par = Params()
    par.enc.framework = Framework.TENSORFLOW
    if socket.gethostname() == '3330L-214940-M':
        # (local debug)
        par.enc.opt_model_type = OptModelType.BOW1
        par.enc.news_source = NewsSource.EIGHT_LEGAL
        id_col, save_size, batch_size, year, df = load_and_process_eight_k_legal_or_pressed(par, args)
        save_size = 10
        # launch the vectorisation
        vectorise_in_batch(id_col=id_col, df=df, save_size=save_size, batch_size=batch_size, par=par, year=year)
    else:
        par.enc.opt_model_type = OptModelType.OPT_13b
        # par.enc.opt_model_type = OptModelType.OPT_6b7
        if args.small == 1:
            print('encodign small')
            par.enc.opt_model_type = OptModelType.OPT_125m
    if args.bow == 1:
        par.enc.opt_model_type = OptModelType.BOW1
    else:
        # if not encoding in bow, we now use the pytorch setup
        par.enc.framework = Framework.TENSORFLOW
        par.enc.opt_model_type = OptModelType.OPT_13b

    if args.eight == 1:
        if args.legal == 1:
            print('START EIGHT, LEGAL', flush=True)
            par.enc.news_source = NewsSource.EIGHT_LEGAL
            id_col, save_size, batch_size, year, df = load_and_process_eight_k_legal_or_pressed(par, args, press_or_legal='legal')

            # launch the vectorisation
            vectorise_in_batch(id_col=id_col, df=df, save_size=save_size, batch_size=batch_size, par=par, year=year)

    if args.eight == 1:
        if args.legal == 0:
            print('START EIGHT, PRESS', flush=True)
            par.enc.news_source = NewsSource.EIGHT_PRESS
            id_col, save_size, batch_size, year, df = load_and_process_eight_k_legal_or_pressed(par, args, press_or_legal='press')
            # launch the vectorisation
            vectorise_in_batch(id_col=id_col, df=df, save_size=save_size, batch_size=batch_size, par=par, year=year)

    if args.wsj == 1:
        if args.one_per_news == 1:
            print('START WSJ, ONE PER NEWS (exploded)', flush=True)
            par.enc.news_source = NewsSource.WSJ_ONE_PER_STOCK
            id_col, save_size, batch_size, year, df = load_and_process_wsj(par, args)
            # launch the vectorisation
            vectorise_in_batch(id_col=id_col, df=df, save_size=save_size, batch_size=batch_size, par=par, year=year)

    if args.news == 1:
        if args.ref == 1:
            # ANGELA THIS ALREADY RAN
            print('START NEWS, REF', flush=True)
            par.enc.news_source = NewsSource.NEWS_REF
            # par.enc.opt_model_type = OptModelType.OPT_125m
            id_col, save_size, batch_size, year, df = load_and_process_news_one_stock_ref_or_third(par, args, 'ref')

        else:
            print('START NEWS, THIRD', flush=True)
            par.enc.news_source = NewsSource.NEWS_THIRD
            id_col, save_size, batch_size, year, df = load_and_process_news_one_stock_ref_or_third(par, args, 'third')
        # launch the vectorisation
        # TO KEEP, this can be usefull if you one day need to change the processing size
        # df, max_id_processed = drop_already_process_text_from_df(df, par)
        max_id_processed = 0

        vectorise_in_batch(id_col=id_col, df=df, save_size=save_size, batch_size=batch_size, par=par, year=year, start_save_id=max_id_processed)

    # data = Data(par)
    # save_dir = data.p_to_vec_main_dir+'/single_stock_news_to_vec/'
    # years = np.unique(np.sort([int(x.split('_')[1].split('.')[0]) for x in os.listdir(save_dir)]))[args.a] # len 27
    # ref = pd.read_pickle(save_dir+f'ref_{years}.p')
    # third = pd.read_pickle(save_dir+f'third_{years}.p')
