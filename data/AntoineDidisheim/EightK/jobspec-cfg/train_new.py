import numpy as np
import pandas as pd
import os
from parameters import *
from data import *
from utils_local.nlp_ticker import *
from didipack.utils_didi.ridge import run_efficient_ridge
from didipack.trainer.trainer_ridge import TrainerRidge
# from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
from didipack.trainer.trainer_logistic_elastic_net import TrainerLogisticElasticNet
from didipack.trainer.train_splitter import get_start_dates,get_chunks,get_tasks_for_current_node
import psutil
from utils_local.general import *
from utils_local.trainer_specials import *
from experiments_params import get_main_experiments,get_main_experiments_train_all

def load_data_for_this_chunks(par: Params):
    # load the correct data
    start = par.grid.year_id - par.train.T_train
    end = par.grid.year_id + par.train.testing_window
    load_dir = par.get_training_norm_dir()
    print('Load dir check',load_dir, flush=True)
    years_in_the_list = np.sort(np.unique([int(x.split('_')[1].split('.')[0]) for x in os.listdir(load_dir)]))
    df = pd.DataFrame()
    x = pd.DataFrame()
    # os.listdir('/data/gpfs/projects/punim2039/EightK/data/training_norm/OPT_13b/EIGHT_LEGAL/ZSCORE/')
    for year in years_in_the_list:
        if (year >= start) & (year <= end):
            df = pd.concat([df, pd.read_pickle(load_dir + f'df_{year}.p')], axis=0)
            x = pd.concat([x, pd.read_pickle(load_dir + f'x_{year}.p')], axis = 0)


    df = df.reset_index(drop=True)
    x = x.reset_index(drop=True)




    if par.train.abny is not None:
        # ugly set of conditions ot make it compatible with old versions
        if par.train.abny ==True:
            y = df[['abret']]
        if par.train.abny =='abn20':
            temp = Data(par).load_abn_return()
            temp = temp.loc[temp['evttime']>=-1,:]
            temp = temp.groupby(['date','permno'])['abret'].mean().reset_index().rename(columns={'abret':'abret_long'})
            df = df.merge(temp,how='left')
            ind = pd.isna(df['abret_long'])
            df.loc[ind,'abret_long'] = df.loc[ind,'abret']
            y = df[['abret_long']]
    else:
        y = df[['ret']]
    y = np.sign(y)
    y = y.replace({0: 1})
    if par.train.news_filter_training is not None:
        # 'news0', 'rtime_nr', 'news0_nr', 'news_with_time', 'news_with_time_nr'
        ind_news = df[par.train.news_filter_training] == 1
        id_a = df['id'].apply(lambda x: str(x).split('-')[0])
        id_b = df[par.train.news_filter_training].astype(str)
        df['id'] = id_a +'-' +id_b
        ids = df['id']
    else:
        ids = df['id']

    dates = df['date'].dt.year

    ind = ~pd.isna(y.iloc[:,0])
    x = x.loc[ind,:]
    y = y.loc[ind,:]
    dates = dates.loc[ind]
    ids = ids.loc[ind]

    print('Data loaded with shape', flush=True)
    print(x.shape,y.shape,flush=True)

    return x,y,dates,ids

def generate_fake_data(N: int = 1000, P: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Generates a dataset of fake data for testing purposes.

    Args:
        N (int): The number of rows in the generated data.
        P (int): The number of columns in the generated data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the fake data, its mean along the columns, a series representing dates, and a series representing ids.
    """
    fake_data = pd.DataFrame(np.random.normal(size=(N, P)))
    y = pd.DataFrame(np.sign(fake_data.mean(1)))
    dates = pd.Series(np.arange(N))
    ids = dates.copy()

    return fake_data, y, dates, ids

if __name__ == "__main__":
    args = didi.parse()
    print('START WORKING ON ',args.a,flush=True)
    par = get_main_experiments_train_all(args.a, train=True)
    print('Parameter defined',flush=True)
    par.print_values()
    trainer = chose_trainer(par)
    temp_save_dir = par.get_res_dir()
    print(temp_save_dir,flush=True)

    already_processed = os.listdir(temp_save_dir)
    save_name = f'{par.grid.year_id}.p'

    if save_name in already_processed:
        print(f'Already processed {save_name}', flush=True)
    else:
        x, y, dates, ids = load_data_for_this_chunks(par)
        y_test, _ = trainer.train_at_time_t(x=x, y=y, ids=ids, times=dates, t_index_for_split=par.grid.year_id)
        y_test.to_pickle(temp_save_dir + save_name)
        print(y_test, flush=True)
        print('saved to',temp_save_dir+save_name,flush=True)