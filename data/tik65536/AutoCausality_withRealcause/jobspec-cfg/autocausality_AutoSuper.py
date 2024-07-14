import pickle
import warnings
import time
warnings.filterwarnings('ignore') # suppress sklearn deprecation warnings for now..

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from auto_causality import AutoCausality
from auto_causality.datasets import generate_synthetic_data
from auto_causality.data_utils import CausalityDataset
from loading import load_realcause_dataset
from data.lbidd import load_lbidd
import pandas as pd
import numpy as np
import argparse
import pymp
metrics = ["effectMSE"]
estimator_list = "all"
out_dir = "./Super_NewData/"
filename_out = "synthetic"
#datapath="./DataSet/"
parser = argparse.ArgumentParser()

parser.add_argument('-dataset', nargs="*", default=['lalonde_psid'],
                    help='lalonde_psid , lalonde_cps , twins, lbidd ' )

parser.add_argument('-numOfSample', type=int, default=4,
                    help='Numer of Sub Dataset to use')

parser.add_argument('-budget', type=int, default=300,
                    help='Time budget fro Components Model')

parser.add_argument('-mp', type=int, default=1,
                    help='Number of parallel process')
args = parser.parse_args()
datasetname=args.dataset
numOfSample=args.numOfSample
components_time_budget = args.budget
mp = args.mp
run=np.arange(80)
test=np.arange(80,100)
np.random.shuffle(run)

trainingset=[]
testset=[]

linkfunc=['linear','quadratic','cubic','log','exp']
log_degy=[  8,  11,   9,   5,  20,  19,  17,  13,   6,  21,  27,  28,  24,
        37,  32,  38,  47,  12,  79,  82,  98,   7,  10,  15,  35,  44,
        30,  25,  16,  88,  76,  99,  53,  89,   4,   2,  14,  18,  40,
        33,  51,  36,  48,  62,  41,   3,  31,  43,  42,  97,  59, 101,
        34,  45,  78,  87,  86,  23,  22,  58,  52,  50,  75]

exp_degy=[  8,  11,   9,   5,  20,  19,  17,  13,   6,  21,  27,  28,  24,
        37,  38,  12,  79,  82,  93,  75,  98,   7,  10,  15,  35,  44,
        32,  25,  16,  47,  76,  99,  53,  89,   4,   2,  14,  18,  33,
        51,  48,  62,  41,  88,   3,  31,  43,  42,  97,  54,  59, 101,
        45,  87,  86,  23,  40,  22,  58]

if('lalonde_psid' in datasetname):
    run=run[:numOfSample]
    print(f"Dataset ID:{run} Start",flush=True)
    df = load_realcause_dataset('lalonde_psid', int(run[0]))
    for di in run[1:]:
       df=df.append(load_realcause_dataset(datasetname,int(di)))
    print(f'Data Set Shape : {df.shape}',flush=True)
    df.rename(columns = {'t':'treatment','y':'outcome','ite':'true_effect'}, inplace = True)
    df.drop(['y0','y1'],axis=1,inplace=True)
    trainingset.append(df)
    t = int(np.random.choice(test,1))
    tmp =  load_realcause_dataset('lalonde_psid', t)
    tmp.rename(columns = {'t':'treatment','y':'outcome','ite':'true_effect'}, inplace = True)
    tmp.drop(['y0','y1'],axis=1,inplace=True)
    testset.append(tmp)

if('lbidd' in datasetname):
    param=pd.read_csv('./datasets/lbidd/scaling/params.csv')
    #for i in linkfunc[:3]:
    #    d=load_lbidd(n=25000,link=i,return_ites=True,n_shared_parents=None)
    #    columns=[ f'W_{idx}' for idx in range(d['w'][0].shape[0]) ]
    #    columns+=['treatment','outcome','true_effect']
    #    dat=np.hstack((d['w'],d['t'].reshape(-1,1),d['y'].reshape(-1,1),d['ites'].reshape(-1,1)))
    #    idx=np.arange(25000)
    #    np.random.shuffle(idx)
    #    df=pd.DataFrame(dat[idx[:20000]],columns=columns)
    #    df.name = f'lbidd_{i}'
    #    trainingset.append(df)
    #    df=pd.DataFrame(dat[idx[20000:]],columns=columns)
    #    df.name = f'lbidd_{i}'
    #    testset.append(df)

    # log linkfunc
    for i in log_degy:
        degzlist=param[(param['size']==25000) & (param['link_type']=='log') & (param['deg(y)']==i)]['deg(z)'].unique()
        for j in degzlist:
            print(f'lbidd log link func, deg(y):{i} , deg(z):{j}')
            d=load_lbidd(n=25000,link='log',degree_y=i,degree_t=j,return_ites=True,n_shared_parents=None)
            columns=[ f'W_{idx}' for idx in range(d['w'][0].shape[0]) ]
            columns+=['treatment','outcome','true_effect']
            dat=np.hstack((d['w'],d['t'].reshape(-1,1),d['y'].reshape(-1,1),d['ites'].reshape(-1,1)))
            idx=np.arange(25000)
            np.random.shuffle(idx)
            df=pd.DataFrame(dat[idx[:20000]],columns=columns)
            df.name = f'lbidd_log_degy_{i}_degj_{j}'
            trainingset.append(df)
            df=pd.DataFrame(dat[idx[20000:]],columns=columns)
            df.name = f'lbidd_log_degy_{i}_degj_{j}'
            testset.append(df)
    # exp linkfunc
    for i in exp_degy:
        degzlist=param[(param['size']==25000) & (param['link_type']=='exp') & (param['deg(y)']==i)]['deg(z)'].unique()
        for j in degzlist:
            print(f'lbidd exp link func, deg(y):{i} , deg(z):{j}')
            d=load_lbidd(n=25000,link='exp',degree_y=i,degree_t=j,return_ites=True,n_shared_parents=None)
            columns=[ f'W_{idx}' for idx in range(d['w'][0].shape[0]) ]
            columns+=['treatment','outcome','true_effect']
            dat=np.hstack((d['w'],d['t'].reshape(-1,1),d['y'].reshape(-1,1),d['ites'].reshape(-1,1)))
            idx=np.arange(25000)
            np.random.shuffle(idx)
            df=pd.DataFrame(dat[idx[:20000]],columns=columns)
            df.name = f'lbidd_exp_degy_{i}_degj_{j}'
            trainingset.append(df)
            df=pd.DataFrame(dat[idx[20000:]],columns=columns)
            df.name = f'lbidd_exp_degy_{i}_degj_{j}'
            testset.append(df)

print(f'TrainingSet len :{len(trainingset)}')
print(f'TestSet len :{len(testset)}')
with pymp.Parallel(mp) as p:
    for index in p.range(len(trainingset)):
        train_df = trainingset[index]
        df_test = testset[index]
        print(f'Training Dataset {train_df.name} Start , shape : {train_df.shape}')
        features_X=list(train_df.columns[:-3])
        print(f"features_X: {features_X}",flush=True)
        starttime=time.time()

        for metric in metrics:
            ac = AutoCausality(
                metric=metric,
                verbose=1,
                components_verbose=1,
                components_time_budget=components_time_budget,
                estimator_list=estimator_list,
                store_all_estimators=True,
                propensity_model="super",
            )

            ac.fit(
                train_df,
                treatment="treatment",
                outcome=["outcome"],
                #common_causes=features_W,
                effect_modifiers=features_X,
            )
            # compute relevant scores (skip newdummy)
            # get scores on train,val,test for each trial,
            # sort trials by validation set performance
            # assign trials to estimators
            estimator_scores = {est: [] for est in ac.scores.keys() if "NewDummy" not in est}
            ds_name='test'
            data=None
            print(f'Test Dataset {df_test.name} Start , shape : {df_test.shape}')
            if not isinstance(df_test, CausalityDataset):
                assert isinstance(df_test, pd.DataFrame)
                data = CausalityDataset(
                    df_test,
                    treatment="treatment",
                    outcomes=["outcome"],
                    #common_causes=features_W,
                    effect_modifiers=features_X,
                )
            for trial in ac.results.trials:
                # estimator name:
                estimator_name = trial.last_result["estimator_name"]
                print(f"Dataset {df_test.name}  Make Score {estimator_name} ",flush=True)
                if  trial.last_result["estimator"]:
                    estimator = trial.last_result["estimator"]
                    scores = {}
                    scores[ds_name] = {}
                    # make scores
                    est_scores = ac.scorer.make_scores(
                        estimator,
                        data.data,
                        #problem=ac.problem,
                        metrics_to_report=ac.metrics_to_report,
                    )

                    # add cate:
                    scores[ds_name]["CATE_estimate"] = estimator.estimator.effect(df_test)
                    # add ground truth for convenience
                    scores[ds_name]["CATE_groundtruth"] = df["true_effect"]
                    scores[ds_name][metric] = est_scores[metric]
                    try:
                        scores[ds_name]['#_Propensity_model']=est_scores['#_Propensity_model']
                        scores[ds_name]['#_Propensity_Para']=est_scores['#_Propensity_model_param']
                        scores[ds_name]['values']=est_scores['values']
                    except KeyError:
                        pass
                    estimator_scores[estimator_name].append(scores)

            # sort trials by validation performance
            for k in estimator_scores.keys():
                estimator_scores[k] = sorted(
                    estimator_scores[k],
                key=lambda x: x["test"][metric],
                reverse=False if (metric == "energy_distance" or metric =="effectMSE") else True,
            )
        results = {
            "best_estimator": ac.best_estimator,
            "best_config": ac.best_config,
            "best_score": ac.best_score,
            "optimised_metric": metric,
            "dataset": df_test.name,
            "scores_per_estimator": estimator_scores,
        }
        print(f"Dataset {train_df.name} End {time.time()-starttime} , best {ac.best_estimator}",flush=True)



        with open(f"{out_dir}{filename_out}_{metric}_{index}_{df_test.name}_run_test.pkl", "wb") as f:
            pickle.dump(results, f)
