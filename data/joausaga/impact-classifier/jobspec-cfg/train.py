import click
import glob
import joblib
import numpy as np
import pandas as pd
import os
import sys

from datetime import datetime
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from tqdm import tqdm
from utils import save_model, get_features_and_labels, plot_learning_curve


@click.group()
def run():
    pass


def define_parameters_gb():
    """
    Define hyper-parameters of
    Gradient Boosting

    Inspired by
    https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling/notebook#Titanic-Top-4%-with-ensemble-modeling
    """
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 7, 10]
    min_samples_leaf = [1, 2, 3]
    loss = ['deviance']
    learning_rate = [0.1, 0.05, 0.01]
    param_grid = {'loss' : loss,
                  'n_estimators' : n_estimators,
                  'learning_rate': learning_rate,
                  'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf,
                  'max_features': max_features,
                  'min_samples_split': min_samples_split
                 }
    return param_grid


def define_parameters_rf():
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 7, 10]
    min_samples_leaf = [1, 2, 3]
    bootstrap = [True, False]
    param_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }
    return param_grid


def define_parameters_lr():
    """
    Define hyper-parameters of
    Logistic Regression.

    Inspired by
    https://chrisalbon.com/machine_learning/model_selection/hyperparameter_tuning_using_grid_search
    """
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(0, 4, 10)
    }
    return param_grid


def define_parameters_svm():
    """
    Define hyper-parameters of
    Support Vector Machine.

    Inspired by
    https://towardsdatascience.com/svm-hyper-parameter-tuning-using-gridsearchcv-49c0bc55ce29
    """
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001, 'scale'],
        'kernel': ['linear','rbf', 'poly']
    }
    return param_grid


def do_hyperparametrization(classifier, param_grid, X_train, y_train, n_iter, 
                            kfold, metric, n_jobs, verbose):
    rs_clf = RandomizedSearchCV(classifier, param_distributions=param_grid, 
                                n_iter=n_iter, cv=kfold, scoring=metric, 
                                n_jobs=n_jobs, verbose=verbose)
    rs_clf.fit(X_train, y_train)
    print(f'Best score: {rs_clf.best_score_}')
    return rs_clf.best_estimator_


@run.command()
@click.argument('algorithm_acronym')
@click.argument('train_data_file')
@click.option('--algorithm_name', help='Name of algorithm to be trained', \
              default='', is_flag=False)
@click.option('--num_splits', help='Number of cross-validation splits', \
              default=3, is_flag=False)
@click.option('--num_iter', help='Number of iterations', \
              default=100, is_flag=False)
@click.option('--metric', help='Optimization metric', \
              default='accuracy', is_flag=False)
def train_model(algorithm_acronym, train_data_file, algorithm_name, num_splits, 
                num_iter, metric):
    do_train_model(algorithm_acronym, train_data_file, algorithm_name, num_splits, 
                   num_iter, metric)


def do_train_model(algorithm_acronym, train_data_file, algorithm_name, num_splits, 
                   num_iter, metric):
    # get data    
    with open(train_data_file, 'rb') as f:
        data = joblib.load(f)
    train_data = data['data']
    y_train = train_data['label'].values
    text_features = list(train_data.iloc[:,0].values)        
    extra_features = np.array(train_data.iloc[:,2:].values)
    X_train = np.concatenate((text_features, extra_features), axis=1)
     # define cross-validation
    kfold = KFold(n_splits=int(num_splits), shuffle=True, random_state=42)
    if algorithm_acronym == 'GB':
        param_grid = define_parameters_gb()
        classifier = GradientBoostingClassifier()
    elif algorithm_acronym == 'RF':
        param_grid = define_parameters_rf()
        classifier = RandomForestClassifier()
    elif algorithm_acronym == 'LR':
        param_grid = define_parameters_lr()
        classifier = LogisticRegression()
    elif algorithm_acronym == 'SVM':
        param_grid = define_parameters_svm()
        classifier = SVC()
    else:
        raise Exception(f'Unknown algorithm acronym: {algorithm_acronym}')
    best_model = do_hyperparametrization(classifier, param_grid, X_train, y_train, 
                                         int(num_iter), kfold, metric, -1, 1)
    save_model(best_model, f'best_{algorithm_name}_{metric}', 'data', train_data_file)
    return best_model


def get_classifier(algorithm_name, random_state):
    if algorithm_name == 'NB':
        classifier = GaussianNB()
    elif algorithm_name == 'SVM':
        classifier = SVC(random_state=random_state)   
    elif algorithm_name == 'LR':
        classifier = LogisticRegression(random_state=random_state)
    elif algorithm_name == 'RF':        
        classifier = RandomForestClassifier(random_state=random_state)
    elif algorithm_name == 'GB':
        classifier = GradientBoostingClassifier(random_state=random_state)
    else:
        print("Unknown algorithm: {0}",format(algorithm_name))
    return classifier


@run.command()
@click.option('--num_splits', help='Number of cross-validation splits', \
              default=5, is_flag=False)
@click.option('--num_iter', help='Number of iterations', \
              default=100, is_flag=False)
@click.option('--metric', help='Optimization metric', \
              default='accuracy', is_flag=False)
def train_models(num_splits, num_iter, metric):
    random_state = np.random.RandomState(1234)
    outputs = []
    data_dir = os.path.join('data', 'train')
    train_files = os.path.join(data_dir, '*.pkl')    
    algorithms = [
        {'name': 'naive-bayes', 'acronym': 'NB'},
        {'name': 'support-vector-machine', 'acronym': 'SVM'},
        {'name': 'logistic-regression', 'acronym': 'LR'},
        {'name': 'random-forest', 'acronym': 'RF'},
        {'name': 'gradient-boosting', 'acronym': 'GB'}
    ]
    # Train models
    print('Training models')
    kfold = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    files = glob.glob(train_files)
    num_files = len([name for name in os.listdir(data_dir) if os.path.isfile(name)])
    num_loops = num_files * len(algorithms)
    with tqdm(total=num_loops, file=sys.stdout) as pbar:
        for file in files:
            with open(file, 'rb') as f:
                data = joblib.load(f)
            train_data = data['data']
            y_train = train_data['label'].values
            text_features = list(train_data.iloc[:,0].values)        
            extra_features = np.array(train_data.iloc[:,2:].values)
            X_train = np.concatenate((text_features, extra_features), axis=1)
            for algorithm in algorithms:  
                classifier = get_classifier(algorithm['acronym'], random_state)                
                scores = cross_val_score(classifier, X_train, y=y_train, 
                                         scoring=metric, cv=kfold, n_jobs=-1)
                outputs.append(
                    {
                        'algorithm': algorithm['acronym'],
                        'train_filename': file,                
                        'metric_scores': scores,
                    }
                )
                pbar.update(1)
    # Load results in a dataframe
    print('Loading experiment results in a dataframe')
    output_df = pd.DataFrame(columns=['algorithm', 'train_data_file', f'mean_{metric}'])
    for output in outputs:
        row = {
            'algorithm': output['algorithm'],
            'train_data_file': output['train_filename'],
            f'mean_{metric}': round(output['metric_scores'].mean(), 2),
            f'std_{metric}': round(output['metric_scores'].std(), 2)
        }
        output_df = output_df.append(row, ignore_index=True)    
    # Save dataframe    
    experiment_dir = 'experiments'
    os.makedirs(experiment_dir, exist_ok=True)
    experiment_filename = 'e_{}.csv'.format(datetime.now().strftime('%d%m%Y_%H%M%S'))
    experiment_file_path = os.path.join(experiment_dir,experiment_filename)
    print(f'Saving experiment results in {experiment_file_path}')
    output_df.to_csv(experiment_file_path, index=False)
    # Train algorithms on data transformation that work best for each of them
    print('Doing hyperparametrization')
    with tqdm(total=len(algorithms), file=sys.stdout) as pbar:
        for algorithm in algorithms:            
            if algorithm['acronym'] != 'NB':
                best_model = output_df[output_df['algorithm']==algorithm['acronym']].\
                    sort_values(by=[f'mean_{metric}', f'std_{metric}'], ascending=False).head(1)
                train_data_file = best_model['train_data_file'].values[0]
                best_model = do_train_model(algorithm['acronym'], train_data_file, algorithm['name'], 
                                            num_splits, num_iter, metric)
                features, labels = get_features_and_labels(train_data_file)
                plot_learning_curve(best_model, f"{algorithm['acronym']} learning curves", 
                                    features, labels, metric, cv=kfold, shuffle=True, 
                                    save_fig=True)
            pbar.update(1)


if __name__ == "__main__":
    run()