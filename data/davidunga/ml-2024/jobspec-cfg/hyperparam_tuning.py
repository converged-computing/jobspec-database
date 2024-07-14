from sklearn.experimental import enable_halving_search_cv  # required
from sklearn import model_selection
from optim_search_cv import OptimSearchCV
from copy import deepcopy
from data import load_data, build_data_prep_pipe, build_cv_pipe, stratified_split
from config import get_base_config, inherit_from_config, FROM_CONFIG, get_config_name, get_modified_config
from typing import Dict, List, Tuple
from object_builder import ObjectBuilder
from estimator_params_manager import EstimatorParamsManager
import numpy as np
import cv_result_manager
from itertools import product
from pipe_cross_val import set_pipecv
import os

os.environ['PYTHONWARNINGS'] = 'ignore'

object_builder = ObjectBuilder(['lightgbm', 'xgboost', 'catboost', model_selection, OptimSearchCV,
                                'sklearn.ensemble', 'sklearn.linear_model', 'sklearn.svm'])

# ---------------------------------------
# Grids:

config_grid = {
    'estimator': ['LogisticRegression', 'LinearSVM', 'RandomForestClassifier',
                  'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier'],
    'cv.base.searcher': ['GridSearchCV'],
    'cv.fine.searcher': ['OptimSearchCV', 'RandomizedSearchCV'],
    'balance.method': ['none', 'SMOTENC', 'NearMiss', 'InstanceHardnessThreshold'],
    'random_state': list(range(10))
}

# -------
# params grid for each balancer
# included balancers are defined in config_grid

balance_params_grid = {
    'none': {
    },
    'RandomUnderSampler': {
        'random_state': [FROM_CONFIG],
    },
    'SDVGauss': {
    },
    'SDVGAN': {
    },
    'SMOTENC': {
        'random_state': [FROM_CONFIG],
        'categorical_features': ['auto'],
        'k_neighbors': [3, 5]
    },
    'NearMiss': {
        'version': [1, 2, 3],
        'n_neighbors': [3],
    },
    'InstanceHardnessThreshold': {
        'estimator': [None, ('sklearn.ensemble.RandomForestClassifier',
                             {'n_estimators': 10, 'random_state': 1})],
        'cv': [3, 5],
        'random_state': [FROM_CONFIG]
    }
}

# -------
# params grid for each estimator

_common_base_grid = {
    'max_depth': np.arange(2, 10),
    'learning_rate': [5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    'n_estimators': [50, 100, 200, 500]
}

estimator_params_grids = {
    'LogisticRegression': {
        'base': {
            'C': [.05, .5, 1., 5.],
            'penalty': [None, 'l2'],
            'class_weight': ['balanced', None]
        },
        'fine': {
            'C': [.05, .1, .5, 1., 5., 10.],
        }
    },
    'LinearSVM': {
        'base': {
            'C': [.05, .5, 1., 5.],
            'penalty': ['l1', 'l2'],
            'class_weight': ['balanced', None]
        },
        'fine': {
            'C': [.05, .1, .5, 1., 5., 10.],
        }
    },
    'RandomForestClassifier': {
        'base': {
            'max_depth': np.arange(2, 20),
            'n_estimators': [10, 20, 50, 100, 200],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        },
        'fine': {
            'min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1, 0.2, 0.3],
            'min_samples_split': np.arange(2, 10),
            'min_samples_leaf': np.arange(1, 5)
        }
    },
    'XGBClassifier': {
        'base': _common_base_grid,
        'fine': {
            'min_child_weight': np.arange(1, 10),
            'gamma': np.linspace(0, 1, 20),
            'subsample': np.linspace(0.4, 1.0, 20),
            'colsample_bytree': np.linspace(0.5, 1.0, 20),
            'reg_alpha': [5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            'reg_lambda': [5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            'scale_pos_weight': [0.5, 1.0, 2.0, 5.0, 10.0]
        }
    },
    'LGBMClassifier': {
        'base': _common_base_grid,
        'fine': {
            'min_child_weight': np.arange(1, 10),
            'subsample': np.linspace(0.4, 1.0, 20),
            'colsample_bytree': np.linspace(0.5, 1.0, 20),
            'num_leaves': np.arange(20, 150, 10),
            'reg_alpha': [0, 0.1, 1, 10],
            'reg_lambda': [0, 0.1, 1, 10]
        }
    },
    'CatBoostClassifier': {
        'base': _common_base_grid,
        'fine': {
            'bagging_temperature': [0, 0.25, 0.5, 0.75, 1, 2, 5, 10],
            'depth': np.arange(4, 12),
            'l2_leaf_reg': [1, 3, 5, 7, 9]
        }
    }
}

# ---------------------------------------
# Search funcs:


def cv_search(fine_tune: bool, skip_existing: bool = True, ests: List[str] = None):
    """ Entry point for cv search. Iterates over the configurations defined by the grids,
        and cv-searches best estimator params for each configuration.
    """
    if ests is not None:
        ests = [_est_shortname(est) for est in ests]
        assert set(ests).issubset([_est_shortname(est) for est in config_grid['estimator']])

    base_config = get_base_config()
    base_config['fine_tune'] = fine_tune
    for config in yield_from_grid(grid_dict=config_grid, default_dict=base_config):
        if ests is not None and _est_shortname(config['estimator']) not in ests:
            continue
        for balance_params in yield_from_grid(balance_params_grid[config['balance.method']]):
            config['balance.params'] = inherit_from_config(balance_params, config)
            cv_search_estimator_params(config, skip_existing)


def cv_search_estimator_params(config: Dict, skip_existing: bool):
    """ cv-search estimator params for a given configuration """

    config = deepcopy(config)
    fine_tune_config = config if config['fine_tune'] else None
    config = get_modified_config(config, fine_tune=False)  # first iteration is always none-finetune

    if skip_existing and cv_result_manager.get_result_files_for_config(config):
        if fine_tune_config is None or cv_result_manager.get_result_files_for_config(fine_tune_config):
            print(f"{get_config_name(config)}: Skipping (exists)")
            return

    print(f"\n--------- {get_config_name(fine_tune_config if fine_tune_config else config)}:\n")

    DEV = False
    seed = config['random_state']
    params_manager = EstimatorParamsManager(config['estimator'])
    cv_args = {'cv': config['cv.n_folds'], 'scoring': config['cv.scores'], 'verbose': 0,
               'refit': config['cv.main_score'], 'return_train_score': True, 'n_jobs': -1}

    # -----
    # prepare datasets:

    # load and prep full data
    raw_data = load_data(config)
    prep_pipe = build_data_prep_pipe(config)  # doesn't include standardizing, balancing, and one-hotting
    Xy = prep_pipe.fit_transform(raw_data)

    # build pipe that goes into the cross validation (=standardizing, balancing, one-hotting)
    cv_pipe = build_cv_pipe(config, Xy)  # Xy is used to initialize the onehot encoder

    def _run_search(params: Dict, config: Dict):

        stage = 'fine' if config['fine_tune'] else 'base'
        estimator_name = config['estimator']
        cv_searcher_name = config[f'cv.{stage}.searcher']

        # -----
        # split data

        param_grid = estimator_params_grids[estimator_name][stage]

        Xy_train, Xy_test = stratified_split(Xy, test_size=config['data.test_size'], seed=seed)

        fit_kws = {}
        if params.get('early_stopping_rounds', None):
            assert stage == 'base'
            # make eval set for early stopping
            Xy_train, Xy_eval = stratified_split(Xy_train, test_size=config['cv.early_stopping_eval_size'], seed=seed)
            fit_kws['eval_set'] = [cv_pipe.fit(Xy_train).transform(Xy_eval)]  # eval set isn't passed through cv-pipe

        # -----
        # initialize search:

        estimator = object_builder.get_instance(estimator_name, params)
        cv_searcher_cls = object_builder.get_class(cv_searcher_name)

        if DEV:
            print("\n\n !! Running in DEV mode !! \n\n")
            #param_grid = _reduce_grid(param_grid)
            if cv_searcher_cls.__name__ == "OptimSearchCV":
                cv_args['visualize'] = True

        try:
            cv_searcher = cv_searcher_cls(estimator, random_state=seed, param_distributions=param_grid, **cv_args)
        except TypeError:
            cv_searcher = cv_searcher_cls(estimator, param_grid=param_grid, **cv_args)

        set_pipecv(cv_searcher, cv_pipe)

        # -----
        # search:

        print("\n\n ------ ")
        print("Starting search for:", get_config_name(config))
        print(f"  Grid shape={tuple(len(ax) for ax in param_grid)}, size={np.prod([len(ax) for ax in param_grid])}")
        print(f"  Estimator params: {params}")
        print("\n\n")

        cv_searcher.fit(*Xy_train, **fit_kws)
        params = cv_searcher.best_params_
        if params.get('early_stopping_rounds', None):
            params['n_estimators'] = params_manager.get_best_iteration(cv_searcher.best_estimator_)

        cv_result_manager.process_result(config, cv_searcher, save=not DEV)

        return params

    params = params_manager.adjust_to_estimator(
        {'random_state': seed, 'early_stopping_rounds': config['cv.early_stopping_rounds'],
         'eval_metric': config['cv.early_stopping_eval_metric']})

    params = _run_search(params, config)
    if fine_tune_config:
        _run_search(params, fine_tune_config)


# ---------------------------------------
# Helper funcs:

def yield_from_grid(grid_dict: Dict, default_dict: Dict = None):
    default_dict = deepcopy(default_dict) if default_dict else {}
    keys = grid_dict.keys()
    if default_dict:
        assert set(keys).issubset(default_dict.keys())
    for vals in product(*grid_dict.values()):
        result = default_dict
        result.update(dict(zip(keys, vals)))
        yield result


def _reduce_grid(grid: Dict) -> Dict:
    return {k: [v[0], v[-1]] if len(v) > 1 else v for k, v in grid.items()}


def _est_shortname(fullname: str) -> str:
    return fullname.lower().replace('classifier', '')


# ---------------------------------------

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-e", "--ests")
    parser.add_argument("--no-finetune", action="store_false", dest="fine_tune", default=True)
    args = parser.parse_args()
    ests = [s.strip() for s in args.ests.split(",")] if args.ests else None
    cv_search(fine_tune=args.fine_tune, ests=ests)
