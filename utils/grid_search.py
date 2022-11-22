import inspect
import torch
import itertools
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

# randomness control
import random
import numpy as np

# scoring fuctions
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


class SearchBase:
    def __init__(self, X_train, y_train, X_test=None, y_test=None, y_ref=None, randomness_control=True):
        self.param_dict = {
            'linear': {
                'fit_intercept': [True, False]
            },
            'ridge': {
                'alpha': [1e-12, 1e-10, 1e-6,
                          1e-4, 1e-3, 1e-2, 1e-1,
                          1, 10, 100, 1000, 10000, 100000],
            },
            'lasso': {
                'alpha': [1e-12, 1e-10, 1e-6,
                          1e-4, 1e-3, 1e-2, 1e-1,
                          1, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]
            },
            'poly': {
                'degree': [1, 2, 3, 4, 5]
            },
            'xgb': {
                'booster': ['gbtree', 'dart'],
                'n_estimators': [1000, 750, 500, 300, 200, 100, 50, 10],
                'learning_rate': [0.001, 0.01, 0.1, 1],
                'max_depth': [3, 6, 9, 12, 15],
                'subsample': [0.75, 0.8, 0.85, 0.9],
                'colsample_bytree': [1],
            },
            'xgb_linear': {
                'reg_alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                'eval_metric': ['rmse'],
                'booster': ['gblinear'],
                'n_estimators': [100, 50, 10],
                'learning_rate': [0.001, 0.01, 0.1, 1],

            }
        }
        if (X_test is None) or (y_test is None):
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_ref = y_ref if y_ref else deepcopy(y_test)

        if randomness_control:
            self._randomness_control()

    def __call__(self, estimator, param_grid=None, poly=False):
        return self.search(estimator, param_grid, poly)

    def _randomness_control(self):
        random_seed = 42
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)

    def _print(self, model):
        return inspect.unwrap(model.__init__)

    def _product_dict(self, **kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def scoring(self, y_test, y_hat, y_ref, verbose):
        r2 = r2_score(y_test, y_hat)
        mse = mean_squared_error(y_test, y_hat)
        pearson, p_p_value = pearsonr(y_ref, y_hat)
        spearman, s_p_value = spearmanr(y_ref, y_hat)
        k_tau, k_p_value = kendalltau(y_ref, y_hat)
        if verbose:
            print(f"R2: {r2:.6f}")
            print(f"MSE: {mse:.6f}")
            print(f"pearsonr: {pearson:.6f}")
            print(f"spearmanr: {spearman:.6f}")
            print(f"kendalltau: {k_tau:.6f}")
            print()
        return r2, mse, pearson, spearman, k_tau, p_p_value, s_p_value, k_p_value

    def search(self, estimator, param_grid=None, poly=False, verbose=False):
        self.parameters, self.best_estimator, self.best_parameters, self.best_result = None, None, None, None
        self.search_df = None
        if param_grid is None:
            param_grid = self.param_dict[estimator]
        if poly:
            param_grid.update(self.param_dict['poly'])
        _model = self.load(estimator)
        parameters = list(self._product_dict(**param_grid))
        self.parameters = deepcopy(parameters)
        result_list = []
        max_value = -100
        for candidates in tqdm(parameters):
            if poly:
                if 'degree' in candidates:
                    degree = candidates.pop('degree')
                model = Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=True)),
                                  ('linear', _model(**candidates))])
            else:
                model = _model(**candidates)
            model.fit(self.X_train, self.y_train)
            y_hat = model.predict(self.X_test)
            y_ref = self.y_ref if self.y_ref is not None else self.y_test
            result = self.scoring(self.y_test, y_hat, y_ref, verbose)
            if result[0] > max_value:
                self.best_estimator = model
                self.best_parameters = candidates
                self.best_result = result
                max_value = result[0]
            result_list.append(result)
        search_df = pd.DataFrame(result_list, columns=['r2', 'mse', 'pearson', 'spearman', 'kendalltau',
                                                       'pearson_p_value', 'spearman_p_value', 'kendalltau_p_value'])
        self.search_df = search_df

    def load(self, estimator):
        self.estimator = estimator
        if estimator == 'linear':
            try:
                from sklearn.linear_model import LinearRegression
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install sklearn with: `pip install scikit-learn`"
                )
            return LinearRegression

        if estimator == 'ridge':
            try:
                from sklearn.linear_model import Ridge
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install sklearn with: `pip install scikit-learn`"
                )
            return Ridge

        if estimator == 'lasso':
            try:
                from sklearn.linear_model import Lasso
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install sklearn with: `pip install scikit-learn`"
                )
            return Lasso

        if estimator == 'xgb' or estimator == 'xgb_linear':
            try:
                from xgboost import XGBRegressor
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install xgboost with: `pip install xgboost`"
                )
            return XGBRegressor

