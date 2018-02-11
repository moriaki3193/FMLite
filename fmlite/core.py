# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from itertools import combinations
from prettytable import PrettyTable
from .constants import VALIDATION_ERR_MSG


class FMLite:
    """2-way Factorization Machine
    """

    def __init__(
            self,
            k=10,
            n_epochs=1000,
            task='regression',
            mode='normal',
            n_entities=None,
            n_features=None,
            biased=True,
            verbose=True):
        """Setting hyper-params & semantics, etc...

        params
        ------
        k : int, hyper-parameter of FM.
            Defines the number of columns of V.
        mode : str, a mode of FM
            Possible values are either `normal` of `combination-dependent`.
            Default to `normal`.
        biased : bool, indicates if the model uses bias term.
            Default to True.
        verbose : bool, indicates if a log is verbose.
            Default to True.
        """
        if mode not in ['normal', 'combination-dependent']:
            raise ValueError(VALIDATION_ERR_MSG['mode'])
        # [START combination-dependent param validation]
        if mode == 'combination-dependent':
            if not isinstance(n_entities, int):
                raise ValueError(VALIDATION_ERR_MSG['cb'])
            elif not isinstance(n_features, int):
                raise ValueError(VALIDATION_ERR_MSG['cb'])
        # [END combination-dependent param validation]
        self.k = k
        self.mode = mode
        self.n_entities = n_entities
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.learning_rate = 0.001  # TODO 学習率が大きいと残渣が得られない
        self.b = None  # set when self.fit is called
        self.w = None  # set when self.fit is called
        self.V = None  # set when self.fit is called
        self.errors = np.array([])

    def _calc_y_pred(self, x):
        """1つの訓練事例に対する現時点での応答変数の推定値を返す.

        param
        -----
        x : np.array, shape = (n_features, )
        """
        interaction_sum = np.array([
            np.dot(self.V[i], self.V[j].T) * x[i] * x[j]
            for i, j in combinations(range(x.shape[0]), 2)
        ]).sum()
        return self.b + np.dot(self.w, x.T) + interaction_sum

    def _partial_normal_V(self, x):
        """Vの(i, f)成分でy_predを偏微分した値を返すメソッド
        """
        n_features = x.shape[0]
        _V = np.empty((n_features, self.k))
        for i in range(n_features):
            for f in range(self.k):
                s = 0.0
                for j in range(n_features):
                    s += self.V[j, f] * x[j]
                _V[i, f] = x[i] * s - self.V[i, f] * x[i] ** 2
        return np.asmatrix(_V)

    def _partial_cb_V(self, x):
        """Vの(i, f)成分でy_predを偏微分した値を返すメソッド

        0 <= i < n_entities && n_entities <= j < 2 * n_entities
        を満たす idx, jdx の組合せのみについてVの勾配を計算する.
        """
        M = self.n_entities
        n = self.n_features
        _V = np.empty((2 * M + n, self.k))
        for i in range(2 * M + n):
            for f in range(self.k):
                s = 0.0
                for j in range(2 * M + n):
                    if 0 <= i < M and 0 <= j < M:
                        continue
                    elif M <= i < 2 * M and M <= j < 2 * M:
                        continue
                    else:
                        s += self.V[j, f] * x[j]
                _V[i, f] = x[i] * s - self.V[i, f] * x[i] ** 2
        return np.asmatrix(_V)

    def _make_evaluate_grad_func(self):
        """LMSに対する偏微分の値を計算する「関数」を返すメソッド
        """
        if self.mode == 'normal':
            # [START normal version]
            def _calc_normal(x_i, diff):
                b_grad = 2 * diff
                w_grad = 2 * diff * x_i
                V_grad = 2 * diff * self._partial_normal_V(x_i)
                return (b_grad, w_grad, V_grad)
            return _calc_normal
            # [END normal version]
        elif self.mode == 'combination-dependent':
            # [START combination-dependent version]
            def _calc_cb(x_i, diff):
                b_grad = 2 * diff
                w_grad = 2 * diff * x_i
                V_grad = 2 * diff * self._partial_cb_V(x_i)
                return (b_grad, w_grad, V_grad)
            return _calc_cb
            # [END combination-dependent version]

    def _gradient_descent(self, X_train, y_train, type='stochastic'):
        """Optimization using Gradient Descent method.

        params
        ------
        X_train : np.matrix, shape = (n_samples, n_features)
        y_train : np.array, shape = (n_samples, )
        type : str, Possible values are either minibatch of stochastic.
            Default to `stochastic`.

        return
        ------
        errors : np.array, shape = (n_samples * n_epochs, )
        """
        errors = []
        n_samples, n_features = X_train.shape
        normal_params = {'scale': 0.01, 'size': (n_features, self.k)}
        self.b = 0.0
        self.w = np.zeros(n_features)
        self.V = np.asmatrix(np.random.normal(**normal_params))
        # [START 関数の固定化による高速化]
        _evaluate_grad = self._make_evaluate_grad_func()
        # [END 関数の固定化による高速化]
        # [START parameter estimation]
        for _ in tqdm(range(self.n_epochs)):
            for ind in np.random.permutation(n_samples):
                x = np.asarray(X_train[ind]).ravel()
                y = y_train[ind]
                y_tmp_pred = self._calc_y_pred(x)
                diff = y_tmp_pred - y
                errors.append(diff)
                b_grad, w_grad, V_grad = _evaluate_grad(x, diff)
                # [START updating parameters]
                self.b = self.b - self.learning_rate * b_grad
                self.w = self.w - self.learning_rate * w_grad
                self.V = self.V - self.learning_rate * V_grad
                # [END updating parameters]
        # [END parameter estimation]
        return np.asarray(errors)

    def fit(self, X_train, y_train):
        """Fitting using SGD methods.

        params
        ------
        X_train : np.matrix, shape = (n_samples, n_features)
        y_train : np.array or list, shape = (n_samples, )

        return
        ------
        self : an fitted instance of FMLite
        """
        if isinstance(y_train, list):
            y_train = np.asarray(y_train)
        if X_train.ndim != 2:
            raise ValueError('ndim of X must be 2.')
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('n_samples of X and y must be equal.')
        self.errors = self._gradient_descent(X_train, y_train)  # fitting
        if self.verbose:
            table = PrettyTable(['FIELD', 'VALUE'])
            table.align['FIELD'] = 'l'
            table.add_row(['epochs', self.n_epochs])
            table.add_row(['RMSE', round(((self.errors ** 2) ** 0.5).mean())])
            table.add_row(['bias', round(self.b, 4)])
            for idx, w_i in enumerate(self.w):
                table.add_row(['feat {}'.format(idx+1), round(w_i, 4)])
            print(table)
        return self

    def predict(self, X):
        return np.asarray([self._calc_y_pred(x) for x in X]).flatten()
