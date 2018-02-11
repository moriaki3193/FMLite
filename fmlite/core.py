# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from itertools import combinations


class FMLite:
    """2-way Factorization Machine
    """

    def __init__(
            self,
            k=10,
            task='regression',
            mode='normal',
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
        self.k = k
        self.mode = mode
        self.verbose = verbose
        self.b = None  # set when self.fit is called
        self.w = None  # set when self.fit is called
        self.V = None  # set when self.fit is called

    def _calc_y_pred(self, x):
        """1つの訓練事例に対する現時点での応答変数の推定値を返す.

        param
        -----
        x : np.array, shape = (n_features, )
        """
        interaction_sum = np.array([
            np.dot(self.V[i], self.V[j].T) * x[i] * x[j]
            for i, j in combinations(x.shape[0], 2)
        ]).sum()
        return self.b + np.dot(self.w, x.T) + interaction_sum

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
        squared_errors : np.array, shape = (n_samples * n_epochs, )
        """
        # [START Vの(i, f)成分でy_predを偏微分した値を返す関数の定義]
        # TODO リファクタリング
        def _partial_V(x):
            _V = np.empty((n_features, self.k))
            for f in range(self.k):
                first_term = np.dot(x, np.asarray(self.V[:, f]).ravel().T)
                second_term = self.V[:, f] * (x ** 2)
                _v = first_term + second_term
                _V[:, f] = _v
            return _V
        # [END Vの(i, f)成分でy_predを偏微分した値を返す関数の定義]

        # [START MLSに対する偏微分の値を返す関数の定義]
        def _evaluate_grad(x, diff):
            b_grad = 2 * diff
            w_grad = 2 * diff * x  # using bloadcasting
            V_grad = 2 * diff * _partial_V(x)  # using bloadcasting
            return (b_grad, w_grad, V_grad)
        # [END MLSに対する偏微分の値を返す関数の定義]
        squared_errors = []
        n_epochs = 100  # エポック数
        n_samples, n_features = X_train.shape
        normal_params = {'scale': 0.01, 'size': (n_features, self.k)}
        self.b = 0.0
        self.w = np.zeros(n_features)
        self.V = np.asmatrix(np.random.normal(**normal_params))
        # [START parameter estimation]
        for _ in tqdm(range(n_epochs)):
            for ind in np.random.permutation(n_samples):
                x = np.asarray(X_train[ind]).ravel()
                y = y_train[ind]
                y_tmp_pred = self._calc_y_pred(x)
                diff = y_tmp_pred - y
                squared_errors.append(diff ** 2)
                b_grad, w_grad, V_grad = _evaluate_grad(x, diff)
                # [START updating parameters]
                self.b = self.b - self.learning_rate * b_grad
                self.w = self.w - self.learning_rate * w_grad
                self.V = self.V - self.learning_rate * V_grad
                # [END updating parameters]
        # [END parameter estimation]
        return np.asarray(squared_errors)

    def fit(self, X_train, y_train):
        """Fitting using SGD methods.
        """
        if X_train.ndim != 2:
            raise ValueError('ndim of X must be 2.')
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('n_samples of X and y must be equal.')
        squared_errors = self._gradient_descent(X_train, y_train)  # fitting
        if self.verbose:
            # squared_errors を利用した学習過程の表示
            print(squared_errors)
        return self

    def predict(self, X):
        return np.array([self._calc_y_pred(x) for x in X])
