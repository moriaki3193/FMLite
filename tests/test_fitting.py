# -*- coding: utf-8 -*-
import unittest
import numpy as np
from itertools import combinations
from fmlite.core import FMLite


class TestFMLiteFitting(unittest.TestCase):

    def setUp(self):
        self.features = np.matrix([
           #  Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
           # A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
            [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
            [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
            [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
            [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
            [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
            [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
            [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
        ])
        features_cb = np.matrix([
           #  Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
           # A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
            [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
            [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
            [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
            [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
            [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
            [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
            [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
        ])
        users = features_cb[:, 0:3]
        features_cb = np.delete(features_cb, 0, 1)
        features_cb = np.delete(features_cb, 1, 1)
        features_cb = np.delete(features_cb, 2, 1)
        self.features_cb = np.hstack((features_cb, users))
        self.target = [5, 3, 1, 4, 5, 1, 5]

    def test_does_model_fit(self):
        fm = FMLite(k=10, n_epochs=100)
        model = fm.fit(self.features[:6, ], self.target[:6])
        self.assertEqual(len(model.errors), 6 * model.n_epochs)

    def test_does_model_fit_combination_dependently(self):
        fm = FMLite(
                k=10,
                C=1000000,
                n_epochs=100,
                mode='combination-dependent',
                n_entities=4,
                n_features=8)
        model = fm.fit(self.features_cb[:6, ], self.target[:6])
        # [START 制約が効いているかの確認]
        V = model.V
        M = model.n_entities
        n = model.n_features
        UU = []
        UV = []
        VV = []
        for i, j in combinations(range(2 * M), 2):
            w_pred = np.dot(V[i], V[j].T)
            if 0 <= i < M and 0 <= j < M:
                UU.append(w_pred)
            elif 0 <= i < M and M <= j < 2 * M:
                UV.append(w_pred)
            elif M <= i < 2 * M and M <= j < 2 * M:
                VV.append(w_pred)
        print('要素数: {}, {}'.format(len(UU), np.asarray(UU).mean()))
        for w_pred in UU:
            print(w_pred)
        print('要素数: {}, {}'.format(len(UV), np.asarray(UV).mean()))
        for w_pred in UV:
            print(w_pred)
        print('要素数: {}, {}'.format(len(VV), np.asarray(VV).mean()))
        for w_pred in VV:
            print(w_pred)
        # [END 制約が効いているかの確認]


if __name__ == '__main__':
    unittest.main()
