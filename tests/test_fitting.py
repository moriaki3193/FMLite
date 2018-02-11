# -*- coding: utf-8 -*-
import unittest
import numpy as np
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
        fm = FMLite(n_epochs=10)
        model = fm.fit(self.features[:6, ], self.target[:6])
        self.assertEqual(len(model.errors), 6 * model.n_epochs)

    def test_does_model_fit_combination_dependently(self):
        fm = FMLite(
                n_epochs=10,
                mode='combination-dependent',
                n_entities=4,
                n_features=8)
        model = fm.fit(self.features_cb[:6, ], self.target[:6])
        print(model.V)


if __name__ == '__main__':
    unittest.main()
