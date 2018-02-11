# -*- coding: utf-8 -*-
import unittest
import numpy as np
from fmlite.core import FMLite


class TestFMLitePrediction(unittest.TestCase):

    def setUp(self):
        fm = FMLite(n_epochs=10)
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
        self.target = np.array([5, 3, 1, 4, 5, 1, 5])
        self.model = fm.fit(self.features[:6, :], self.target[:6])

    def test_does_model_predict(self):
        tar_pred = self.model.predict(self.features[6:])
        self.assertTrue(isinstance(tar_pred, np.ndarray))
        self.assertEqual(tar_pred.shape, self.target[6:].shape)
