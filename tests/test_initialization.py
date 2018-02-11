# -*- coding: utf-8 -*-
import unittest
from fmlite.core import FMLite


class TestFMLiteInitialization(unittest.TestCase):

    def test_initialization(self):
        self.assertRaises(ValueError, lambda: FMLite(mode='foo'))
        self.assertRaises(
            ValueError,
            lambda: FMLite(mode='combination-dependent'))


if __name__ == '__main__':
    unittest.main()
