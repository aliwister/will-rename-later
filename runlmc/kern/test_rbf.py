# Copyright (c) 2016, Vladimir Feinberg
# Licensed under the BSD 3-clause license (see LICENSE)

import unittest
import math

import numdifftools as nd
import numpy as np

from .rbf import RBF
from ..util.numpy_convenience import map_entries, smallest_eig
from ..util.testing_utils import BasicModel, check_np_lists


class RBFTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.cases = np.arange(10.0)
        self.inv_lengthscale = 4
        self.testK = RBF(self.inv_lengthscale, 'wierd-name')

    def f(self, r):
        exp = math.exp(-0.5 * self.inv_lengthscale * r * r)
        return exp

    def df_dl(self, r):
        exp = math.exp(-0.5 * self.inv_lengthscale * r * r)
        return exp * -0.5 * r * r

    def test_defaults(self):
        k = RBF()
        self.assertEqual(k.inv_lengthscale, 1)
        self.assertEqual(k.name, 'rbf')

    def test_values(self):
        actual = self.testK.from_dist(self.cases)
        expected = map_entries(self.f, self.cases)
        np.testing.assert_almost_equal(actual, expected)

    def test_gradients(self):
        actual = self.testK.kernel_gradient(self.cases)
        expected = [map_entries(self.df_dl, self.cases)]
        check_np_lists(actual, expected)

    def test_numerical_gradients(self):
        actual = self.testK.kernel_gradient(self.cases)

        def deriv(x):
            return nd.Derivative(lambda l: RBF(l).from_dist(x))
        expected = [deriv(x)(self.inv_lengthscale)[0] for x in self.cases]
        check_np_lists(actual, [expected])

    def test_to_gpy(self):
        gpy = self.testK.to_gpy()
        self.assertEqual(gpy.name, self.testK.name)
        self.assertEqual(float(gpy.variance[0]), 1)
        self.assertEqual(gpy.input_dim, 1)
        self.assertAlmostEqual(gpy.lengthscale[0],
                               self.inv_lengthscale ** -0.5)
        self.assertEqual(gpy.ARD, False)
        self.assertEqual(gpy.active_dims, [0])

    def test_psd(self):
        top = self.testK.from_dist(self.cases)
        self.assertGreaterEqual(smallest_eig(top), 0)

    def test_optimization(self):
        dists = np.arange(10)
        Y = np.sin(np.arange(10))
        m = BasicModel(dists, Y, self.testK)
        ll_before = m.log_likelihood()
        m.optimize('lbfgsb')
        self.assertGreaterEqual(m.log_likelihood(), ll_before)

    def test_optimization_priors_one_step(self):
        # TODO(priors): check that priors propogate
        pass
