import unittest
import numpy as np

from tresearcher.researcher.records import *

class TestResults(unittest.TestCase):
    def setUp(self):
        self.res = load_experiment("tresearcher/researcher/data/", "example_record.json").results
        self.res_multi_epoch = load_experiment("tresearcher/researcher/data/", "example_epoch_record.json").results
    
    def test_correctly_gathers_metric(self):
        mses = self.res.get_metric("mse")   

        self.assertEqual(len(mses), 5)
        self.assertEqual(len(mses[0]), 1)
        self.assertEqual(len(mses[1]), 1)
        self.assertEqual(len(mses[2]), 1)
        self.assertEqual(len(mses[3]), 1)
        self.assertEqual(len(mses[4]), 1)

        mses = self.res_multi_epoch.get_metric("mse")   

        self.assertEqual(len(mses), 5)
        self.assertEqual(len(mses[0]), 2)
        self.assertEqual(len(mses[1]), 2)
        self.assertEqual(len(mses[2]), 2)
        self.assertEqual(len(mses[3]), 2)
        self.assertEqual(len(mses[4]), 2)

    def test_correctly_gathers_aggregated_metric(self):
        agg_mse = self.res.get_fold_aggregated_metric("mse", np.mean)   
        self.assertEqual(len(agg_mse), 1)
        self.assertEqual(agg_mse[0], 23481789543.946037)

        agg_mse = self.res.get_fold_aggregated_metric("mse", np.max)   
        self.assertEqual(len(agg_mse), 1)
        self.assertAlmostEqual(agg_mse[0], 65922637417.14631)

        agg_mse = self.res_multi_epoch.get_fold_aggregated_metric("mse", np.mean)   
        self.assertEqual(len(agg_mse), 2)
        self.assertAlmostEqual(agg_mse[0], 0.584)
        self.assertAlmostEqual(agg_mse[1], 0.382)