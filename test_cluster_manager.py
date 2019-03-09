#!/usr/bin/env python

from collections import Counter
import unittest

from cluster_manager import *


class TestOverride(unittest.TestCase):

    def setUp(self):
        self.default_options = {
            'num_workers': [20],
            'num_concurrent_jobs': [24],
            'should_colocate_ps_worker': [False],
            'args.target_global_step': [30000],
        }

    def tearDown(self):
        pass

    def test_seperate_ps_worker(self):
        options = self.default_options
        options['should_colocate_ps_worker'] = [False]
        overrides = get_overrides(options)
        for configs in overrides:
            self.assertEqual(configs['ps_idx'], [kINVENTORY[0]])
            self.assertEqual(set(configs['worker_idx']),
                             set(kINVENTORY[1:1 + configs['num_workers']]))

    def test_should_colocate_workers(self):
        options = self.default_options
        options['num_concurrent_jobs'] = [1]
        options['should_colocate_workers'] = [True]
        options['args.model'] = ['resnet32', 'resnet56', 'resnet110']
        options['args.cross_replica_sync'] = [True, False]
        overrides = get_overrides(options)
        expected_ps_idx = 0
        for configs in overrides:
            self.assertEqual(configs['ps_idx'], [kINVENTORY[expected_ps_idx]])
            self.assertEqual(configs['num_workers'],
                             len(configs['worker_idx']))
            self.assertEqual(set(configs['worker_idx']),
                             set([kINVENTORY[1 + expected_ps_idx]]))
            expected_ps_idx += 2

    def test_should_shift_ps(self):
        options = self.default_options
        options['num_concurrent_jobs'] = [21]
        options['should_colocate_ps_worker'] = [False]
        options['concurrent_ps'] = ['con_ps_shift']
        overrides = get_overrides(options)
        expected_ps_idx = 0
        for configs in overrides:
            self.assertEqual(configs['ps_idx'], [kINVENTORY[expected_ps_idx]])
            inventory = kINVENTORY[expected_ps_idx:] + \
                kINVENTORY[:expected_ps_idx]
            self.assertEqual(set(configs['worker_idx']),
                             set(inventory[1:1 + configs['num_workers']]))
            expected_ps_idx = (expected_ps_idx + 1) % len(kINVENTORY)

    def test_concurrent_ps_distribution(self):
        options = self.default_options
        options['num_concurrent_jobs'] = [21]
        options['should_colocate_ps_worker'] = [False]
        options['concurrent_ps'] = ['see_concurrent_ps_distribution']

        for concurrent_ps_distribution in [
                '21',
                '5, 16',
                '10, 11',
                '7, 7, 7',
                '5, 5, 5, 6',
                '4, 4, 4, 4, 5',
                '3, 3, 3, 3, 3, 3, 3',
                '1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1', ]:
            options['concurrent_ps_distribution'] = [
                concurrent_ps_distribution]
            overrides = get_overrides(options)
            counts = Counter([','.join(map(str, configs['ps_idx']))
                              for configs in overrides])
            self.assertEqual(sorted(counts.values()),
                             map(int, concurrent_ps_distribution.split(',')))


if __name__ == '__main__':
    unittest.main()
