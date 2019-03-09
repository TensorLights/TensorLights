#!/usr/bin/python

import mock
import unittest

import tc_agent


class TestTCAgent(unittest.TestCase):

    def setUp(self):
        self.num_workers = 20
        self.default_setups = {
            'addresses': {'ps': ['ps.ip.0.0:2000'],
                          'worker': ['wk.ip.0.{ip}:{p}'.format(ip=w, p=3000 + w)
                                     for w in range(0, self.num_workers)]},
            'num_concurrent_jobs': 21,
            'should_colocate_ps_worker': False,
            'concurrent_ps_configs': [
                {'ps_port': 2001 + i, 'wk_port': 3001 + i, 'model': 'test_model'}
                for i in range(0, 21)
            ]
        }

    def test_default(self):
        setups = self.default_setups
        setups['poison'] = 'FIFO'
        with mock.patch('tc_agent.exe_local') as exe_call:
            tc_agent.setup_tc(setups)
            self.assertEqual(exe_call.call_count, 0)

    def test_multiple_jobs_per_prio_band_same_model(self):
        setups = self.default_setups
        setups['poison'] = 'TLsOne'
        with mock.patch('tc_agent.exe_local') as exe_call:
            tc_agent.setup_tc(setups)
            exe_call.assert_called_once_with('TLsOne', mock.ANY)
            [self.assertIn('prio 7', a[0][1]) for a in exe_call.call_args_list]
            [self.assertNotIn('prio 8', a[0][1])
             for a in exe_call.call_args_list]

    def test_multiple_jobs_per_prio_band_diff_model(self):
        setups = self.default_setups
        setups['poison'] = 'TLsOne'
        models_to_use = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']
        setups['concurrent_ps_configs'] = [
            {'ps_port': 2001 + i, 'wk_port': 3001 + i,
             'model': models_to_use[i % len(models_to_use)]}
            for i in range(0, setups['num_concurrent_jobs'] )
        ]
        with mock.patch('tc_agent.exe_local') as exe_call:
            tc_agent.setup_tc(setups, rotate_count=0)
            exe_call.assert_called_once_with('TLsOne', mock.ANY)
            [self.assertIn('prio 7', a[0][1]) for a in exe_call.call_args_list]
            [self.assertNotIn('prio 8', a[0][1])
             for a in exe_call.call_args_list]

        with mock.patch('tc_agent.exe_local') as exe_call:
            tc_agent.setup_tc(setups, rotate_count=3)
            exe_call.assert_called_once_with('TLsOne', mock.ANY)
            [self.assertIn('prio 7', a[0][1]) for a in exe_call.call_args_list]
            [self.assertNotIn('prio 8', a[0][1])
             for a in exe_call.call_args_list]

if __name__ == '__main__':
    unittest.main()
