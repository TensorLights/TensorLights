#!/usr/bin/python

# self defined headers

from global_variables import *
from job_launcher import *
from utils import *

import controller
import global_variables as gv

# standard libraries
import itertools  # to produce value products of options dict
import optparse  # command line parser
import os  # get version code of tensorflow used for experiment
import random  # used to shuffle the worker machines
import sys

from functools import partial


def get_shifted_inventory_from_concurrent_ps_distribution(overrides, kNumPS, inventory, run_idx):
    if 'concurrent_ps' not in overrides or overrides['concurrent_ps'] == 'con_ps_same':
        return inventory

    if overrides['concurrent_ps'] == 'con_ps_shift':
        shift_by = (kNumPS * run_idx) % len(inventory)

    if overrides['concurrent_ps'] == 'see_concurrent_ps_distribution' \
            and overrides['concurrent_ps_distribution'] == '0':
        # all workers in the same job run on the same machine
        shift_by = (kNumPS * run_idx) % len(inventory)

    if overrides['concurrent_ps'] == 'see_concurrent_ps_distribution'\
            and overrides['concurrent_ps_distribution'] != '0':
        ps_distribution = map(int, overrides[
            'concurrent_ps_distribution'].split(','))
        assert(sum(ps_distribution) == overrides['num_concurrent_jobs'])
        shift_by_from_run_idx = list(itertools.chain.from_iterable(
            [[shift_by] * num_runs for shift_by, num_runs in enumerate(ps_distribution)]))
        shift_by = (kNumPS * shift_by_from_run_idx[run_idx]) % len(inventory)

    inventory_after_shift = inventory[shift_by:] + inventory[:shift_by]

    return inventory_after_shift


def shuffle_worker_idx_if_needed(overrides):
    # Shuffle worker index to evenly distribute the chief worker from
    # concurrent jobs. Note that there is no need to shuffle worker when
    # concurrent_ps == 'shift', bacause the chief worker has been shifted
    # together with the ps.
    # If not specified in overrides, 'shuffle_worker' is 'shift' by default.

    if 'concurrent_ps' in overrides and overrides['concurrent_ps'] == 'con_ps_shift':
        return overrides['worker_idx']

    if 'shuffle_worker' not in overrides or overrides['shuffle_worker'] == 'shfl_wk_shift':
        # 'shuffle_worker' is 'shift' by default.
        shift_by = overrides['0run'] - overrides['base_run_idx']
        shift_by %= len(overrides['worker_idx'])
        worker_idx = overrides['worker_idx']
        return worker_idx[-shift_by:] + worker_idx[:-shift_by]

    if overrides['shuffle_worker'] == 'shfl_wk_same':
        return overrides['worker_idx']

    if overrides['shuffle_worker'] == 'shfl_wk_rand':
        rseed = overrides['0run'] - overrides['base_run_idx']
        random.seed(rseed)
        worker_idx = overrides['worker_idx']
        random.shuffle(worker_idx)
        overrides['worker_idx'] = worker_idx
        return worker_idx

    return overrides['worker_idx']


def get_overrides(options):
    gv.NEW_STAGE_BEGIN_USEC = now_microsec()

    global kINVENTORY

    if gv.DEBUG_LEVEL > 0:
        gv.DB_TO_USE = 'learning_debug'
        options['num_workers'] = [2]
        options['num_concurrent_jobs'] = [1]
        options['concurrent_ps_distribution'] = ['1']
        options['args.target_global_step'] = [20]
        options['args.batch_size'] = [4]
        options['args.eval_calculate_accuracy'] = [False]
        options['args.cross_replica_sync'] = [True]
        options['should_colocate_ps_worker'] = [False]

    if 'concurrent_ps' in options \
            and len(options['concurrent_ps']) == 1 \
            and options['concurrent_ps'][0] == 'see_concurrent_ps_distribution' \
            and len(options['concurrent_ps_distribution']) == 1:

        if options['concurrent_ps_distribution'][0] == '0':
            # all workers in the same job run on the same machine
            options['should_colocate_ps_worker'] = [True]
            options['should_colocate_workers'] = [True]
        if options['concurrent_ps_distribution'][0] == '1':
            options['num_concurrent_jobs'] = [1]
            options['should_colocate_ps_worker'] = [False]
            options['should_colocate_workers'] = [False]
            # options['num_threads'] = [6]

    assert('num_workers' in options)
    assert('num_concurrent_jobs' in options)
    assert('should_colocate_ps_worker' in options)

    product = [x for x in apply(itertools.product, options.values())]
    override_list = [dict(zip(options.keys(), p)) for p in product]

    override_list = sorted(override_list, key=lambda i: (
        i['num_workers'], i['num_concurrent_jobs']))

    inventory = kINVENTORY

    # Give warming about over-commiting CPU
    max_num_concurrent_jobs = max(
        [d['num_concurrent_jobs'] for d in override_list])
    if MAX_NUM_THREADS_PER_MACHINE < 2 * max_num_concurrent_jobs:
        print('{}overrides: CPU contention is possible: num of threads per machine ({}) '
              '< threads per job (2) * max num of concurrent jobs per machine '
              '({}){}').format(Format.PINK, MAX_NUM_THREADS_PER_MACHINE,
                               max_num_concurrent_jobs, Format.END)

    # if ps is NOT colocating with the chief worker, we need to
    # 1. offset all worker by wk_offset (the ps host), and
    # 2. account more hosts for ps
    # wk_offsets = how many host PSes take up exclusively in each job
    # 3. if we ask all workers to be place on the same machine, then reset
    #    num_workers_list to adjust the ps bases

    num_ps_list = [0 if ('should_colocate_ps_worker' in d
                         and d['should_colocate_ps_worker'])
                   else kNumPS for d in override_list]
    num_workers_list = [1 if ('should_colocate_workers' in d
                              and d['should_colocate_workers'])
                        else d['num_workers'] for d in override_list]
    ps_bases = cum_sum([p + w for p, w in zip(num_ps_list,  num_workers_list)])

    # One stage of jobs are generated using the template below. In this mode,
    # we concurrently launch multiple runs of a job. Job configurations, unless
    # overridden in the for loop below, are exactly the same for each run. '0run'
    # differenciates different runs. Each run comes at a different time specified
    # by launch_interval_sec, which can be set to very small to mimic two jobs
    # launching at the same time. launch_interval_sec should not be zero, so as
    # to avoid temperary high load on rpc/ssh to launch the job. tc_setup() also
    # relay on launch_interval_sec, so that only the first arrival job will
    # setup tc rules for all jobs to reuse.
    experiments_overlap_runs = []
    for overrides_, ps_base_, wk_offset_ in zip(override_list, ps_bases, num_ps_list):
        should_colocate_workers_ = 'should_colocate_workers' in overrides_ \
            and overrides_['should_colocate_workers']
        num_concurrent_jobs_ = overrides_['num_concurrent_jobs']
        num_wk_ = overrides_['num_workers']
        target_global_step_ = overrides_['args.target_global_step']
        base_run_idx_ = overrides_[
            'base_run_idx'] if 'base_run_idx' in overrides_ else 100
        assert(len(inventory) >= ((ps_base_ + wk_offset_) if should_colocate_workers_
                                  else (ps_base_ + wk_offset_ + num_wk_)))
        for run_idx_ in range(0, num_concurrent_jobs_):
            # how many re-occurring runs per job.
            inventory_after_shift = get_shifted_inventory_from_concurrent_ps_distribution(
                overrides_, kNumPS, inventory, run_idx_)
            expd = {'ps_idx': inventory_after_shift[ps_base_: ps_base_ + kNumPS],
                    'worker_idx': inventory_after_shift[ps_base_ + wk_offset_:
                                                        ps_base_ + wk_offset_ + num_wk_]
                    if not should_colocate_workers_ else [inventory_after_shift[ps_base_ + wk_offset_]] * num_wk_,
                    'launch_interval_sec': run_idx_ / 10.0,
                    'base_run_idx': base_run_idx_,
                    '0run': base_run_idx_ + run_idx_,
                    'num_concurrent_jobs': num_concurrent_jobs_,
                    'args.target_global_step': target_global_step_,
                    'args.num_batches': 1 + target_global_step_ / num_wk_,
                    }
            expd.update(overrides_)
            expd['worker_idx'] = shuffle_worker_idx_if_needed(expd)
            if 'shift_models_per_run' in overrides_:
                models = get_models_for_dataset(
                    overrides_['shift_models_per_run'])
                assert(len(models) > 0)
                expd['args.model'] = models[run_idx_ % len(models)]
            experiments_overlap_runs.append(expd)

    # add a unique job id for each job
    for job_id, expd in enumerate(experiments_overlap_runs):
        expd['job_id'] = job_id

    print 'overrides: Jobs to run concurrently ' + '-' * 60
    display = sorted(experiments_overlap_runs,
                     key=lambda item: (item['ps_idx'][0], item['0run']))
    print ',\t'.join(display[0].keys())  # header
    for e in display:
        print '\t'.join([str(v) for k, v in e.iteritems()])
    return experiments_overlap_runs


def clean(ip):
    command = ('killall screen; rm -r {}/logs/*').format(gv.TF_HOME)
    exe_ssh(command, ip, 'clean',
            errors_to_ignore=['screen: no process found',
                              'Operation not permitted',
                              'No such file or directory'])

    exe_ssh('killall screen', ip, 'kill_screen_root', use_root=True,
            errors_to_ignore=['screen: no process found'])


def run_one_stage(experiments):
    if MAX_NUM_PROCESS_CLUSTER_MANAGER < len(experiments):
        print('{}Not enough processes to monitor concurrent jobs. Expected '
              'one process per job, but found {} processes and {} jobs. {}'
              ).format(Format.PINK, MAX_NUM_PROCESS_CLUSTER_MANAGER,
                       len(experiments), Format.END)
        exit(-1)

    print '{fc}[{nt}] {fb}Going to run ( {m} ) experiments concurrently... {fe}'.format(
        fc=Format.DARKCYAN, nt=now(), fb=Format.BOLD, m=len(experiments), fe=Format.END)

    if not gv.COLLECT_RESULTS_ONLY:
        ips = []
        for setups in experiments:
            for ip_port in setups['addresses']['ps'] + setups['addresses']['worker']:
                ips.append(ip_port.split(':')[0])
        ips = set(ips)
        [clean(ip) for ip in ips]
        print 'Clean up machines at {}'.format(ips)

    # All experiments and the one control process
    num_processes_to_wait = len(experiments) + 1
    pre_train_barrier = Barrier(num_processes_to_wait)
    post_train_barrier = Barrier(num_processes_to_wait)

    # Here is the real work.
    pool = Pool(processes=num_processes_to_wait)

    should_exit = False
    try:
        control_func = partial(
            controller.control_and_monitor, pre_train_barrier, post_train_barrier)
        pool.apply_async(control_func, (experiments, ))

        launchers = [Launcher(setups) for setups in experiments]
        launch_func = partial(
            launch_one_run, pre_train_barrier, post_train_barrier)
        pool.map(launch_func, launchers)

    except KeyboardInterrupt:
        print('main: Caught Ctrl-C. Will exit after all workers '
              'finish cleanup and will not run more experiments.')
        should_exit = True  # Here is a global exit.
    finally:
        pool.close()  # no more task is allowed in the pool
        pool.join()

    if should_exit:
        exit(-1)

    print '{}[{}] {}Done with ( {} ) concurrent experiments! {}\n\n\n'.format(
        Format.BLUE, now(), Format.BOLD, len(experiments), Format.END)

    if gv.DEBUG_LEVEL > 0:
        exit(0)

    if gv.COLLECT_RESULTS_ONLY:
        exit(0)

if __name__ == "__main__":

    parser = optparse.OptionParser(usage='run_me.py [options]')

    parser.add_option('--collect_results', action='store_true',
                      help=('If specified, no experiment will be run '
                            'and we only collect results'))
    parser.set_defaults(collect_results=False)

    parser.add_option('-m', '--mode',
                      help='Mode to run. Choices are overlap_run')
    parser.add_option('-n', '--name',
                      help='Name of experiment to run')
    parser.add_option('-d', '--debug_level', type='int',
                      help='Debug level higher for more debugging')
    parser.add_option('-r', '--num_runs',
                      help='Number of runs')

    parser.add_option('--sync', dest='sync', action='store_true',
                      help=('If specified, use sync training. '
                            'Otherwise, default is async'))
    parser.add_option('--async', dest='sync', action='store_false')
    parser.set_defaults(sync=False)

    parser.add_option('--profile_barrier',
                      dest='profile_barrier', action='store_true')
    parser.set_defaults(profile_barrier=False)
    parser.add_option('--profile_vmstat',
                      dest='profile_vmstat', action='store_true')
    parser.set_defaults(profile_vmstat=False)
    parser.add_option('--profile_ifstat',
                      dest='profile_ifstat', action='store_true')
    parser.set_defaults(profile_ifstat=False)
    parser.add_option('--profile_tcpdump',
                      dest='profile_tcpdump', action='store_true')
    parser.set_defaults(profile_tcpdump=False)

    parser.add_option('--shuffle_worker',
                      help=('how to shuffle the worker index w.r.t. machine '
                            'index for a job. Choices are shift, random, none.'))
    parser.set_defaults(shuffle_worker='shfl_wk_shift')

    parser.add_option('--concurrent_ps',
                      help=('When there are multiple concurrent jobs, on which '
                            'machine to place the PSes. Choices are same, shift.'))
    parser.set_defaults(concurrent_ps='con_ps_same')

    (opts, args) = parser.parse_args()

    if opts.debug_level:
        gv.DEBUG_LEVEL = opts.debug_level

    if opts.collect_results:
        gv.COLLECT_RESULTS_ONLY = True

    num_runs = int(opts.num_runs or -1)
    mode = opts.mode
    start_ts = now()

    basic = config_template[opts.name].copy()
    basic['should_vmstat'] = opts.profile_vmstat
    basic['should_ifstat'] = opts.profile_ifstat
    basic['should_tcpdump'] = opts.profile_tcpdump
    basic['should_barrier'] = opts.profile_barrier
    basic['shuffle_worker'] = opts.shuffle_worker
    basic['concurrent_ps'] = opts.concurrent_ps

    if mode == 'jct_measurement':
        for concurrent_ps_distribution in [
            '21',
            '5, 16',
            '10, 11',
            '7, 7, 7',
            '5, 5, 5, 6',
            '4, 4, 4, 4, 5',
            '3, 3, 3, 3, 3, 3, 3',
            '1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1',
        ]:
            for policy in ['FIFO', 'TLsRR', 'TLsOne', ]:
                basic['poison'] = policy
                basic['args']['batch_size'] = 4
                basic['args']['model'] = 'resnet32'
                basic['args']['cross_replica_sync'] = True
                basic['args']['eval_calculate_accuracy'] = False
                basic['num_threads'] = 6
                concurrent_runs = get_overrides(
                    options={
                        # 1/2 expanding variables.
                        'num_workers': [20],  # required entry
                        # required entry
                        'num_concurrent_jobs': [21],
                        # 2/2 fixed variables on all experiments.
                        # required entry
                        'args.target_global_step': [30005],
                        # required entrys
                        'should_colocate_ps_worker': [False],
                        'shuffle_worker': ['shfl_wk_rand'],
                        'concurrent_ps': ['see_concurrent_ps_distribution'],
                        'concurrent_ps_distribution': [concurrent_ps_distribution],
                    })
                experiments = [overwrite_setups(
                    basic, overrides) for overrides in concurrent_runs]
                run_one_stage(experiments)
    elif mode == 'sensitivity_batch_size':
        for batch_size in [1, 2, 8, 16]:
            for policy in ['FIFO', 'TLsOne',  'TLsRR', ]:
                basic['poison'] = policy
                basic['args']['batch_size'] = batch_size
                basic['args']['model'] = 'resnet32'
                basic['args']['cross_replica_sync'] = True
                basic['args']['eval_calculate_accuracy'] = False
                basic['num_threads'] = 6
                concurrent_runs = get_overrides(
                    options={
                        # 1/2 expanding variables.
                        # required entry
                        'num_workers': [20],
                        # required entry
                        'num_concurrent_jobs': [21],
                        # 2/2 fixed variables on all experiments.
                        # required entry
                        'args.target_global_step': [30005],
                        # required entrys
                        'should_colocate_ps_worker': [False],
                        'shuffle_worker': ['shfl_wk_rand'],
                        'concurrent_ps': ['see_concurrent_ps_distribution'],
                        'concurrent_ps_distribution': ['21'],
                    })
                experiments = [overwrite_setups(
                    basic, overrides) for overrides in concurrent_runs]
                run_one_stage(experiments)
    elif mode == 'profile_measurement':
        if opts.name == 'cifar10TF':
            for (profile_item, placement) in [
                    ('ifstat',  '21'), ('vmstat', '21',), ('barrier', '21'),
                    ('barrier', '1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1')]:
                # Other choices of profile_item are None and 'tcpdump'.
                policies = (['TLsOne', 'FIFO', 'TLsRR', ]
                            if placement == '21' else ['FIFO'])
                for policy in policies:
                    concurrent_runs = get_overrides(
                        options={
                            # 1/2 expanding variables.
                            'num_workers': [20],  # required entry
                            # required entry
                            'num_concurrent_jobs': [21],
                            # 2/2 fixed variables on all experiments.
                            'args.target_global_step': [30005],
                            'should_colocate_ps_worker': [False],
                            'shuffle_worker': ['shfl_wk_rand'],
                            'concurrent_ps': ['see_concurrent_ps_distribution'],
                            'concurrent_ps_distribution': [placement],
                        })
                    basic_copy = copy.deepcopy(basic)
                    basic_copy['poison'] = policy
                    basic_copy['profile_item'] = profile_item
                    if profile_item:
                        basic_copy['should_' + profile_item] = True
                    basic_copy['args']['batch_size'] = 4
                    basic_copy['args']['model'] = 'resnet32'
                    # 'resnet32', 'resnet56', 'resnet110'
                    basic_copy['args']['cross_replica_sync'] = True
                    basic_copy['args'][
                        'eval_calculate_accuracy'] = False
                    basic_copy['num_threads'] = 6
                    experiments = [overwrite_setups(
                        basic_copy, overrides) for overrides in concurrent_runs]
                    run_one_stage(experiments)
        elif opts.name == 'imagenetTF':
            # Compare cluster performance with that reported online for the
            # TensorFlow benchmarks. Link is at
            # https://www.tensorflow.org/guide/performance/benchmarks#distributed_training_with_nvidia%C2%AE_tesla%C2%AE_k80
            gv.DB_TO_USE = 'learning_imagenet'
            for model in ['inception3', 'resnet50', 'resnet152', ]:
                concurrent_runs = get_overrides(
                    options={
                        # 1/2 expanding variables.
                        'num_workers': [16],  # required entry
                        'num_concurrent_jobs': [1],  # required entry
                        # 2/2 fixed variables on all experiments.
                        'args.target_global_step': [960],
                        'should_colocate_ps_worker': [False],
                    })
                basic['poison'] = 'FIFO'
                basic['args']['batch_size'] = 32 if model == 'resnet152' else 64
                basic['args']['model'] = model
                basic['args']['cross_replica_sync'] = True
                basic['args']['eval_calculate_accuracy'] = False
                basic['num_threads'] = 12
                experiments = [overwrite_setups(
                    basic, overrides) for overrides in concurrent_runs]
                run_one_stage(experiments)
    else:
        print '{}Invalid mode {}{}'.format(Format.PINK, mode, Format.END)

    print '{fb}Done with all experiments! Started at {st}, ended at {et}{fe}\n\n\n'.format(
        fb=Format.BOLD, st=start_ts, et=now(), fe=Format.END)
