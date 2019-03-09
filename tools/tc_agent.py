#!/usr/bin/python

import datetime
import itertools
import optparse  # command line parser
import pickle
import math
import subprocess as sp

from time import sleep

MODEL_ORDER = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
               'densenet40_k12', 'densenet100_k12', 'densenet100_k24',
               'vgg11', 'vgg16', 'vgg19',
               'resnet50', 'resnet101', 'resnet152',
               'googlenet', 'inception3', 'inception4', ]

DEV_TO_USE = 'eth2'


def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def cum_sum(l):
    total = 0
    for x in l:
        yield total
        total += x


def exe_local(caller_name, command, errors_to_ignore=[]):
    exe = sp.Popen(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    has_error = False
    for l in exe.stderr:
        if any([error_snippet in l for error_snippet in errors_to_ignore]):
            continue
        has_error = True
        print '{caller}(); {msg}'.format(caller=caller_name, msg=l)
    return has_error, exe.stdout.readlines()


def reset_dev_local():
    command = ('tc qdisc del dev {dev} root; '
               'tc qdisc del dev {dev} ingress; ').format(dev=DEV_TO_USE)

    exe_local('reset_dev_local', command, errors_to_ignore=[
        'RTNETLINK answers: No such file or directory',
        'RTNETLINK answers: Invalid argument'])


def set_saturate_prio(caller_policy, saturate_bw, prio_configs,
                      dir_ps_to_wk=True, dir_wk_to_ps=True):
    assert(dir_ps_to_wk or dir_wk_to_ps)
    min_bw = '100kbit'
    command = ('tc qdisc add dev {dev} root handle 1: htb default 10; '
               'tc class add dev {dev} parent 1: classid 1:1 htb rate {sbw}; '
               'tc class add dev {dev} parent 1:1 classid 1:10 '
               '    htb rate {mbw} ceil {sbw} prio 7; '
               ).format(dev=DEV_TO_USE, mbw=min_bw, sbw=saturate_bw)

    for pidx, prio_config in enumerate(prio_configs):
        ps_port = prio_config['ps_port']
        wk_port = prio_config['wk_port']
        prio = prio_config['prio']
        fid = 11 + pidx
        prio_config['fid'] = fid
        command += ('tc class add dev {dev} parent 1:1 classid 1:{fid} '
                    '   htb rate {mbw} ceil {sbw} prio {prio}; '
                    ).format(dev=DEV_TO_USE, fid=fid, mbw=min_bw,
                             sbw=saturate_bw, prio=prio)
        if dir_ps_to_wk:
            command += (
                # control ps->wk traffic by setting sport at ps
                'tc filter add dev {dev} protocol ip parent 1: prio {prio} u32 '
                '   match ip sport {psport} 0xffff flowid 1:{fid}; '
                'tc filter add dev {dev} protocol ip parent 1: prio {prio} u32 '
                '   match ip dport {wkport} 0xffff flowid 1:{fid}; '
            ).format(dev=DEV_TO_USE,  prio=prio, fid=fid,
                     psport=ps_port, wkport=wk_port)
        if dir_wk_to_ps:
            command += (
                # control wk->ps traffic by setting dport at wk
                'tc filter add dev {dev} protocol ip parent 1: prio {prio} u32 '
                '   match ip dport {psport} 0xffff flowid 1:{fid}; '
                'tc filter add dev {dev} protocol ip parent 1: prio {prio} u32 '
                '   match ip sport {wkport} 0xffff flowid 1:{fid}; '
            ).format(dev=DEV_TO_USE,  prio=prio, fid=fid,
                     psport=ps_port, wkport=wk_port)
    exe_local(caller_policy, command)
    print '[{}] Poison with {} @ {} '.format(now(), caller_policy, DEV_TO_USE)
    print str(prio_configs).replace('}, ', '},\n')
    # print str(command).replace('; ', ';\n')


def setup_tc(tc_configs, rotate_count=0):

    # Please note: TensorFlow is smart enough to eliminate network
    # communication if sender/receiver are in the same machine

    if 'FIFO' in tc_configs['poison']:
        return

    if tc_configs['should_colocate_ps_worker'] and len(tc_configs['addresses']['worker']) <= 1:
        # when in single worker mode, and colocating ps and the single worker
        # poison is useless since both ps and the single worker are on the
        # same machine and no tx happens.
        return

    # Ideally, we want this many (priority class) priorities. But as the actual
    # number of priority band is limited, we later will need to ask multiple
    # priority class to share one piority band.
    link_rate = '10Gbit'
    if 'link_rate' in tc_configs:
        link_rate = tc_configs['link_rate']
    # now choose the appropriate tc setup function
    if any([p in tc_configs['poison'] for p in ['TLsRR', 'TLsOne']]):
        prio_configs = tc_configs['concurrent_ps_configs']
        models = [cps['model'] for cps in prio_configs]
        if 1 == len(set(models)):
            # make sure only 0-6 prio bands are used
            num_prio_classes = len(prio_configs)
            num_jobs_each_prio_level = num_prio_classes / 7 + (
                1 if num_prio_classes % 7 != 0 else 0)
            prios = range(0, num_prio_classes)
            shift_by = rotate_count % len(prios)
            prios = prios[shift_by:] + prios[:shift_by]
            for p, cps in zip(prios, prio_configs):
                cps['prio'] = p / num_jobs_each_prio_level
            set_saturate_prio(tc_configs['poison'], link_rate, prio_configs,
                              dir_ps_to_wk=True, dir_wk_to_ps=False)
        else:
            unique_models = sorted(
                set(models), key=lambda m: MODEL_ORDER.index(m))
            model_count = [models.count(m) for m in unique_models]
            max_prio = 6
            num_prio = [max(1, int(math.floor(
                float(1 + max_prio) * mc) / sum(model_count)))
                for mc in model_count]
            model_to_num_prio_band = {
                model: num_prio for model, num_prio
                in zip(unique_models, num_prio)}
            model_to_starting_prio = {
                model: starting_prio for model, starting_prio
                in zip(unique_models, cum_sum(num_prio))}
            for model, group in itertools.groupby(
                    prio_configs, key=lambda cps: cps['model']):
                # we need to fit equivalent_runs into num_prio bands
                equivalent_runs = list(group)
                num_prio_band = model_to_num_prio_band[model]
                num_jobs_each_prio_band = len(equivalent_runs) / num_prio_band + (
                    1 if len(equivalent_runs) % num_prio_band != 0 else 0)
                prios = range(0,  len(equivalent_runs))
                shift_by = rotate_count % len(prios)
                prios = prios[shift_by:] + prios[:shift_by]
                for p, run in zip(prios, equivalent_runs):
                    run['prio'] = p / num_jobs_each_prio_band + \
                        model_to_starting_prio[model]
            set_saturate_prio(tc_configs['poison'], link_rate, prio_configs,
                              dir_ps_to_wk=True, dir_wk_to_ps=False)
    else:
        print 'setup_tc(): Invalid poison mode [{}].'.format(
            tc_configs['poison'] if 'poison' in tc_configs else '')


def rotate_tc(tc_configs, interval_sec):
    rotate_count = 0
    while True:
        reset_dev_local()
        setup_tc(tc_configs, rotate_count)
        rotate_count += 1
        sleep(interval_sec)

if __name__ == "__main__":

    parser = optparse.OptionParser(usage='tc_agent.py [options]')
    parser.add_option('-d', '--dev', help='Network interface to use')
    parser.set_defaults(concurrent_ps='eth2')

    parser.add_option('-f', '--func', help='Func to call')
    parser.set_defaults(concurrent_ps='setup_tc')

    parser.add_option('-c', '--tc_configs_dir', help='directory to experiment '
                      'configuration file.')

    parser.add_option('--enable_rotate', dest='enable_rotate',
                      action='store_true', help='if specified, enable rotation')
    parser.set_defaults(enable_rotate=False)

    parser.add_option('-r', '--rotate_interval_sec', type='int',
                      help='Interval in seconds to rotate the tc setup')

    parser.set_defaults(rotate_interval_sec=20)

    (opts, args) = parser.parse_args()

    DEV_TO_USE = opts.dev

    if opts.func == 'reset_dev':
        reset_dev_local()
    elif opts.func == 'setup_tc':
        tc_configs = pickle.load(open(opts.tc_configs_dir, 'rb'))
        if opts.enable_rotate:
            rotate_tc(tc_configs, opts.rotate_interval_sec)
        else:
            reset_dev_local()
            setup_tc(tc_configs)

    print 'Done with {} and exit. Bye!'.format(opts.func)
