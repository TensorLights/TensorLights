#!/usr/bin/python

import datetime
import global_variables as gv
import subprocess as sp
import time


def cum_sum(l):
    total = 0
    for x in l:
        yield total
        total += x


def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def now_microsec():
    return int(time.time() * 1e6)


def canonical_task_name(setups, t, idx=None):
    exp = canonical_exp_name(setups)
    return '{}_{}{:02d}'.format(exp, t, idx) if idx != None else '{}_{}'.format(exp, t)


def canonical_exp_name(setups):
    ''' Each job should have a unique name in a batch of experiments.'''
    name = setups['0name']
    num_workers = setups['0num_workers']
    run = setups['0run']
    batch_size = setups['args']['batch_size']
    poison = setups['poison'].replace(':', '')
    ps_host = setups['addresses']['ps'][0].split(':')[0]
    ps_host_idx = ps_host.split('.')[-1]
    sync = 'Sync' if setups['args']['cross_replica_sync'] else ''
    return '{n}w{w:02d}r{r}b{b}{p}{h}{s}'.format(
        n=name, w=num_workers, r=run, b=batch_size, p=poison, h=ps_host_idx, s=sync)


def exe_ssh(command, ip, caller_name, use_root=False, errors_to_ignore=[]):
    ip_suffix = ip.split('.')[-1]
    ip = gv.get_control_ip_port(ip_suffix)
    command_in_ssh = 'ssh {user}{ip} "{command}"'.format(
        user='root@' if use_root else '', ip=ip, command=command)
    ssh = sp.Popen(command_in_ssh, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)

    has_error = False
    connection_lost = False

    connection_lost_msg = ['Connection closed by remote host',
                           'Connection reset by peer', 'Connection timed out']
    for l in ssh.stderr:
        if any([error_snippet in l for error_snippet in connection_lost_msg]):
            connection_lost = True
            continue
        if any([error_snippet in l for error_snippet in errors_to_ignore]):
            continue
        has_error = True
        print '{}{}(); {}; {}{}'.format(
            gv.Format.RED, caller_name, ip, l, gv.Format.END)
    return connection_lost, has_error, ssh.stdout.readlines()


def scp_to_remote(caller_name, remote_ip, local_file_dir, remote_file_dir):
    command = ('scp {} {}:{}').format(
        local_file_dir, remote_ip, remote_file_dir)
    scp = sp.Popen(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    for msg in scp.stderr:
        print '{}{}(); scp to remote @ {}; {}{}'.format(
            gv.Format.RED, caller_name, remote_ip, msg, gv.Format.END)


def kill_by_tasks(tasks, use_root=False):
    def kill_screen(task_name, ip_port, use_root):
        command = ('screen -X -S {t} quit').format(t=task_name)
        exe_ssh(command, ip_port.split(':')[0],
                'kill_screen', use_root=use_root)

    [kill_screen(t, ip_port, use_root) for (t, ip_port) in tasks]
    for task, ip_port in tasks:
        if not check('dead', t, ip_port, use_root=use_root,
                     verbose=gv.DEBUG_LEVEL >= 2):
            print '{}Error: Fail to kill {}. Please check.{}'.format(
                gv.Format.RED, task, gv.Format.END)


def check(alive_or_dead, task_name, ip_port, verbose=False, use_root=False,
          return_true_on_connection_lost=False):
    if alive_or_dead == 'alive':
        num_line_expected_stdout = 1
    elif alive_or_dead == 'dead':
        num_line_expected_stdout = 0
    else:
        print 'check (alive | dead , ...)'
        return
    command = ('screen -list | grep \\"{}\\" ').format(task_name)

    connection_lost, has_error, stdout = exe_ssh(
        command, ip_port.split(':')[0], 'check', use_root=use_root)

    if connection_lost and return_true_on_connection_lost:
        return True

    if has_error:
        return False

    if len(stdout) != num_line_expected_stdout:
        if verbose:
            print('{}Error: expected {} screen under task_name[{}] on {}, '
                  'but found {} {}').format(
                gv.Format.RED, num_line_expected_stdout, task_name, ip_port,
                len(stdout), gv.Format.END)
        return False
    if verbose:
        if len(stdout) == 1:
            print '{}{}Alive{} {} on {}.'.format(
                gv.Format.BOLD, gv.Format.GREEN, gv.Format.END, task_name, ip_port)
        else:
            print '{}Dead{} {} on {}. stdout : {}'.format(
                gv.Format.BOLD, gv.Format.END, task_name, ip_port, stdout)
    return True


def get_models_for_dataset(data_name_and_model_type):
    data_name_and_model_type = data_name_and_model_type.split(':')
    data_name = data_name_and_model_type[0]
    data_type = 'all'
    if len(data_name_and_model_type) > 1:
        data_type = data_name_and_model_type[1]

    # Each param in the model is a 4 byte float
    models = []
    if data_name == 'cifar10':
        models = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                  # number of params: 0.27m, 0.46m, 0.66m, 0.85m, 1.7m,
                  'densenet40_k12', 'densenet100_k12', 'densenet100_k24',
                  # number of params: 1.0m, 7.0m, 27.2m
                  # 'nasnet',
                  # number of params: 3.3m
                  ]
    if data_name == 'imagenet':
        models = ['vgg11', 'vgg16', 'vgg19',
                  # number of params: 133m, 138m, 144m,
                  'resnet50', 'resnet101', 'resnet152',
                  # number of params: 25.6m, 44.5m, 60.2m,
                  'googlenet', 'inception3', 'inception4',
                  # number of params: 5m, 23.8m, ?
                  # 'nasnet', 'nasnetlarge'
                  # number of params: ?, 88.9m
                  # 'lenet',  'overfeat', 'alexnet',
                  # number of params: ?, ?, 60m
                  ]

    if data_type != 'all':
        models = filter(lambda model_name: data_type in model_name, models)

    return models


# def release_ports(ip, ports_to_free):
#     command = ('fuser -k {ports}').format(
#         ports=' '.join(['{}/tcp'.format(p) for p in ports_to_free]))
#     exe_ssh(command, ip, 'release_ports', use_root=True)
