#!/usr/bin/python

import global_variables as gv
import utils

import pickle


def reset_dev_remote(ip):
    command = ('tc qdisc del dev {dev} root; '
               'tc qdisc del dev {dev} ingress; ').format(
        dev=gv.DATA_PLANE_INF)

    utils.exe_ssh(command, ip, 'reset_dev_remote', use_root=True,
                  errors_to_ignore=['RTNETLINK answers: No such file or directory',
                                    'RTNETLINK answers: Invalid argument'])


def setup_remote_tc(tc_configs):

    info = utils.canonical_exp_name(tc_configs)

    tc_configs_file_name = '{}/logs/tc_configs.{}'.format(gv.TF_HOME, info)
    pickle.dump(tc_configs, open(tc_configs_file_name, 'wb'))

    ps_ip = tc_configs['addresses']['ps'][0].split(':')[0]

    print '[{}] setup_remote_tc: Setting up tc_agent on {}'.format(
        utils.now(), ps_ip)

    utils.scp_to_remote('setup_remote_tc', ps_ip,
                        tc_configs_file_name, tc_configs_file_name)

    enable_rotate = 'TLsRR' in tc_configs['poison']
    rotate_interval_sec = 10
    if gv.DEBUG_LEVEL == 0 and enable_rotate:
        if ':' in tc_configs['poison']:
            rotate_interval_sec = tc_configs['poison'].split(':')[-1]
        else:
             # if interval is not specified after TLsRR, use 20 sec by
             # default.
            rotate_interval_sec = 20

    tc_agent_args = {'func': 'setup_tc', 'dev': gv.DATA_PLANE_INF,
                     'tc_configs_dir': tc_configs_file_name,
                     'rotate_interval_sec': rotate_interval_sec}

    task_name = 'tc_agent_{ip_suffix}_{exp}'.format(
        ip_suffix=ps_ip.split('.')[-1], exp=info)

    launch_tc_agent_command = (
        'source ~/.bash_profile; '  # force unbuffer print()
        '{venv}/bin/python -u {h}/tools/tc_agent.py {args} {rotate}'
        '  2>{h}/logs/{task}.err >{h}/logs/{task}.out').format(
        venv=gv.VENV_DIR, h=gv.TF_HOME, task=task_name,
        args=' '.join(['--{}={}'.format(k, v)
                       for k, v in tc_agent_args.iteritems()]),
        rotate='--enable_rotate' if enable_rotate else '')
    command = ('cd {h}/logs/; screen -dmS {t} bash -c \\"{l}\\"').format(
        h=gv.TF_HOME, t=task_name, l=launch_tc_agent_command)

    utils.exe_ssh(command, ps_ip, 'setup_remote_tc', use_root=True)

    return (task_name, ps_ip)
