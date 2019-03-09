#!/usr/bin/python

import global_variables as gv
import remote_tc
import utils


import copy  # to expand new config from basic config
import itertools  # to produce value products of options dict
import multiprocessing
import os
import subprocess as sp
import traceback
import time


def monitor(monitor_me, ip, exp_name='', log_every_sec=1, interface='',
            time_to_start_tcpdump_epoch_sec=0, duration_tcpdump_sec=0,
            use_root=False):
    monitor_command = (
        '{h}/tools/{m}.sh {sec} {ip} {itf} {start_tcpdump} {duration_tcpdump} '
        '> monitor_{m}_{exp}.txt').format(
        h=gv.TF_HOME, m=monitor_me, sec=log_every_sec, ip=ip, itf=interface,
        exp=exp_name, start_tcpdump=time_to_start_tcpdump_epoch_sec,
        duration_tcpdump=duration_tcpdump_sec)
    task_name = 'monitor_{m}_{ip}_{exp}'.format(
        m=monitor_me, ip=ip.split('.')[-1], exp=exp_name)
    command = ('''cd {h}/logs/; screen -dmS  {t} bash -c \\"{c}\\"''').format(
        h=gv.TF_HOME, t=task_name, c=monitor_command)
    utils.exe_ssh(command, ip, 'monitor_' + monitor_me, use_root=use_root)
    return (task_name, ip)


def write_monitor_logs_to_db(setups, all_ips_in_experiments):
    # handle vmstat/ifstat/tcpdump profile logs
    def handle_machine_profile(monitor_stat,  # db_write_function,
                               setups):
        local_log_file_name = './{}/{}.txt'.format(
            monitor_stat, utils.canonical_exp_name(setups))
        sp.call(['rm', '-f', local_log_file_name])

        for ip in all_ips_in_experiments:
            command = ('{{ echo \'-----{exp} at {ip} log begins------\'; '
                       'ssh {ip} "find {tfh}/logs/ -name monitor_{m}_{exp}.txt '
                       '          -exec cat {{}} \;";'
                       '}} >> {f}').format(
                exp=utils.canonical_exp_name(setups), m=monitor_stat, ip=ip,
                h=ip.split('.')[-1], tfh=gv.TF_HOME, f=local_log_file_name)
            scp = sp.Popen(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
            for l in scp.stderr:
                print '{}write_monitor_logs_to_db(); {}; {}{}'.format(
                    gv.Format.RED, ip, l, gv.Format.END)
            # has_useful_info = db_write_function(setups, local_log_file_name)

    # Sunny: Comment out db functionalities
    handle_machine_profile('vmstat',  # dump_to_db.write_logs_vmstat,
                           setups)
    handle_machine_profile('ifstat',  # dump_to_db.write_logs_ifstat,
                           setups)
    handle_machine_profile('tcpdump',  # dump_to_db.write_logs_tcpdump,
                           setups)


def control_and_monitor(pre_train_barrier, post_train_barrier, experiments):
    multiprocessing.log_to_stderr()
    ctrlC_to_exit = False

    try:
        signature_setups = sorted(experiments, key=lambda run: run['0run'])[0]
        signature_info = utils.canonical_exp_name(signature_setups)
        all_ips_in_experiments = set()
        for setups in experiments:
            for ip_port in setups['addresses']['ps'] + setups['addresses']['worker']:
                all_ips_in_experiments.add(ip_port.split(':')[0])

        if gv.COLLECT_RESULTS_ONLY:
            write_monitor_logs_to_db(signature_setups, all_ips_in_experiments)
            return

        # For simplicity, all concurrent runs must agree to monitor in order to
        # launch the monitor task.
        monitor_tasks_under_user = []
        has_monitor_logs = False
        if all(['should_vmstat' in setups and setups['should_vmstat']
                for setups in experiments]):
            monitor_tasks_under_user += [
                monitor('vmstat', ip, signature_info) for ip in all_ips_in_experiments]

            print '[{n}] controller: {fg}vmstat monitor enabled{fe}'.format(
                n=utils.now(), fg=gv.Format.GREEN, fe=gv.Format.END)
        if all(['should_ifstat' in setups and setups['should_ifstat']
                for setups in experiments]):
            monitor_tasks_under_user += [
                monitor('ifstat', ip, signature_info,
                        interface=gv.DATA_PLANE_INF)
                for ip in all_ips_in_experiments]
            print '[{n}] controller: {fg}ifstat monitor enabled{fe}'.format(
                n=utils.now(), fg=gv.Format.GREEN, fe=gv.Format.END)
        monitor_tasks_under_root = []
        if all(['should_tcpdump' in setups and setups['should_tcpdump']
                for setups in experiments]):
            # Start logging 4 minutes after stage begins and lasts for 20 seconds.
            # Under debug model, tcpdump starts 1 minutes after stage begins.
            start_sec = int(gv.NEW_STAGE_BEGIN_USEC / 1e6)
            start_sec += 60 * (1 if gv.DEBUG_LEVEL > 0 else 4)
            monitor_tasks_under_root += [
                monitor('tcpdump', ip, signature_info,
                        interface=gv.DATA_PLANE_INF, use_root=True,
                        time_to_start_tcpdump_epoch_sec=start_sec,
                        duration_tcpdump_sec=40)
                for ip in all_ips_in_experiments]
            print '[{n}] controller: {fg}tcpdump monitor enabled{fe}'.format(
                n=utils.now(), fg=gv.Format.GREEN, fe=gv.Format.END)

        tc_leaders_configs = []
        for key, group in itertools.groupby(experiments, key=lambda run: [
                ipp.split(':')[0] for ipp in run['addresses']['ps']]):
            runs_one_group = list(group)
            leader_setups = sorted(
                runs_one_group, key=lambda run: run['0run'])[0]
            leader_setups_copy = copy.deepcopy(leader_setups)
            leader_setups_copy['concurrent_ps_configs'] = [{
                'ps_port': run_['addresses']['ps'][0].split(':')[1],
                'wk_port': run_['addresses']['worker'][0].split(':')[1],
                'model': run_['args']['model'],
            } for run_ in runs_one_group]
            tc_leaders_configs.append(leader_setups_copy)

        print('[{}] controller: Reset and setup tc command '
              'on {} PS leaders'.format(utils.now(), len(tc_leaders_configs)))
        # clean up any pre-existing rules for traffic control
        [remote_tc.reset_dev_remote(ip) for ip in all_ips_in_experiments]
        tc_tasks_under_root = [remote_tc.setup_remote_tc(
            tc_configs) for tc_configs in tc_leaders_configs]

        pre_train_barrier.wait()

        post_train_barrier.wait()

    except KeyboardInterrupt:
        ctrlC_to_exit = True
        print('{}controller: Caught Ctrl-C. Will exit after cleanup {}{}').format(
            gv.Format.RED, info, gv.Format.END)
    except Exception as e:
        multiprocessing.get_logger().error(traceback.format_exc())
    finally:
        # remove monitor tasks
        [utils.kill_by_tasks(monitor_tasks_under_user) for i in range(0, 3)]
        [utils.check('dead', n, ip, verbose=gv.DEBUG_LEVEL > 0)
         for (n, ip) in monitor_tasks_under_user]

        # remove remote_tc tasks, which run under root
        [utils.kill_by_tasks(tc_tasks_under_root + monitor_tasks_under_root,
                             use_root=True) for i in range(0, 3)]
        [utils.check('dead', n, ip, use_root=True, verbose=gv.DEBUG_LEVEL > 0)
         for (n, ip) in tc_tasks_under_root + monitor_tasks_under_root]

        # reset tc on all hosts
        print('{}[{}] controller: after finishing one stage of experiment, '
              'now reset tc on {} {}').format(
            gv.Format.YELLOW, utils.now(), all_ips_in_experiments,  gv.Format.END)
        [remote_tc.reset_dev_remote(ip) for ip in all_ips_in_experiments]

        # write monitor logs to db
        if not ctrlC_to_exit:
            write_monitor_logs_to_db(signature_setups, all_ips_in_experiments)

        print('[{}] Exiting controller. Bye!'.format(utils.now()))
