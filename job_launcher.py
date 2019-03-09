#!/usr/bin/python

import global_variables as gv

from utils import *

import copy  # to expand new config from basic config
import os
from multiprocessing import Pool, Semaphore, Lock, Value, Manager
from multiprocessing.dummy import Pool as ThreadPool  # threading
import subprocess as sp
from time import sleep


def screen(task_name, ip_port, py_script, env_exports, args_list):
    host = ip_port.split(':')[0]
    launch_task = ('source ~/.bash_profile; {} '
                   # 'export TF_CPP_MIN_VLOG_LEVEL=1; '
                   '{}/bin/python -u {}/{} {} '  # force unbuffer print()
                   '  2>{}/logs/{}.err >{}/logs/{}.out').format(
        ' '.join(env_exports), gv.VENV_DIR, gv.MODEL_DIR, py_script,
        ' '.join(args_list), gv.TF_HOME, task_name, gv.TF_HOME, task_name)
    # if 'eval' in task_name:
    #     print '-' * 75
    #     print launch_task.replace(
    #         '; ', ';\n').replace('--', '\n  --').replace'2>', '\n  2>')
    command = ('cd {h}/logs/; screen -dmS {t} bash -c \\"{l}\\"').format(
        h=gv.TF_HOME, t=task_name, l=launch_task)
    exe_ssh(command, host, 'screen')


def overwrite_setups(basic_setups, overrides):
    setups = copy.deepcopy(basic_setups)
    # entries that needs special treatments
    if '0run' in overrides:
        setups['0run'] = overrides['0run']

    should_colocate_workers = ('should_colocate_workers' in setups
                               and setups['should_colocate_workers']) \
        or ('should_colocate_workers' in overrides
            and overrides['should_colocate_workers'])
    port_offset = setups['0run']
    if 'ps_idx' in overrides:
        setups['addresses']['ps'] = [
            gv.get_data_ip_port(
                x, 2000 + port_offset + (i if should_colocate_workers else 0))
            for i, x in enumerate(overrides['ps_idx'])]
        del overrides['ps_idx']
    if 'worker_idx' in overrides:
        setups['addresses']['worker'] = [
            gv.get_data_ip_port(
                x, 3000 + port_offset + (i if should_colocate_workers else 0))
            for i, x in enumerate(overrides['worker_idx'])]
        del overrides['worker_idx']

    # entries that belongs to args
    for arg_key in filter(lambda k: 'args.' in k,  list(overrides.keys())):
        arg_name = arg_key.split('.')[-1]
        setups['args'][arg_name] = overrides[arg_key]
        del overrides[arg_key]

    # by default, transcribe the rest
    for key, val in overrides.iteritems():
        setups[key] = val

    # derive basic metrics
    setups['0num_workers'] = len(setups['addresses']['worker'])
    setups['args']['train_dir'] = '{}/logs/{}'.format(
        gv.TF_HOME, canonical_exp_name(setups))
    setups['args']['eval_dir'] = '{}/logs/{}'.format(
        gv.TF_HOME, canonical_exp_name(setups))
    setups['network_used'] = gv.get_data_ip_port()

    return setups


class Launcher():

    def __init__(self, setups):
        self.setups_ = setups
        self.info_ = canonical_exp_name(setups)
        self.ps_hosts_ = '--ps_hosts=\'{}\''.format(
            ','.join(setups['addresses']['ps']))
        self.worker_hosts_ = '--worker_hosts=\'{}\''.format(
            ','.join(setups['addresses']['worker']))

        self.args_list_ = ['--{}={}'.format(k, v)
                           for (k, v) in setups['args'].iteritems()]

        if 'should_barrier' in setups and setups['should_barrier']:
            self.args_list_.append('--measure_barrier_wait=True')

        self.env_exports_ = ['export {}={};'.format(k, v)
                             for (k, v) in setups['env_vars'].iteritems()]
        self.all_tasks_ = \
            [(canonical_task_name(setups, 'ps', idx), 'ps', idx, ip_port)
                for idx, ip_port in enumerate(setups['addresses']['ps'])] \
            + [(canonical_task_name(setups, 'eval'), 'eval', 0, setups['addresses']['ps'][0])] \
            + [(canonical_task_name(setups, 'wk', idx), 'wk', idx, ip_port)
               for idx, ip_port in enumerate(setups['addresses']['worker'])]

    def launch_ps_eval(self):

        config = self.setups_
        all_tasks = self.all_tasks_
        args_list = self.args_list_
        env_exports = self.env_exports_
        ps_hosts = self.ps_hosts_
        worker_hosts = self.worker_hosts_

        ps_eval_tasks = filter(lambda x: x[1] in ['ps', 'eval'],
                               self.all_tasks_)

        for (task_name, _, idx, ip_port) in filter(lambda x: x[1] == 'ps', ps_eval_tasks):
            screen(task_name, ip_port, config['binary'], env_exports,
                   args_list + [ps_hosts, worker_hosts] +
                   ['--job_name=\'ps\'',
                    '--task_index={}'.format(idx),
                    # '--trace_file={}/logs/{}.trace'.format(gv.TF_HOME, task_name),
                    ])

        for (task_name, _, idx, ip_port) in filter(lambda x: x[1] == 'eval', ps_eval_tasks):
            eval_args_list = []
            for (k, v) in config['args'].iteritems():
                if k == 'variable_update' and v == 'distributed_replicated':
                    arg = '--variable_update=independent'
                elif k == 'eval_calculate_accuracy':
                    # never eval accuracy while in training
                    arg = '--eval_calculate_accuracy=False'
                else:
                    arg = '--{}={}'.format(k, v)
                eval_args_list.append(arg)

            screen(task_name, ip_port, config['binary'], env_exports,
                   eval_args_list +
                   ['--eval=true',
                    '--variable_update=parameter_server',
                    '--save_model_secs={}'.format(config['save_model_secs']),
                    # Since this eval is a light weight thread, it should wake up
                    # more often to check if global step/accuray.
                    # '--eval_interval_secs={}'.format(min(60, config['save_model_secs'] / 3)),
                    '--eval_interval_secs=60',
                    '--num_intra_threads={:d}'.format(1),
                    '--num_inter_threads={:d}'.format(1)])
        check_results = [check('alive', name, ipp, verbose=gv.DEBUG_LEVEL > 0)
                         for (name, _, _, ipp) in ps_eval_tasks]

    def launch_eval(self):

        config = self.setups_
        all_tasks = self.all_tasks_
        args_list = self.args_list_
        env_exports = self.env_exports_
        ps_hosts = self.ps_hosts_
        worker_hosts = self.worker_hosts_

        eval_tasks = filter(lambda x: x[1] == 'eval', self.all_tasks_)

        for (task_name, _, idx, ip_port) in eval_tasks:
            # eval as many checkpoints as possible
            eval_args_list = ['--eval_from_latest=False']
            for (k, v) in config['args'].iteritems():
                if k == 'variable_update' and v == 'distributed_replicated':
                    arg = '--variable_update=independent'
                elif k == 'eval_calculate_accuracy':
                    # when launching eval without ps, we assume eval is used
                    # to eval accuracy on the checkpoints available on the
                    # directory.
                    arg = '--eval_calculate_accuracy=True'
                else:
                    arg = '--{}={}'.format(k, v)
                eval_args_list.append(arg)
            screen(task_name, ip_port, config['binary'], env_exports,
                   eval_args_list +
                   ['--eval=true',
                    '--variable_update=parameter_server',
                    '--save_model_secs={}'.format(config['save_model_secs']),
                    # Since all checkpoints are ready to be evaluated, there is no
                    # need for eval to wait to evaluate.
                    '--eval_interval_secs=1',
                    '--num_intra_threads={:d}'.format(1),
                    '--num_inter_threads={:d}'.format(1)])
        check_results = [check('alive', name, ipp, verbose=False)
                         for (name, _, _, ipp) in eval_tasks]

    def launch_wk(self):
        config = self.setups_
        args_list = self.args_list_
        env_exports = self.env_exports_
        ps_hosts = self.ps_hosts_
        worker_hosts = self.worker_hosts_
        all_tasks = self.all_tasks_

        num_threads = config['num_threads'] if 'num_threads' in config else 1

        def target(args):
            task_name, _, idx, ip_port = args
            screen(task_name, ip_port, config['binary'], env_exports,
                   args_list + [ps_hosts, worker_hosts] +
                   ['--job_name=\'worker\'',
                    '--task_index={}'.format(idx),
                    '--tf_random_seed={}'.format(idx),
                    '--save_model_secs={}'.format(config['save_model_secs']),
                    '--num_intra_threads={:d}'.format(num_threads),
                    '--num_inter_threads={:d}'.format(num_threads),
                    # '--trace_file={}/logs/{}.trace'.format(gv.TF_HOME, task_name),
                    ])

        wk_tasks = filter(lambda x: x[1] == 'wk', all_tasks)
        should_colocate_workers = 'should_colocate_workers' in config \
            and config['should_colocate_workers']
        if not should_colocate_workers:
            launch_pool = ThreadPool(len(wk_tasks))
            launch_pool.map(target, wk_tasks)
        else:
            [target(wk_task) for wk_task in wk_tasks]

        all_tasks_name_ipp = [(name, ipp) for (name, _, _, ipp) in all_tasks]

        check_results = [check('alive', name, ipp, verbose=False)
                         for (name, ipp) in all_tasks_name_ipp]

        if all(check_results):
            return all_tasks_name_ipp, True
        else:
            print '{}Error: Fail to launch {}. Going to kill its tasks.{}'.format(
                gv.Format.RED, canonical_exp_name(config), gv.Format.END)
            kill_by_tasks(all_tasks_name_ipp)
            return all_tasks_name_ipp, False


class Barrier:

    def __init__(self, n):
        self.n = n
        self.count = Manager().Value('i', 0)
        self.mutex = Manager().Lock()
        self.barrier = Manager().Semaphore(0)

    def wait(self):
        self.mutex.acquire()
        self.count.value += 1
        # print '{}: {} -> {}'.format(id(self.count), self.count.value, self.n)
        self.mutex.release()
        if self.count.value == self.n:
            self.barrier.release()
        self.barrier.acquire()
        self.barrier.release()


def launch_one_run(pre_train_barrier, post_train_barrier, launcher):

    setups = launcher.setups_

    # Interleave the launch time to avoid overloading ssh connections. When to
    # launch ps and eval does not matter because the actual launch time is
    # measured as the time to launch workers.
    if 'launch_interval_sec' in setups and setups['launch_interval_sec'] > 0:
        sleep(setups['launch_interval_sec'])

    if gv.COLLECT_RESULTS_ONLY:
        write_to_db(setups, '')
        return

    if gv.DEBUG_LEVEL > 0:
        print str(launcher.all_tasks_).replace('),', '),\n').replace(')]', ')]\n')

    launcher.launch_ps_eval()

    pre_train_barrier.wait()

    info = canonical_exp_name(setups)

    # After waiting for the pretrain barrier, we again need to interleave the
    # workers launch time of  concurrent jobs.
    if 'launch_interval_sec' in setups and setups['launch_interval_sec'] > 0:
        # print('{fcolor}[{now}] Sleep for {delay} seconds before launching '
        #       'workers for {i}{fend}').format(
        #     fcolor=gv.Format.DARKCYAN, now=now(),
        #     delay=setups['launch_interval_sec'], i=info, fend=gv.Format.END)
        sleep(setups['launch_interval_sec'])

    timeout = False
    ctrlC_to_exit = False

    timeout_start_ts = datetime.datetime.now()

    launch_time = now()
    tasks, has_launch = launcher.launch_wk()

    is_training_done = False
    timeout_hr = 3  # initialized to the time allowed for training

    print '{fcray}[{n}] Launched {i}{fend} (took {ld} seconds) {bar}'.format(
        fcray=gv.Format.DARKCYAN, n=now(), i=info, fend=gv.Format.END,
        ld=(datetime.datetime.strptime(now(), '%Y-%m-%d %H:%M:%S') -
            datetime.datetime.strptime(launch_time, '%Y-%m-%d %H:%M:%S')
            ).total_seconds(),
        bar='({}barrier enabled{})'.format(
            gv.Format.GREEN, gv.Format.END) if setups['should_barrier'] else '',
    )

    if has_launch:
        try:
            while True:
                # We assume there is only 1 single eval task
                t, addr = filter(lambda t: 'eval' in t[0], tasks)[0]
                # wait until the eval process has finished.
                done = not check('alive', t, addr,
                                 return_true_on_connection_lost=True)

                if datetime.datetime.now() - timeout_start_ts \
                        > datetime.timedelta(hours=timeout_hr):
                    timeout = True
                    done = True

                # relaunch eval to calculate accuracy
                if done and not is_training_done:
                    is_training_done = True
                    print '{}[{}] Training Done for {}{}'.format(
                        gv.Format.DARKCYAN, now(), info, gv.Format.END)
                    # kill tasks for 3 times
                    [kill_by_tasks(tasks) for i in range(0, 3)]
                    [check('dead', name, ipp, verbose=True) for (name, ipp) in
                        filter(lambda t: 'ps' in t[0], tasks)]

                    post_train_barrier.wait()

                    if 'eval_calculate_accuracy' in setups['args'] \
                            and setups['args']['eval_calculate_accuracy']:
                        print '{}[{}] Relaunching eval for {}{}'.format(
                            gv.Format.DARKCYAN, now(), info, gv.Format.END)
                        # Interleave the launch time to avoid overloading ssh.
                        # When to launch eval does not matter.
                        sleep(setups['launch_interval_sec'])
                        launcher.launch_eval()
                        # Reset clock to time out eval.
                        timeout_start_ts = datetime.datetime.now()
                        # Allow how many hours for eval.
                        timeout_hr = 8
                        done = False

                if done:
                    break  # while

                # target task(s) under the experiment is alive. Keep waiting.
                check_every_sec = 60 if gv.DEBUG_LEVEL < 1 else 3
                sleep(check_every_sec)  # check health each n minutes
        except KeyboardInterrupt:
            ctrlC_to_exit = True
            print('{}(r{}) Caught Ctrl-C. Will exit after cleanup {}{}').format(
                gv.Format.RED, setups['0run'], info, gv.Format.END)
        if timeout:
            print '{}[{}] TIMEOUT {} {}'.format(gv.Format.RED, now(), info, gv.Format.END)
        elif ctrlC_to_exit:
            print '{}[{}] Ctrl-C {} {}'.format(gv.Format.BLUE, now(), info, gv.Format.END)
        else:
            print '{}[{}] DONE {} {}'.format(gv.Format.BLUE, now(), info, gv.Format.END)

        # Interleave the time to start cleaning up to avoid overloading ssh.
        sleep(setups['launch_interval_sec'])
        kill_by_tasks(tasks)
        if not ctrlC_to_exit:
            write_to_db(setups, launch_time)

    # pause for a while before moving on
    count_down(3)  # in seconds


def count_down(seconds):
    if gv.DEBUG_LEVEL >= 2:
        sys.stdout.write('Pausing for a while ..')
        for count in range(seconds, 0, -1):
            sys.stdout.write('{}..'.format(count))
            sys.stdout.flush()
            sleep(1)
        sys.stdout.write('\n')
    else:
        sleep(seconds)


def write_to_db(setups, launch_time):

    # handle eval file - eval.out is copied to tf_run/ dir in local host
    local_eval_file_name = './evals/{}.eval'.format(
        canonical_exp_name(setups))
    eval_host = setups['addresses']['ps'][0].split(':')[0]
    command = ('scp {}:{}/logs/{}*eval.out {}').format(
        eval_host, gv.TF_HOME, canonical_exp_name(setups), local_eval_file_name)
    scp = sp.Popen(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    for l in scp.stderr:
        print '{}write_to_db() collecting evals; {}; {}{}'.format(
            gv.Format.RED, eval_host, l, gv.Format.END)
    # TODO: write out launch_time to file.
    # dump_to_db.write_learning_curve(
    #     setups, '{}'.format(local_eval_file_name), launch_time)

    def retrieve_std_file(err_or_out, local_std_file_name):
        sp.call(['rm', '-f', local_std_file_name])
        for ip_port in setups['addresses']['ps'] + setups['addresses']['worker']:
            t = 'ps' if ip_port in setups['addresses']['ps'] else 'wk'
            idx = setups['addresses']['ps'].index(ip_port) if t == 'ps' else setups[
                'addresses']['worker'].index(ip_port)
            ip = ip_port.split(':')[0]
            command = ('{{ echo \'-----{task_name} at {ip} log begins------\'; '
                       'ssh {ip} "find {tfh}/logs/ -name {task_name}.{eoo} -exec cat {{}} \;";'
                       '}} >> {f}').format(
                task_name=canonical_task_name(setups, t, idx), eoo=err_or_out,
                ip=ip, tfh=gv.TF_HOME, f=local_std_file_name)
            scp = sp.Popen(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
            for l in scp.stderr:
                print '{}write_to_db() collecting stderr; {}; {}{}'.format(
                    gv.Format.RED, ip, l, gv.Format.END)

    # retrieve stdout files for reference
    local_stdout_file_name = './stdout/{}.stdout'.format(
        canonical_exp_name(setups))
    retrieve_std_file('out', local_stdout_file_name)

    # retrieve stderr files for data analysis
    local_stderr_file_name = './stderr/{}.stderr'.format(
        canonical_exp_name(setups))
    retrieve_std_file('err', local_stderr_file_name)

    # Sunny: Comment out db functionalities
    # has_useful_info = dump_to_db.write_logs_barrier(
    #     setups, local_stderr_file_name)
    # has_useful_info = dump_to_db.write_logs_rendez(
    #     setups, local_stderr_file_name) or has_useful_info
    # has_useful_info = dump_to_db.write_logs_comm(
    #     setups, local_stderr_file_name) or has_useful_info
    # if not has_written_barrier_log and not has_written_rendez_log:
    # The barrier + rendez log is larger than 10 MB = 10 * 1024 * 1024 bytes,
    # we have to remove it.
