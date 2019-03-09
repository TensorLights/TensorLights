#!/usr/bin/python

# ATTENTION: Please configure the following paths to fit your machine setups.
# TF_HOME is the home directory for all files
# For each machine in the cluster, we assume the following directory strucutre
#    TF_HOME/
#    - data_directory
#    - model_bins
#    - tf_venv                  (the virtual env where TensorFlow is installed)
#    - tools/
#    - logs/      (logs to be collected by the cluster manager for evaluations)

# A few things to note:

# 1. MODEL_DIR/binary is the python script that we will call to launch a task.
#    'binary' is specified in a job's config_template.

# 2. We copy the 'scripts/tf_cnn_benchmarks' subdirectory found the TensorFlow
#    benchmarks repository to be under model_bins/, so that we have
#    model_bins/tf_cnn_benchmarks/tf_cnn_benchmarks.py to run, which matches the
#    binary setup in our jobs' config_template. You may reconfigure MODEL_DIR or
#    binary for your own fit.

# 3. Please ensure a copy of the /tools subdirectory is available in
#    /TF_HOME/tools/ on each machine.

TF_HOME = '/path/to/your/tf_home'
MODEL_DIR = TF_HOME + '/model_bins'
VENV_DIR = TF_HOME + '/tf_venv'

# ATTENTION: Please configure the following network interfaces and IPs to fit
# your network setups.

# A few things to note:

# 1. We assume each machine has two interfaces (i.e. DATA_PLANE_INF and
#    CONTROL_PLANE_INF) for two parallel networks. The application traffic will
#    use the DATA_PLANE_INF, and the python scripts as the cluster manager use
#    CONTROL_PLANE_INF for remote control including copying log files from
#    remote machines.

# 2. We assume each machines has two IPs for the two interfaces. IPs are in the
#    form of 'aaa.bbb.ccc.ddd' The 4th field ('ddd') of the IPs should be the
#    same for both interface. The first 3 fields of the IPs (i.e. 'aaa.bbb.ccc')
#    for the same interface are identical. For example, the IPs for
#    DATA_PLANE_INF are '1.2.3.x', and the IPs for CONTROL_PLANE_INF are
#    '4.5.6.x'. Each machine has two IPs, i.e. '1.2.3.y' and '4.5.6.y', and 'y'
#    is the index for a machine as specified in kINVENTORY.

# 3. If your cluster setup is different from our assumptions, you may consider
#    to rewrite functions of get_data_ip_port() and get_control_ip_port() to fit
#    your cluster setups.

# Interface used for TensorFlow traffic.
DATA_PLANE_INF = 'eth0'
DATA_PLANE_IP_BASE = '1.2.3'
# Interface used for remote control. Also used for copying log files.
CONTROL_PLANE_INF = 'eth1'
CONTROL_PLANE_IP_BASE = '4.5.6'
 
# Indexes of machines to use in the cluster.
kINVENTORY = range(0, 21)

# how many process can this script (cluster_manager.py) use to launch
# jobs. At most this number of jobs will be launched at a time. This
# number should be <= number of jobs per stage.
MAX_NUM_PROCESS_CLUSTER_MANAGER = 65
# how many threads can be used for each worker machines. Set to 6
# because each machine has 6 real CPU cores.
MAX_NUM_THREADS_PER_MACHINE = 6

DEBUG_LEVEL = 0
COLLECT_RESULTS_ONLY = False

NEW_STAGE_BEGIN_USEC = 0

kNumPS = 1

# DB_TO_USE is useless after removing db writeout.
# Retain this variable for future reference only. 
DB_TO_USE = 'learning_default'

# Set a large number to enable the first and last checkpoint
SAVE_MODEL_SEC = 60 * 60 * 24  # one day

#=======================================================================


class Format:
    PINK = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def get_data_ip_port(ip_suffix='', port=''):
    ip = DATA_PLANE_IP_BASE
    if ip_suffix:
        ip = '{}.{}'.format(ip, ip_suffix)
    if port:
        ip = '{}:{}'.format(ip, port)
    return ip


def get_control_ip_port(ip_suffix='', port=''):
    ip = CONTROL_PLANE_IP_BASE
    if ip_suffix:
        ip = '{}.{}'.format(ip, ip_suffix)
    if port:
        ip = '{}:{}'.format(ip, port)
    return ip


def get_db_to_use():
    return DB_TO_USE

config_template = {
    'cifar10TF': {
        '0name': 'cifar10TF',
        'binary': 'tf_cnn_benchmarks/tf_cnn_benchmarks.py',
        'addresses': {
            'ps': [get_data_ip_port(0, 2000)],
            'worker': [get_data_ip_port(1, 3000)],
        },
        'save_model_secs': SAVE_MODEL_SEC,
        'poison': 'FIFO',
        'env_vars': {},
        'args': {
            'target_global_step': 5000,
            'batch_size': 32,
            'display_every': 100,
            'data_dir': '{}/data-cifar-10-py/'.format(TF_HOME),
            'data_name': 'cifar10',
            'train_dir': '{}/logs/'.format(TF_HOME),
            'eval_dir': '{}/logs/'.format(TF_HOME),
            'print_training_accuracy': 'True',
            'device': 'cpu',
            'data_format': 'NHWC',
            'model': 'resnet56',
            # distributed_replicated, parameter_server
            'variable_update': 'parameter_server',
            'cross_replica_sync': False,
            'local_parameter_device': 'cpu',
        },
    },
    'imagenetTF': {
        '0name': 'imagenetTF',
        'binary': 'tf_cnn_benchmarks/tf_cnn_benchmarks.py',
        'addresses': {
            'ps': [get_data_ip_port(0, 2000)],
            'worker': [get_data_ip_port(1, 3000)],
        },
        'save_model_secs': SAVE_MODEL_SEC,
        'poison': 'FIFO',
        'env_vars': {},
        'args': {
            'num_batches': 10000,
            'batch_size': 32,
            'display_every': 100,
            'data_dir': '{}/data-imagenet/'.format(TF_HOME),
            'data_name': 'imagenet',
            'train_dir': '{}/logs/'.format(TF_HOME),
            'eval_dir': '{}/logs/'.format(TF_HOME),
            'print_training_accuracy': 'True',
            'device': 'cpu',
            'data_format': 'NHWC',
            'model': 'resnet101',
            'variable_update': 'parameter_server',
            'cross_replica_sync': False,
            'local_parameter_device': 'cpu',
        },
    },
}
