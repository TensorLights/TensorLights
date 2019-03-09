# TensorLights

TensorLights is a traffic scheduler for the host NICs to mitigate the traffic
contention among distributed deep learning applications (e.g. TensorFlow).
TensorLights is described in:

>[Green, Yellow, Yield: End-Host Traffic Scheduling for Distributed Deep Learning with TensorLights](https://www.cs.rice.edu/~eugeneng/papers/HPBDC19.pdf)

>[Xin Sunny Huang](http://www.cs.rice.edu/~xinh/),
[Ang Chen](http://www.cs.rice.edu/~angchen/),
and [T. S. Eugene Ng](http://www.cs.rice.edu/~eugeneng/)

>In _The 5th IEEE International Workshop on High-Performance Big Data and Cloud Computing ([HPBDC 2019](http://web.cse.ohio-state.edu/~lu.932/hpbdc2019/index.html))_

This repository contains the TensorLights implementation, as well as the necessary tools to launch applications and perform measurements, to replicate the experimental results reported in the paper. 

## Benchmark ##

This repository should work with [our TensorFlow benchmark](https://github.com/TensorLights/benchmarks) under the [TensorLights project](https://github.com/TensorLights). Our TensorLights-compatible benchmark comes from the official TensorFlow benchmark with minimal modifications to facilitate analysis. 

All  of our modifications to the benchmark are in the `scripts/tf_cnn_benchmarks/benchmark_cnn.py`. As an alternative to use our benchmark so as to leverage the [most updated benchmark from the official TensorFlow repository](https://github.com/tensorflow/benchmarks), one may modify the `benchmark_cnn.py` accordingly so that the benchmark may work with our cluster scheduler and TensorLights implementation.

## Prerequisite ##
- Read `global_variables.py` for the necessary machine setups.

Then configure each machine in the cluster as follows.

- Enable python virtual environment and install TensorFlow (r1.7) in the virtual environment. The directory for virtual environment should match
`VENV_DIR`.
- Have a copy of dataset under `TF_HOME`.
- Have a copy of the `\tools` directory under `TF_HOME`.
- Have a copy of TensorFlow benchmarks under `MODEL_DIR`. We instrumented the original benchmarks with basic support for our measurements. Please refer to another repo for our version of benchmarks.
- Allow root login via ssh from the the master machine that runs our cluster manager. This can be done by adding the public key of the master machine to the root's authorized_keys in each machine in the cluster.

`VENV_DIR`, `TF_HOME` and `MODEL_DIR` mentioned above are specified in `global_variables.py`.

## How to run ##

```
python cluster_manager.py -n cifar10TF -m mode
```

Choices for `mode` are `jct_measurement` and `profile_measurement`.

- `jct_measurement` performs basic measurements for job completion time without any profiling.
- `profile_measurement` may enable profiling for `vmstat`, `ifstat` and barrier wait time. Please see `cluster_manager.py` for details.

We recommend running our cluster manager on an extra machine that does NOT need
to run TensorFlow jobs. Machines running the TensorFlow jobs are specified by
`kINVENTORY` in  `global_variables.py`.


## Output / Logging ##
Our cluster manager is originally designed to pre-process the logs and dump all
performance metrics (e.g. job completion time) into an external database for
data analysis and visualization.

However, database connection requires installing new library and driver. To
remove such dependency and allow a quick launch on any machine, I remove or
comment out the code that relies on database connection.

Instead of writing metrics to database, the manager will now collect and store
logs under the corresponding directories, such as `stderr`,  `evals`, `vmstat`,
and `ifstat`.


## Testing ##

We have provided tests in several test  files named `test_*.py`. To run the
tests, run `python test_file.py`. Running tests requires the library of python
`unittest` and `mock`.
