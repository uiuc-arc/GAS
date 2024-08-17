# GAS: Generating Fast and Accurate Surrogate Models for Autonomous Vehicle Systems

This repository partially reproduces the results shown in our paper:

**GAS: Generating Fast and Accurate Surrogate Models for Autonomous Vehicle Systems**\
Keyur Joshi, Chiao Hsieh, Sayan Mitra, Sasa Misailovic\
*35th IEEE International Symposium on Software Reliability Engineering (ISSRE 2024)*\
[Paper](paper/paper.pdf) | [Appendix](paper/appendix.pdf)

The structure of this repository is as follows:

* `acas`: contains the ACAS-Tab and ACAS-NN benchmarks
* `agbot`: contains the Crop-Monitor benchmark
* `lanenet`: contains the Cart-Straight and Cart-Curved benchmarks
* `misc`: contains a scalability experiment
* `paper`: contains the [main paper](paper/paper.pdf) and the [appendix](paper/appendix.pdf) containing additional plots and data

Additionally, the `agbot` and `lanenet` folders contain the `perceptionError` subfolder, which contains data for creating the perception model.

## General instructions for all benchmarks

### Requirements

Running the benchmark scripts requires Python 3.12 (we tested with Python 3.12.5 using [Miniconda](https://docs.anaconda.com/miniconda/)).
You will also need to install some packages by executing **one** of the following commands from the root folder of this repository:

    conda install --channel conda-forge --file requirements.txt

or

    python3 -m pip install -r requirements.txt

This repository may or may not work with other versions of Python 3 and the required packages.

### Safe state probability

Execute `GPC.py` in each benchmark folder to estimate the probability that the vehicle will remain in a safe state (i.e., not violate the safety property).
After execution, inspect the last ~10 lines of the output:

* The `Max KS` output shows the maximum value of the KS statistic over all time steps for each of the output state variables (Column 5 of Table 2)
* The `Max Wass` output similarly shows the maximum value of the Wasserstein metric (Column 6 of Table 2)
* If there were any time steps for which the t-test failed, the time steps where this occurred and the p-value will be displayed next in a list
* The `Total time` output shows the total time taken by the GPC model (`GPC`) and the original vehicle model (`MCS`)
* The `min t-val` output shows the minimum t-test p-value across all time steps
* The `l2Scaled` output shows the l2 error (Column 3 of Table 3)
* The `corr` output shows the cross correlation (Column 4 of Table 3)

#### Options

You can change various options within the benchmark to see how they affect the results by changing the corresponding variables.
A non-exhaustive list of options is given below:

* `TimeSteps` (default 100): the number of time steps to simulate
* `Order` (default 4): the order of the polynomial surrogate model produced by GPC; *note that if you change this option, you will have to remove the hard-coded optimizations for evaluation of order 4 polynomials*
* `NSamplesEvalGPC` (default 10000): the number of samples to use when evaluating the GPC surrogate model
* `NSamplesEvalMC` (default 1000): the number of samples to use when evaluating the original vehicle model

### Sensitivity analysis

Execute `sensitivity.py` in each benchmark folder for sensitivity analysis.

The maximum difference in sensitivity indices is shown in the first two numbers in the second-to-last line of output.
The numbers presented in Table 4 are the maximum of these two numbers for the corresponding type of sensitivity indiex.
Sensitivity analysis times are presented at the bottom, divided into multiple categories.

For sensitivity analysis with the abstracted vehicle model, the time presented in Column 2 of Table 8 is the sum of the following categories:

* `'Samples'`: time taken to create empirical samples for sensitivity analysis
* `'MCSSensEm'`: time taken for empirical sensitivity estimation with the abstracted vehicle model

For empirical sensitivity analysis with the GPC model, the time presented in Column 3 of Table 8 is the sum of the following categories:

* `'GPCInit'`: time taken to create the GPC model
* `'Samples'`: time taken to create empirical samples for sensitivity analysis
* `'GPCSensEm'`: time taken for empirical sensitivity estimation with the GPC model

For analytical sensitivity analysis with the GPC model, the time presented in Column 4 of Table 8 is the sum of the following categories:

* `'GPCInit'`: time taken to create the GPC model
* `'GPCSensAn'`: time taken for analytical sensitivity estimation with the GPC model

#### Options

You can change various options within the benchmark to see how they affect the results by changing the corresponding variables.
A non-exhaustive list of options is given below:

* `Order` (default 4): the order of the polynomial surrogate model produced by GPC; *note that if you change this option, you will have to remove the hard-coded optimizations for evaluation of order 4 polynomials*
* `NSamplesEval` (default 1000000): the number of samples to use for empirical estimation of sensitivity indices
* `deltaSensitivity` (default False): set to `True` for delta sensitivity indices (Column 3 of Table 4), or to `False` for normal sensitivity indices (Column 2 of Table 4)

## `acas`: ACAS-Tab and ACAS-NN benchmarks

The ACAS benchmarks use data from [this repository](https://github.com/sisl/HorizontalCAS).

### Safe state probability

See the general instructions for safe state probability analysis above.
There is one more option at the top: set `useNN` to `True` for ACAS-NN, and `False` for ACAS-Tab.

### Sensitivity analysis

See the general instructions for sensitivity analysis above.
There is one more option at the top: set `useNN` to `True` for ACAS-NN, and `False` for ACAS-Tab.

## `agbot`: Crop-Monitor benchmark

You can use `percept_mdl.py` in the `perceptionError` folder to re-create the perception model using image data captured within Gazebo.

### Safe state probability

See the general instructions for safe state probability analysis above.

**NOTE:** the `mcTraces` folder contains the traces of the vehicle using the original vehicle model captured within Gazebo.
We do not include the Gazebo component of the benchmarks as they require extensive setup and several hours of runtime.
Additionally, the time shown by the experiment for MCS using the original vehicle model *does not* include the time required to use Gazebo; it only shows the time required to read these trace files.

### Perception model parameters study

You can swap the perception model used to create the GPC model by modifying `perceptionModel.py`.

### Sensitivity analysis

See the general instructions for sensitivity analysis above.

## `lanenet`: Cart-Straight and Cart-Curved benchmarks

The Cart benchmarks use the simulator from [this repository](https://gitlab.engr.illinois.edu/gemillins/POLARIS_GEM_e2).

You can use `percept_mdl.py` in the `perceptionError` folder to re-create the perception model using image data captured within Gazebo.

`perceptionModel.py` contains perception models for both straight and curved roads.
To switch between the Cart-Straight and Cart-Curved benchmarks, simpliy change the perception model used within `perceptionModel.py`.
The rest of the benchmark will correspondingly switch between the two scenarios.

### Safe state probability

See the general instructions for safe state probability analysis above.

**NOTE:** the `mcTraces` folder contains the traces of the vehicle using the original vehicle model captured within Gazebo.
We do not include the Gazebo component of the benchmarks as they require extensive setup and several hours of runtime.
Additionally, the time shown by the experiment for MCS using the original vehicle model *does not* include the time required to use Gazebo; it only shows the time required to read these trace files.

### Perception model parameters study

You can swap the perception model used to create the GPC model by modifying `perceptionModel.py`.

### Sensitivity analysis

See the general instructions for sensitivity analysis above.

## `misc`: scalability experiment

`trunctest.py` runs a timing experiment for GPC model creation.
There are three options, which can be changed at the top:

* `order`: controls the order of GPC - set this to 4
* `dimensions`: controls the number of input dimensions
* `crossTruncation`: controls the cross truncation coefficient - lower is faster (due to greater truncation) but less accurate