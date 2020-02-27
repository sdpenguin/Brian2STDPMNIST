# Learning with STDP

## Overview

This repository contains code for an on-going project to explore learning in spiking neural networks using local STDP-based learning rules. This project builds upon work presented in the paper by [Diehl & Cook (2015) "Unsupervised Learning of Digit recognition using STDP"](https://doi.org/10.3389/fncom.2015.00099). The code in this repo was originally based on the [original Brian code accompanying that paper](https://github.com/peter-u-diehl/stdp-mnist), subsequently [reimplemented for Brian 2 by zxzhijia](https://github.com/zxzhijia/Brian2STDPMNIST). The code in this repo has been modified for Python 3 and substantially refactored to aid understanding, robustness and flexibility, in preparation further development.

A first goal was to simply to understand the code and reproduce the results from Diehl & Cook (2015) – both the example supplied in the original code, as well as the other results presented in the paper regarding network size and different learning rules.

Subsequent goals include:
* exploring the resiliance of the trained SNN to noise (with comparison to more conventional ANNs),
* implementing supervised/associative learning (mostly done, but requiring further work),
* exploring the benefit of incorprating feedback connections,
* exploring the potential for training deeper networks,
* attempting to reduce the level of parameter-tuning required in the model,
* applying the model to more challenging datasets,
* and using this model as the basis for a range of further studies.

The initial part of this work was performed in collaboration with two MSci project students, Alexander Coles and Leam Howe.

## Using the code
This code is not yet intended for consumption by others and there is no additional documentation. I provide no assurances as to its performance. However, in the spirit of open science and open software, you are welcome to use it, providing you attribute the author in any subsequent code distribution or publications. I would also be grateful if you let me know! I aim to provide a more official licence and bibliographic reference here soon.

A `brian2` environment, including all dependencies, can be created using [conda](https://conda.io/):
```shell
conda env create -f environment.yml
```

A test using pre-trained weights (intended to reproduce that of the original Brian 1 code), may be achieved by running:
```shell
simulation.py --test
```

The corresponding example of training the weights from scratch is available by running:
```shell
simulation.py --train
```

Further usage information can be obtained via:
```shell
simulation.py --help
```
