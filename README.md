# VPipe

If you have any questions about VPipe, please contact sxzhao@cs.hku.hk for quick response.

## Overview

Repo architecture

**runtime**: contains our initial system and initial results

**cpm**: GPT-2 workaround on a Chinese dataset. Under active development to make vPipe support 3-D parallellism, NCCL backend, and dynamic scaling.

## Setup

For multi-node, make sure [nv_peer_mem](https://github.com/Mellanox/nv_peer_memory) driver is installed to achieve optimal communication performance.


### BERT

1. Setup Enviroment 

Note that you should modify the docker base image version to the Nvidia pytorch docker release 20.01. 

This may help you avoid an issue caused by the PyTorch variable version checking.

Docker file refer to : https://github.com/NVIDIA/DeepLearningExamples/blob/24b8c9c7fdfd1fa5b80d5c342f96dd922feffd24/PyTorch/LanguageModeling/BERT/Dockerfile


2. Download and preprocess the dataset.

BERT pre-training uses the following datasets:
-   Wikipedia
-   BookCorpus

To download, verify, extract the datasets, and create the shards in `.hdf5` format, see:  

https://github.com/NVIDIA/DeepLearningExamples/blob/24b8c9c7fdfd1fa5b80d5c342f96dd922feffd24/PyTorch/LanguageModeling/BERT/Dockerfile


3. Set up your machine and data locations in config files (see example, configs/bert_8vpipe.yml)


## Reproducing Experiments

### BERT

```bash
cd runtime
```

VPipe's optimal configuration for 8 GPUs
```bash
python driver.py --config_file configs/bert_8vpipe.yml
```

PipeDream's optimal configuration for 8 GPUs
```bash
python driver.py --config_file configs/bert_8pipedream.yml
```

GPipe's optimal configuration for 8 GPUs
```bash
python driver.py --config_file configs/bert_8gpipe.yml
```
## Some Raw Results (For your reference)

Environment:
2 host with each 4 RTX 2080 ti GPUs

Epoch hour:

vPipe: 1.28 hour
GPipe: 1.72 hour
Pipedream: 2.14 hour

