# VPipe


## Setup

For multi-node, make sure [nv_peer_mem](https://github.com/Mellanox/nv_peer_memory) driver is installed to achieve optimal communication performance.


## Experiments
### BERT

1. Setup Enviroment 


2. Download and preprocess the dataset.

BERT pre-training uses the following datasets:
-   Wikipedia
-   BookCorpus

To download, verify, extract the datasets, and create the shards in `.hdf5` format, run:  
```bash
/workspace/bert/data/create_datasets_from_start.sh
```

## Reproducing Experiments

### BERT

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


