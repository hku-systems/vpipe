# VPipe

This repository is under actively updating and merging our development branches.

README is simultaneously updating.

If you have any questions about VPipe, please contact sxzhao@cs.hku.hk for quick response.

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


### Download

```bash
python3 /workspace/bert/data/bertPrep.py --action download --dataset bookscorpus
python3 /workspace/bert/data/bertPrep.py --action download --dataset wikicorpus_en

python3 /workspace/bert/data/bertPrep.py --action download --dataset google_pretrained_weights  # Includes vocab

python3 /workspace/bert/data/bertPrep.py --action download --dataset squad
```

### Properly format the text files
```bash
python3 /workspace/bert/data/bertPrep.py --action text_formatting --dataset bookscorpus
python3 /workspace/bert/data/bertPrep.py --action text_formatting --dataset wikicorpus_en
```

### Shard the text files (group wiki+books then shard)
```bash
python3 /workspace/bert/data/bertPrep.py --action sharding --dataset books_wiki_en_corpus
```

### Create HDF5 files Phase 1
```bash
python3 /workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset books_wiki_en_corpus --max_seq_length 128 \
 --max_predictions_per_seq 20 --vocab_file $BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1
```

### Create HDF5 files Phase 2
```bash
python3 /workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset books_wiki_en_corpus --max_seq_length 512 \
 --max_predictions_per_seq 80 --vocab_file $BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1
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


