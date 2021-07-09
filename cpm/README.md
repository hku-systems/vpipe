# CPM

## Setup

1. Install Python dependencies.
```bash
pip install -r requirements.txt
```

2. Install Apex.
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3. Preprocess the dataset and save it to `data`.

## Finetune CPM

On each host, run the following commands in the `cpm` directory.
```bash
export GLOO_SOCKET_IFNAME=NETWORK_INTERFACE_TO_USE
# set $t to the total number of gpus, $x to the total number of hosts, $y to the rank of each host (e.g., four hosts with node rank 0, 1, 2, 3), $z to the number of gpus per host and $addr to the master address
python -m launch --nnodes $x --node_rank $y --nproc_per_node $z main_with_runtime.py --data_dir data --master_addr $addr --module medium_$t --checkpoint_dir output --partition medium_$t/vpipe.json --sync_mode asp --distributed_backend gloo -b 2 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 100 --verbose 0 --num_ranks_in_server $z --config_path medium_$t/mp_conf.json
```

## Hybrid Parallelism 

Change the --config_path to medium_$t/dp_conf.json.

In dp_conf.json, map a stage to multiple GPU ranks, in order to apply data parallelism to per stage on multiple GPUs.

For example, in medium_4/dp_conf.json, we use 4 hosts with each holding 4 GPUs. We partition a CPM Medium model into 4 stages, and 
each stage is placed on one host. Inside a host, a stage is 
replicated on 4 GPUs with data parallelism. 

Note that, in dp_conf.json, the rank mapping is per GPU rank. And the node rank $y is per host rank. Thus, GPUs on node rank 0 is with GPU rank 0,1,2,3; GPUs on node rank 1 is with GPU rank 4,5,6,7. By configuring dp_conf.json, users can configure various hybrid topologies of DP+PP.

<!-- python3 -m launch --nnodes 1 --node_rank 0 --nproc_per_node 4 main_with_runtime.py --data_dir data --master_addr localhost --module medium_4 --checkpoint_dir output --partition medium_4/vpipe.json --sync_mode asp --distributed_backend gloo -b 2 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 100 --verbose 0 --num_ranks_in_server 4 --config_path medium_4/mp_conf.json -->

## 32 GPU (8 GPUs per host), model: CPM_Large, Pipeline stage  x 8  Data parallel replica 4 , batch size 2

Set NCCL interface name:

```export NCCL_SOCKET_IFNAME=ens5```


host 0 

```python3 -m launch --nnodes 4 --node_rank 0 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr localhost --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/32dp_conf.json -b 1
```
host 1 
```
python3 -m launch --nnodes 4 --node_rank 1 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136 --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/32dp_conf.json -b 1
```
host 2
```
python3 -m launch --nnodes 4 --node_rank 2 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136 --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/32dp_conf.json -b 1
```
host 3
```
python3 -m launch --nnodes 4 --node_rank 3 --nproc_per_node 8 main_with_runtime.py --data_dir data --master_addr 172.31.7.136 --module large_8 --checkpoint_dir output --partition large_8/gpipe.json --sync_mode asp --distributed_backend nccl --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 10 --verbose 0 --num_ranks_in_server 8 --config_path large_8/32dp_conf.json -b 1
```
