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

## Finetuen CPM

On each host, run the following commands in the `cpm` directory.
```bash
export GLOO_SOCKET_IFNAME=NETWORK_INTERFACE_TO_USE
# set $t to the total number of gpus, $x to the total number of hosts, $y to the rank of each host, $z to the number of gpus per host and $addr to the master address
python -m launch --nnodes $x --node_rank $y --nproc_per_node $z main_with_runtime.py --data_dir data --master_addr $addr --module medium_$t --checkpoint_dir output --partition medium_$t/vpipe.json --sync_mode asp --distributed_backend gloo -b 2 --lr 0.000600 --lr_policy polynomial --weight-decay 0.000000 --epochs 20 --print-freq 100 --verbose 0 --num_ranks_in_server $z --config_path medium/mp_conf.json
```