import argparse
import os
import threading
import time
import torch
import torch.distributed as dist

NUM_TRIALS = 2

def receive_tensor_helper(tensor, src_rank, group, tag, num_iterations,
                          intra_server_broadcast):
    for i in range(num_iterations):
        print("to receive")
        if intra_server_broadcast:
            dist.broadcast(tensor=tensor, group=group, src=src_rank)
        else:
            dist.recv(tensor=tensor, src=src_rank, tag=tag)
    print("Done with tensor size %s" % tensor.size())

def send_tensor_helper(tensor, dst_rank, group, tag, num_iterations,
                       intra_server_broadcast):
    for i in range(num_iterations):
        print("to send")
        if intra_server_broadcast:
            dist.broadcast(tensor=tensor, group=group, src=1-dst_rank)
        else:
            dist.send(tensor=tensor, dst=dst_rank, tag=tag)
    print("Done with tensor size %s" % tensor.size())

def start_helper_thread(func, args):
    helper_thread = threading.Thread(target=func,
                                     args=tuple(args))
    helper_thread.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test lightweight communication library')
    parser.add_argument("--backend", type=str, default='gloo',
                        help="Backend")
    parser.add_argument('-s', "--send", action='store_true',
                        help="Send tensor (if not specified, will receive tensor)")
    parser.add_argument("--master_addr", required=True, type=str,
                        help="IP address of master")
    parser.add_argument("--use_helper_threads", action='store_true',
                        help="Use multiple threads")
    parser.add_argument("--rank", required=True, type=int,
                        help="Rank of current worker")
    parser.add_argument('-p', "--master_port", default=12345,
                        help="Port used to communicate tensors")
    parser.add_argument("--intra_server_broadcast", action='store_true',
                        help="Broadcast within a server")

    args = parser.parse_args()

    num_ranks_in_server = 1
    if args.intra_server_broadcast:
        num_ranks_in_server = 2
    local_rank = args.rank % num_ranks_in_server
    torch.cuda.set_device(local_rank)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    world_size = 2
    dist.init_process_group(args.backend, rank=args.rank, world_size=world_size)

    tensor_sizes = [10, 100, 1000, 10000, 100000, 1000000, 10000000]

    groups = []
    for tag in range(len(tensor_sizes)):
        if args.intra_server_broadcast:
            group = dist.new_group([0, 1])
            groups.append(group)
    
    if args.send:
        dist.broadcast(torch.zeros(1).cuda(local_rank), group=group,src=0)
    else:
        dist.broadcast(torch.zeros(1).cuda(local_rank), group=group,src=0)

    for tag, tensor_size in enumerate(tensor_sizes):
        if args.intra_server_broadcast:
            group = groups[tag]
        else:
            group = None

        if args.send:
            if args.intra_server_broadcast:
                tensor = torch.tensor(range(tensor_size), dtype=torch.float32).cuda(
                    local_rank)
            else:
                tensor = torch.tensor(range(tensor_size), dtype=torch.float32).cpu()
            if args.use_helper_threads:
                start_helper_thread(send_tensor_helper,
                                    [tensor, 1-args.rank,
                                     group, tag, NUM_TRIALS,
                                     args.intra_server_broadcast])
            else:
                send_tensor_helper(tensor, 1-args.rank, group, tag,
                                   NUM_TRIALS, args.intra_server_broadcast)
        else:
            if args.intra_server_broadcast:
                tensor = torch.zeros((tensor_size,), dtype=torch.float32).cuda(
                    local_rank)
            else:
                tensor = torch.zeros((tensor_size,), dtype=torch.float32).cpu()
            if args.use_helper_threads:
                start_helper_thread(receive_tensor_helper,
                                    [tensor, 1-args.rank,
                                     group, tag, NUM_TRIALS,
                                     args.intra_server_broadcast])
            else:
                receive_tensor_helper(tensor, 1-args.rank, group, tag,
                                      NUM_TRIALS, args.intra_server_broadcast)