from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================

import argparse
from collections import OrderedDict
import importlib
import json
import os
import shutil
import sys
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms

import itertools
import math
from fairseq import data, distributed_utils, options, progress_bar, tasks, utils, bleu, tokenizer
from fairseq.fp16_trainer import FP16Trainer
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.data import dictionary
from fairseq.tasks import TASK_REGISTRY
from fairseq.criterions import CRITERION_REGISTRY
from fairseq.optim import lr_scheduler
from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY

sys.path.append("..")
import runtime
import lamb
import sgd
import adam

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
## Pipedream parameters
parser.add_argument('--data_dir', type=str,
                    help='path to dataset')
parser.add_argument('--distributed_backend', type=str,
                    help='distributed backend to use (gloo|nccl)')
parser.add_argument('--module', '-m', required=True,
                    help='name of module that contains model and tensor_shapes definition')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--max-tokens', default=2560, type=int,
                    metavar='N', help='maximum number of tokens in a batch (default: 2560)')
parser.add_argument('--grad-clip', default=5.0, type=float,
                    help='enabled gradient clipping and sets maximum gradient norm value')
parser.add_argument('--eval-batch-size', default=8, type=int,
                    help='eval mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_policy', default='step', type=str,
                    help='policy for controlling learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--fp16', action='store_true',
                    help='train model in fp16 precision')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='static loss scale, positive power of 2 to improve fp16 convergence')
parser.add_argument('--master_addr', default=None, type=str,
                    help="IP address of master (machine with rank 0)")
parser.add_argument('--config_path', default=None, type=str,
                    help="Path of configuration file")
parser.add_argument('--partition', default=None, type=str,
                    help="Path of partition configuration file")
parser.add_argument('--rank', default=None, type=int,
                    help="Rank of worker")
parser.add_argument('--local_rank', default=0, type=int,
                    help="Local rank of worker")
parser.add_argument('--forward_only', action='store_true',
                    help="Run forward pass only")
parser.add_argument('--num_minibatches', default=None, type=int,
                    help="Number of minibatches to run")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint_dir', default='', type=str, metavar='PATH',
                    help='path to directory to save checkpoints')
parser.add_argument('--checkpoint_dir_not_nfs', action='store_true',
                    help='checkpoint dir is not on a shared NFS server')
parser.add_argument('-s', '--synthetic_data', action='store_true',
                    help="Use synthetic data")
parser.add_argument('-v', '--verbose_frequency', default=0, type=int, metavar='N',
                    help="Log verbose information")
parser.add_argument('--num_ranks_in_server', default=1, type=int,
                    help="number of gpus per machine")

# Recompute tensors from forward pass, instead of saving them.
parser.add_argument('--recompute', action='store_true',
                    help='Recompute tensors in backward pass')
parser.add_argument('--sync_mode', type=str, choices=['asp', 'bsp'],
                    required=True, help='synchronization mode')
# Macrobatching reduces the number of weight versions to save,
# by not applying updates every minibatch.
parser.add_argument('--macrobatch', action='store_true',
                    help='Macrobatch updates to save memory')

parser.add_argument('--skip-invalid-size-inputs-valid-test', action='store_true',
                    help='ignore too long or too short lines in valid and test set')
parser.add_argument('--max-sentences', '--batch-size', type=int, metavar='N',
                    help='maximum number of sentences in a batch')
parser.add_argument('--sentencepiece', action='store_true',
                    help='use when dataset uses sentencepiece encoding')

parser.add_argument('--train-subset', default='train', metavar='SPLIT',
                    choices=['train', 'valid', 'test'],
                    help='data subset to use for training (train, valid, test)')
parser.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                    help='comma separated list of data subsets to use for validation'
                    ' (train, valid, valid1, test, test1)')
parser.add_argument('--max-sentences-valid', type=int, metavar='N',
                    help='maximum number of sentences in a validation batch'
                    ' (defaults to --max-sentences)')

parser.add_argument('--gen-subset', default='test', metavar='SPLIT',
                    help='data subset to generate (train, valid, test)')
parser.add_argument('--num-shards', default=1, type=int, metavar='N',
                    help='shard generation over N shards')
parser.add_argument('--shard-id', default=0, type=int, metavar='ID',
                    help='id of the shard to generate (id < num_shards)')

parser.add_argument('--task', metavar='TASK', default='translation', choices=TASK_REGISTRY.keys(),
                    help='task: {} (default: {})'.format(', '.join(TASK_REGISTRY.keys()), 'translation'))
parser.add_argument('--source-lang', default=None, metavar='SRC',
                    help='source language')
parser.add_argument('--target-lang', default=None, metavar='TARGET',
                    help='target language')
parser.add_argument('--raw-text', action='store_true',
                    help='load raw text dataset')
parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                    help='pad the source on the left (default: True)')
parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                    help='pad the target on the left (default: False)')
parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                    help='max number of tokens in the source sequence')
parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                    help='max number of tokens in the target sequence')
parser.add_argument('--pad-sequence', default=1, type=int, metavar='N',
                    help='Pad sequences to a multiple of N')

parser.add_argument('--criterion', default='label_smoothed_cross_entropy', metavar='CRIT',
                    choices=CRITERION_REGISTRY.keys(),
                    help='training criterion: {} (default: label_smoothed_cross_entropy)'.format(
                    ', '.join(CRITERION_REGISTRY.keys())),)
parser.add_argument('--label-smoothing', default=0.1, type=float, metavar='D',
                    help='epsilon for label smoothing, 0 means no label smoothing')

parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                    help='warmup the learning rate linearly for the first N updates')
parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                    help='initial learning rate during warmup phase; default is 0')
parser.add_argument('--lr-scheduler', default='inverse_sqrt',
                    help='learning rate scheduler: {} (default: inverse_sqrt)'.format(
                         ', '.join(LR_SCHEDULER_REGISTRY.keys())))

BSP = 'bsp'
ASP = 'asp'

best_prec1 = 0


# Helper methods.
def is_first_stage():
    return args.stage is None or (args.stage == 0)

def is_last_stage():
    return args.stage is None or (args.stage == (args.num_stages-1))

def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.data = args.data_dir
    # torch.cuda.set_device(args.local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.local_rank}"

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    load_dataset_splits(task, ['train', 'valid'])

    # Build criterion
    criterion = task.build_criterion(args)

    # create stages of the model
    partition = json.load(open(args.partition, 'r'))
    module = importlib.import_module(args.module)
    model = module.model(criterion, partition["partition"], partition["recompute_ratio"])

    max_positions = (args.max_source_positions, args.max_target_positions)
    dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens, max_positions)
    inputs = dummy_batch['net_input']
    input0 = inputs['src_tokens']
    input1 = input0.eq(1)
    input2 = inputs['prev_output_tokens']
    target = dummy_batch['target']

    training_tensor_shapes = {"input0": list(input0.size()), "input1": list(input1.size()),
                              "input2": list(input2.size()), "target": list(target.size()), "ntokens": [1]}
    dtypes = {"input0": input0.dtype, "input1": input1.dtype,
              "input2": input2.dtype, "target": target.dtype, "ntokens": torch.int8}
    inputs_module_destinations = {"input0": 0, "input1": 0, "input2": 0}
    target_tensor_names = {"target", "ntokens"}
    for module_id, (stage, inputs, outputs) in enumerate(model[:-1]):  # Skip last layer (loss).
        input_tensors = []
        for module_input in inputs:
            if module_input in inputs_module_destinations:
                inputs_module_destinations[module_input] = module_id

            input_tensor = torch.ones(tuple(training_tensor_shapes[module_input]),
                                      dtype=dtypes[module_input]).cuda()
            input_tensors.append(input_tensor)
        stage.cuda()
        # PyTorch should not maintain metadata for a backward pass on
        # synthetic inputs. Without the following line, the runtime is
        # as much as 1.5x slower in a full DP configuration.
        with torch.no_grad():
            output_tensors = stage(*tuple(input_tensors))
        stage.ctxs = []
        if not type(output_tensors) is tuple:
            output_tensors = [output_tensors]
        for output, output_tensor in zip(outputs,
                                         list(output_tensors)):
            training_tensor_shapes[output] = list(output_tensor.size())
            dtypes[output] = output_tensor.dtype

    eval_tensor_shapes = {}
    for key in training_tensor_shapes:
        eval_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])
        training_tensor_shapes[key] = tuple(
            training_tensor_shapes[key])

    configuration_maps = {
        'module_to_stage_map': None,
        'stage_to_rank_map': None,
        'stage_to_depth_map': None
    }
    if args.config_path is not None:
        json_config_file = json.load(open(args.config_path, 'r'))
        configuration_maps['module_to_stage_map'] = json_config_file.get("module_to_stage_map", None)
        configuration_maps['stage_to_rank_map'] = json_config_file.get("stage_to_rank_map", None)
        configuration_maps['stage_to_rank_map'] = {
            int(k): v for (k, v) in configuration_maps['stage_to_rank_map'].items()}
        configuration_maps['stage_to_depth_map'] = json_config_file.get("stage_to_depth_map", None)

    r = runtime.StageRuntime(
        model=model, distributed_backend=args.distributed_backend,
        fp16=args.fp16, loss_scale=args.loss_scale,
        training_tensor_shapes=training_tensor_shapes,
        eval_tensor_shapes=eval_tensor_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations=inputs_module_destinations,
        target_tensor_names=target_tensor_names,
        configuration_maps=configuration_maps,
        master_addr=args.master_addr,
        rank=args.rank, local_rank=args.local_rank,
        num_ranks_in_server=args.num_ranks_in_server,
        verbose_freq=args.verbose_frequency,
        model_type=runtime.TRANSFORMER,
        enable_recompute=args.recompute)

    # stage needed to determine if current stage is the first stage
    # num_stages needed to determine if current stage is the last stage
    # num_ranks needed to determine number of warmup_minibatches in case of pipelining
    args.stage = r.stage
    args.num_stages = r.num_stages
    args.num_ranks = r.num_ranks
    if not is_first_stage():
        args.synthetic_data = True

    # number of versions is the total number of machines following the current
    # stage, shared amongst all replicas in this stage
    num_versions = r.num_warmup_minibatches + 1

    # if specified, resume from checkpoint
    if args.resume:
        checkpoint_file_path = "%s.%d.pth.tar" % (args.resume, r.stage)
        assert os.path.isfile(checkpoint_file_path)
        print("=> loading checkpoint '{}'".format(checkpoint_file_path))
        checkpoint = torch.load(checkpoint_file_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        r.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_file_path, checkpoint['epoch']))

    # TODO: make this configurable by args
    use_adam_optimizer = True
    if use_adam_optimizer:
        optimizer = adam.AdamWithWeightStashing(
            modules=r.modules(), master_parameters=r.master_parameters,
            model_parameters=r.model_parameters, loss_scale=args.loss_scale,
            num_versions=num_versions, lr=args.lr, betas=(0.9,0.997),
            weight_decay=args.weight_decay, verbose_freq=args.verbose_frequency,
            macrobatch=args.macrobatch)
    else:
        optimizer = sgd.SGDWithWeightStashing(
            modules=r.modules(), master_parameters=r.master_parameters,
            model_parameters=r.model_parameters, loss_scale=args.loss_scale,
            num_versions=num_versions, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, verbose_freq=args.verbose_frequency)
    scheduler = lr_scheduler.build_lr_scheduler(args, optimizer)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    distributed_sampler = False
    if configuration_maps['stage_to_rank_map'] is not None:
        num_ranks_in_first_stage = len(configuration_maps['stage_to_rank_map'][0])
        if num_ranks_in_first_stage > 1:
            distributed_sampler = True

    # if checkpoint is loaded, start by running validation
    if args.resume:
        assert args.start_epoch > 0
        validate(val_loader, r, args.start_epoch-1)

    train_loader = data.EpochBatchIterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=16,
        seed=1,
        num_shards=1,
        shard_id=0,
    )

    for epoch in range(args.start_epoch, args.epochs):
        # train or run forward pass only for one epoch
        if args.forward_only:
            validate(val_loader, r, epoch)
        else:
            train(train_loader, r, optimizer, epoch, scheduler)

            # evaluate on validation set
            # prec1 = validate(val_loader, r, epoch)
            prec1 = 0
            if r.stage != r.num_stages: prec1 = 0

            # remember best prec@1 and save checkpoint
            best_prec1 = max(prec1, best_prec1)

            should_save_checkpoint = args.checkpoint_dir_not_nfs or r.rank_in_stage == 0
            if args.checkpoint_dir and should_save_checkpoint:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': r.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict()
                }, args.checkpoint_dir, r.stage, epoch)


def train(train_loader, r, optimizer, epoch, lr_scheduler):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    n = r.num_iterations(loader_size=len(train_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)

    if not is_first_stage(): train_loader = None
    r.set_loader(train_loader)

    num_warmup_minibatches = r.num_warmup_minibatches

    # current step
    s = 0
    warmup_steps = 0

    epoch_start_time = 0
    batch_start_time = 0

    def pipelining(steps, print_freq, weight_stash=False):
        nonlocal s, epoch_start_time, batch_start_time
        # start num_warmup_minibatches forward passes
        for i in range(num_warmup_minibatches):
            r.run_forward()

        for i in range(steps - num_warmup_minibatches):
            s += 1
            
            # perform forward pass
            r.run_forward()

            if is_last_stage():
                # measure accuracy and record loss
                output, target, loss, num_tokens = r.output, r.target, r.loss.item(), r.num_tokens()
                losses.update(loss / num_tokens / math.log(2), num_tokens)

                if s == warmup_steps:
                    epoch_start_time = time.time()
                    batch_start_time = time.time()

                if s % print_freq == 0 and s > warmup_steps:
                    # measure elapsed time
                    batch_time.update((time.time() - batch_start_time)/print_freq)
                    batch_start_time = time.time()
                    epoch_time = (time.time() - epoch_start_time) / 3600.0
                    full_epoch_time = (epoch_time / float(s-warmup_steps)) * float(n)

                    print('Stage: [{0}] Epoch: [{1}][{2}/{3}]\t'
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Epoch time [hr]: {epoch_time:.3f} ({full_epoch_time:.3f})\t'
                          'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          args.stage, epoch, s, n, batch_time=batch_time,
                          epoch_time=epoch_time, full_epoch_time=full_epoch_time,
                          loss=losses, # top1=top1, top5=top5,
                          memory=(float(torch.cuda.memory_allocated()) / 10**9),
                          cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                    import sys; sys.stdout.flush()
            else:
                if s % print_freq == 0 and s > warmup_steps:
                    print('Stage: [{0}] Epoch: [{1}][{2}/{3}]\tMemory: {memory:.3f} ({cached_memory:.3f})'.format(
                          args.stage, epoch, s, n, memory=(float(torch.cuda.memory_allocated()) / 10**9),
                          cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                    import sys; sys.stdout.flush()

            # perform backward pass
            if not weight_stash:
                r.run_backward()
            else:
                optimizer.zero_grad()
                optimizer.load_old_params()
                r.run_backward()
                optimizer.load_new_params()
                lr_scheduler.step_update(s)
                optimizer.step()

        # finish remaining backward passes
        for i in range(num_warmup_minibatches):
            s += 1
            if not weight_stash:
                r.run_backward()
            else:
                optimizer.zero_grad()
                optimizer.load_old_params()
                r.run_backward()
                optimizer.load_new_params()
                lr_scheduler.step_update(s)
                optimizer.step()

        if not weight_stash:
            lr_scheduler.step_update(s)
            optimizer.base_optimizer.step()
            optimizer.zero_grad()

    if args.sync_mode == BSP:
        accumulation_steps = 32
        n -= (n % accumulation_steps)
        r.train(n)
        r.set_loss_scale(4 / accumulation_steps)
        print_freq = (args.print_freq // accumulation_steps) * accumulation_steps
        warmup_steps = 5*print_freq
        for t in range(n // accumulation_steps):
            pipelining(accumulation_steps, print_freq)
    else:
        r.train(n)
        warmup_steps = 5*args.print_freq
        pipelining(n, args.print_freq, weight_stash=True)


    # wait for all helper threads to complete
    r.wait()

    print("Epoch %d: %.3f seconds" % (epoch, time.time() - epoch_start_time))
    print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))


def validate(val_loader, r, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    n = r.num_iterations(loader_size=len(val_loader))
    if args.num_minibatches is not None:
        n = min(n, args.num_minibatches)
    r.eval(n)
    if not is_first_stage(): val_loader = None
    r.set_loader(val_loader)

    end = time.time()
    epoch_start_time = time.time()

    num_warmup_minibatches = r.num_warmup_minibatches

    with torch.no_grad():
        for i in range(num_warmup_minibatches):
            r.run_forward()

        for i in range(n - num_warmup_minibatches):
            # perform forward pass
            r.run_forward()
            r.run_ack()

            if is_last_stage():
                output, target, loss = r.output, r.target, r.loss.item()

                # measure accuracy and record loss
                # prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss)
                # top1.update(prec1[0], output.size(0))
                # top5.update(prec5[0], output.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Test: [{0}][{1}/{2}]\t'
                          'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Memory: {memory:.3f} ({cached_memory:.3f})\t'
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           epoch, i, n, batch_time=batch_time, loss=losses,
                           memory=(float(torch.cuda.memory_allocated()) / 10**9),
                           cached_memory=(float(torch.cuda.memory_cached()) / 10**9)))
                    import sys; sys.stdout.flush()

        if is_last_stage():
            print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        for i in range(num_warmup_minibatches):
             r.run_ack()

        # wait for all helper threads to complete
        r.wait()

        print('Epoch %d: %.3f seconds' % (epoch, time.time() - epoch_start_time))
        print("Epoch start time: %.3f, epoch end time: %.3f" % (epoch_start_time, time.time()))

    return top1.avg


# TODO: Verify that checkpointing works correctly for GNMT
def save_checkpoint(state, checkpoint_dir, stage, epoch):
    assert os.path.isdir(checkpoint_dir)
    checkpoint_file_path = os.path.join(checkpoint_dir, "checkpoint.%d.pth.tar.epoch.%d" % (stage, epoch))
    torch.save(state, checkpoint_file_path)
    print("Saved checkpoint to %s" % checkpoint_file_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
