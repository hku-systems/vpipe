import torch
import re
from apex.normalization.fused_layer_norm import FusedLayerNorm
from mpu.layers import VocabParallelEmbedding
from mpu.transformer import GPT2ParallelSelfAttention
from mpu.transformer import GPT2ParallelMLP
from mpu import copy_to_model_parallel_region, gather_from_model_parallel_region
from mpu import get_model_parallel_world_size

import torch.utils.checkpoint as cp
import checkpoint as pipe_cp

class CPM():
    def __init__(self, declares, calculations):
        self.declares = declares
        self.calculations = calculations

    def generate_layer_blocks(self):
        self.layers = {}
        for layer in self.declares.split('\n'):
            m = re.search(r'self.layer([0-9]+)', layer)
            layer_id = int(m.group(1))
            self.layers[layer_id] = layer

        self.blocks = [[]]
        for line in self.calculations.split('\n'):
            self.blocks[-1].append(line)
            if '+' in line:
                self.blocks.append([])

    def generate_stage(self, start, end):
        inputs = []
        outputs = []
        declare = []
        calculation = []
        for i in range(start, end):
            for line in self.blocks[i]:
                calculation.append(line)
                m = re.search(r'self.layer([0-9]+)', line)
                if m is not None:
                    layer_id = int(m.group(1))
                    declare.append(self.layers[layer_id])
                out = re.findall(r'out\d+', line)
                for arg in out[1:]:
                    if arg not in outputs and arg not in inputs:
                        inputs.append(arg)
                if out[0] not in outputs:
                    outputs.append(out[0])
        return declare, calculation, inputs, outputs

class Stage(torch.nn.Module):
    def __init__(self, inputs, outputs, declares, calcus, fraction):
        super(Stage, self).__init__()

        # print("{} {} {}".format(inputs, outputs, fraction), flush = True)
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.ctxs = []
        self.pipe = False

        exec('\n'.join(declares))

        back = int(fraction * len(calcus))

        if back == len(calcus):
            no_cp_ = ["{} = args[{}]".format(name, i) for i, name in enumerate(inputs)]
            no_cp_.append("cp_out = cp.checkpoint(self.cp_forward, {}, self.dummy)".format(','.join(inputs)))

            cp_ = calcus
            cp_i = 0
            cp_return = []
            no_cp_return = []
            for output in outputs:
                if output not in inputs:
                    cp_return.append(output)
                    no_cp_return.append("cp_out[{}]".format(cp_i))
                    cp_i += 1
                else:
                    no_cp_return.append(output)

            cp_ = ["{} = args[{}]".format(name, i) for i, name in enumerate(inputs)] + cp_
            cp_.append("self.cp_out = ({},)".format(', '.join(cp_return)))
            no_cp_.append("self.out = ({},)".format(', '.join(no_cp_return)))

            self.cp = '\n'.join(cp_)
            self.no_cp = '\n'.join(no_cp_)
        elif back == 0:
            self.cp = "assert 1 == 0"
            no_cp_ = calcus

            no_cp_ = ["{} = args[{}]".format(name, i) for i, name in enumerate(inputs)] + no_cp_
            no_cp_.append("self.out = ({})".format(', '.join(outputs)))

            self.no_cp = '\n'.join(no_cp_)
        else:
            # self.pipe = True
            no_cp_ = calcus[:-back]
            cp_ = calcus[-back:]

            no_cp_ = ["{} = args[{}]".format(name, i) for i, name in enumerate(inputs)] + no_cp_
            
            cp_inputs = []
            cp_outputs = []
            for line in cp_:
                out = re.findall(r'out\d+', line)
                for arg in out[1:]:
                    if arg not in cp_outputs and arg not in cp_inputs:
                        cp_inputs.append(arg)
                if out[0] not in cp_outputs:
                    cp_outputs.append(out[0])

            cp_i = 0
            cp_return = []
            no_cp_return = []
            for output in outputs:
                if output in cp_outputs:
                    cp_return.append(output)
                    no_cp_return.append("cp_out[{}]".format(cp_i))
                    cp_i += 1
                else:
                    no_cp_return.append(output)

            # no_cp_.append("cp_out = pipe_cp.checkpoint(self.cp_forward, self.ctxs, {})".format(', '.join(cp_inputs)))
            no_cp_.append("cp_out = cp.checkpoint(self.cp_forward, {})".format(', '.join(cp_inputs)))
            no_cp_.append("self.out = ({},)".format(', '.join(no_cp_return)))
            cp_ = ["{} = args[{}]".format(name, i) for i, name in enumerate(cp_inputs)] + cp_
            cp_.append("self.cp_out = ({},)".format(', '.join(cp_return)))

            self.cp = '\n'.join(cp_)
            self.no_cp = '\n'.join(no_cp_)

    def forward(self, *args):
        exec(self.no_cp)
        return self.out

    def cp_forward(self, *args):
        exec(self.cp)
        return self.cp_out

    # def pre_backward(self):
    #     if self.pipe:
    #         pipe_cp.pre_backward(self.ctxs.pop(0))