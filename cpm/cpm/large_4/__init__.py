import sys
sys.path.append("..")
from vpipe import Stage
from vpipe import CPM

def model(criterion, partition, recompute_ratio):
    _declares = get_declares()
    _calculations = get_caculations()
    module = CPM(_declares, _calculations)
    module.generate_layer_blocks()
    start = 0
    inputs = []
    outputs = [['out203']]
    all_outputs = []
    declares = []
    calculations = []
    for i in partition:
        stage = module.generate_stage(start, start + i)
        start += i
        declares.append(stage[0])
        calculations.append(stage[1])
        inputs.append(stage[2])
        all_outputs.append(stage[3])
    
    for i in range(len(partition)-1, 0, -1):
        previous_output = []
        for name in inputs[i]:
            if name != 'out0' and name != 'out1' and name != 'out2':
                previous_output.append(name)
        for name in outputs[0]:
            if name not in all_outputs[i] and name not in previous_output:
                previous_output.append(name)
        outputs.insert(0, previous_output)

    return [
        (Stage(inputs[0], outputs[0], declares[0], calculations[0], recompute_ratio[0]), replace(inputs[0]), outputs[0]),
        (Stage(inputs[1], outputs[1], declares[1], calculations[1], recompute_ratio[1]), replace(inputs[1]), outputs[1]),
        (Stage(inputs[2], outputs[2], declares[2], calculations[2], recompute_ratio[2]), replace(inputs[2]), outputs[2]),
        (Stage(inputs[3], outputs[3], declares[3], calculations[3], recompute_ratio[3]), replace(inputs[3]), outputs[3]),
        (criterion, outputs[3], ["loss"])
    ]

def replace(inputs):
    for i in range(len(inputs)):
        if inputs[i] == 'out0':
            inputs[i] = 'input0'
        elif inputs[i] == 'out1':
            inputs[i] = 'input1'
        elif inputs[i] == 'out2':
            inputs[i] = 'input2'
    return inputs

def get_declares():
    return '''self.layer6 = VocabParallelEmbedding(30000, 2560)
self.layer7 = torch.nn.Embedding(1024, 2560)
self.layer9 = torch.nn.Dropout(p=0.1, inplace=False)
self.layer10 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer11 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer13 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer14 = GPT2ParallelMLP(2560, 0.1)
self.layer16 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer17 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer19 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer20 = GPT2ParallelMLP(2560, 0.1)
self.layer22 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer23 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer25 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer26 = GPT2ParallelMLP(2560, 0.1)
self.layer28 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer29 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer31 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer32 = GPT2ParallelMLP(2560, 0.1)
self.layer34 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer35 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer37 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer38 = GPT2ParallelMLP(2560, 0.1)
self.layer40 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer41 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer43 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer44 = GPT2ParallelMLP(2560, 0.1)
self.layer46 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer47 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer49 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer50 = GPT2ParallelMLP(2560, 0.1)
self.layer52 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer53 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer55 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer56 = GPT2ParallelMLP(2560, 0.1)
self.layer58 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer59 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer61 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer62 = GPT2ParallelMLP(2560, 0.1)
self.layer64 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer65 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer67 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer68 = GPT2ParallelMLP(2560, 0.1)
self.layer70 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer71 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer73 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer74 = GPT2ParallelMLP(2560, 0.1)
self.layer76 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer77 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer79 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer80 = GPT2ParallelMLP(2560, 0.1)
self.layer82 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer83 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer85 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer86 = GPT2ParallelMLP(2560, 0.1)
self.layer88 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer89 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer91 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer92 = GPT2ParallelMLP(2560, 0.1)
self.layer94 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer95 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer97 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer98 = GPT2ParallelMLP(2560, 0.1)
self.layer100 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer101 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer103 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer104 = GPT2ParallelMLP(2560, 0.1)
self.layer106 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer107 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer109 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer110 = GPT2ParallelMLP(2560, 0.1)
self.layer112 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer113 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer115 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer116 = GPT2ParallelMLP(2560, 0.1)
self.layer118 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer119 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer121 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer122 = GPT2ParallelMLP(2560, 0.1)
self.layer124 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer125 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer127 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer128 = GPT2ParallelMLP(2560, 0.1)
self.layer130 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer131 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer133 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer134 = GPT2ParallelMLP(2560, 0.1)
self.layer136 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer137 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer139 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer140 = GPT2ParallelMLP(2560, 0.1)
self.layer142 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer143 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer145 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer146 = GPT2ParallelMLP(2560, 0.1)
self.layer148 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer149 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer151 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer152 = GPT2ParallelMLP(2560, 0.1)
self.layer154 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer155 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer157 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer158 = GPT2ParallelMLP(2560, 0.1)
self.layer160 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer161 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer163 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer164 = GPT2ParallelMLP(2560, 0.1)
self.layer166 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer167 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer169 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer170 = GPT2ParallelMLP(2560, 0.1)
self.layer172 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer173 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer175 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer176 = GPT2ParallelMLP(2560, 0.1)
self.layer178 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer179 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer181 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer182 = GPT2ParallelMLP(2560, 0.1)
self.layer184 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer185 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer187 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer188 = GPT2ParallelMLP(2560, 0.1)
self.layer190 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer191 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer193 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer194 = GPT2ParallelMLP(2560, 0.1)
self.layer196 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer197 = GPT2ParallelSelfAttention(2560, 32, 0.1, 0.1)
self.layer199 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer200 = GPT2ParallelMLP(2560, 0.1)
self.layer202 = FusedLayerNorm(2560, eps=1e-05, elementwise_affine=True)
self.layer203 = torch.nn.Linear(in_features=2560, out_features=30000//get_model_parallel_world_size(), bias=False)'''

def get_caculations():
    return '''out6 = self.layer6(out0)
out7 = self.layer7(out1)
out6 = out6 + out7
out9 = self.layer9(out6)
out10 = self.layer10(out9)
out11 = self.layer11(out10, out2)
out11 = out11 + out9
out13 = self.layer13(out11)
out14 = self.layer14(out13)
out14 = out14 + out11
out16 = self.layer16(out14)
out17 = self.layer17(out16, out2)
out17 = out17 + out14
out19 = self.layer19(out17)
out20 = self.layer20(out19)
out20 = out20 + out17
out22 = self.layer22(out20)
out23 = self.layer23(out22, out2)
out23 = out23 + out20
out25 = self.layer25(out23)
out26 = self.layer26(out25)
out26 = out26 + out23
out28 = self.layer28(out26)
out29 = self.layer29(out28, out2)
out29 = out29 + out26
out31 = self.layer31(out29)
out32 = self.layer32(out31)
out32 = out32 + out29
out34 = self.layer34(out32)
out35 = self.layer35(out34, out2)
out35 = out35 + out32
out37 = self.layer37(out35)
out38 = self.layer38(out37)
out38 = out38 + out35
out40 = self.layer40(out38)
out41 = self.layer41(out40, out2)
out41 = out41 + out38
out43 = self.layer43(out41)
out44 = self.layer44(out43)
out44 = out44 + out41
out46 = self.layer46(out44)
out47 = self.layer47(out46, out2)
out47 = out47 + out44
out49 = self.layer49(out47)
out50 = self.layer50(out49)
out50 = out50 + out47
out52 = self.layer52(out50)
out53 = self.layer53(out52, out2)
out53 = out53 + out50
out55 = self.layer55(out53)
out56 = self.layer56(out55)
out56 = out56 + out53
out58 = self.layer58(out56)
out59 = self.layer59(out58, out2)
out59 = out59 + out56
out61 = self.layer61(out59)
out62 = self.layer62(out61)
out62 = out62 + out59
out64 = self.layer64(out62)
out65 = self.layer65(out64, out2)
out65 = out65 + out62
out67 = self.layer67(out65)
out68 = self.layer68(out67)
out68 = out68 + out65
out70 = self.layer70(out68)
out71 = self.layer71(out70, out2)
out71 = out71 + out68
out73 = self.layer73(out71)
out74 = self.layer74(out73)
out74 = out74 + out71
out76 = self.layer76(out74)
out77 = self.layer77(out76, out2)
out77 = out77 + out74
out79 = self.layer79(out77)
out80 = self.layer80(out79)
out80 = out80 + out77
out82 = self.layer82(out80)
out83 = self.layer83(out82, out2)
out83 = out83 + out80
out85 = self.layer85(out83)
out86 = self.layer86(out85)
out86 = out86 + out83
out88 = self.layer88(out86)
out89 = self.layer89(out88, out2)
out89 = out89 + out86
out91 = self.layer91(out89)
out92 = self.layer92(out91)
out92 = out92 + out89
out94 = self.layer94(out92)
out95 = self.layer95(out94, out2)
out95 = out95 + out92
out97 = self.layer97(out95)
out98 = self.layer98(out97)
out98 = out98 + out95
out100 = self.layer100(out98)
out101 = self.layer101(out100, out2)
out101 = out101 + out98
out103 = self.layer103(out101)
out104 = self.layer104(out103)
out104 = out104 + out101
out106 = self.layer106(out104)
out107 = self.layer107(out106, out2)
out107 = out107 + out104
out109 = self.layer109(out107)
out110 = self.layer110(out109)
out110 = out110 + out107
out112 = self.layer112(out110)
out113 = self.layer113(out112, out2)
out113 = out113 + out110
out115 = self.layer115(out113)
out116 = self.layer116(out115)
out116 = out116 + out113
out118 = self.layer118(out116)
out119 = self.layer119(out118, out2)
out119 = out119 + out116
out121 = self.layer121(out119)
out122 = self.layer122(out121)
out122 = out122 + out119
out124 = self.layer124(out122)
out125 = self.layer125(out124, out2)
out125 = out125 + out122
out127 = self.layer127(out125)
out128 = self.layer128(out127)
out128 = out128 + out125
out130 = self.layer130(out128)
out131 = self.layer131(out130, out2)
out131 = out131 + out128
out133 = self.layer133(out131)
out134 = self.layer134(out133)
out134 = out134 + out131
out136 = self.layer136(out134)
out137 = self.layer137(out136, out2)
out137 = out137 + out134
out139 = self.layer139(out137)
out140 = self.layer140(out139)
out140 = out140 + out137
out142 = self.layer142(out140)
out143 = self.layer143(out142, out2)
out143 = out143 + out140
out145 = self.layer145(out143)
out146 = self.layer146(out145)
out146 = out146 + out143
out148 = self.layer148(out146)
out149 = self.layer149(out148, out2)
out149 = out149 + out146
out151 = self.layer151(out149)
out152 = self.layer152(out151)
out152 = out152 + out149
out154 = self.layer154(out152)
out155 = self.layer155(out154, out2)
out155 = out155 + out152
out157 = self.layer157(out155)
out158 = self.layer158(out157)
out158 = out158 + out155
out160 = self.layer160(out158)
out161 = self.layer161(out160, out2)
out161 = out161 + out158
out163 = self.layer163(out161)
out164 = self.layer164(out163)
out164 = out164 + out161
out166 = self.layer166(out164)
out167 = self.layer167(out166, out2)
out167 = out167 + out164
out169 = self.layer169(out167)
out170 = self.layer170(out169)
out170 = out170 + out167
out172 = self.layer172(out170)
out173 = self.layer173(out172, out2)
out173 = out173 + out170
out175 = self.layer175(out173)
out176 = self.layer176(out175)
out176 = out176 + out173
out178 = self.layer178(out176)
out179 = self.layer179(out178, out2)
out179 = out179 + out176
out181 = self.layer181(out179)
out182 = self.layer182(out181)
out182 = out182 + out179
out184 = self.layer184(out182)
out185 = self.layer185(out184, out2)
out185 = out185 + out182
out187 = self.layer187(out185)
out188 = self.layer188(out187)
out188 = out188 + out185
out190 = self.layer190(out188)
out191 = self.layer191(out190, out2)
out191 = out191 + out188
out193 = self.layer193(out191)
out194 = self.layer194(out193)
out194 = out194 + out191
out196 = self.layer196(out194)
out197 = self.layer197(out196, out2)
out197 = out197 + out194
out199 = self.layer199(out197)
out200 = self.layer200(out199)
out200 = out200 + out197
out202 = self.layer202(out200)
out203 = gather_from_model_parallel_region(self.layer203(copy_to_model_parallel_region(out202)))'''